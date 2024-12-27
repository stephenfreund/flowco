from dataclasses import dataclass, field
import json
import re
from typing import Any, Callable, List, Optional, Tuple
from flowco.dataflow.phase import Phase
import streamlit as st
from pydantic import BaseModel, Field, create_model

from flowco.builder.build import PassConfig
from flowco.builder.new_passes import (
    interactive_algorithm_assistant,
    interactive_code_assistant,
    interactive_requirements_assistant,
)
from flowco.dataflow.dfg import DataFlowGraph, Node
from flowco.dataflow.extended_type import ExtendedType
from flowco.assistant.openai import OpenAIAssistant
from flowco.page.page import Page
from flowco.ui.ui_page import UIPage
from flowco.ui.ui_util import (
    phase_for_last_shown_part,
    show_algorithm,
    show_code,
    show_requirements,
    visible_phases,
)
from flowco.util.config import config
from flowco.util.output import log
from flowco.util.text import string_diff, string_lists_diff


def default_height(text):
    return max(68, int(len(text.split("\n")) * 18 + 30))


def update_from_session_state():
    st.session_state.edit_state.update_current(
        label=st.session_state.label_current_value,
        requirements=[
            x[2:] for x in st.session_state.requirements_current_value.split("\n")
        ],
        algorithm=st.session_state.algorithm_current_value.split("\n"),
        code=st.session_state.code_current_value.split("\n"),
    )


def create_string(key: str, current: str, show_section=True):
    if show_section:
        c = st.columns(2)
        with c[0]:
            with st.container(key=f"current_{key}"):
                st.text_area(
                    key.title(),
                    current,
                    height=default_height(current),
                    key=f"{key}_current_value",
                    label_visibility="collapsed",
                    on_change=update_from_session_state,
                    placeholder=f"Enter {key} here",
                )
        with c[1]:
            with st.container(key=f"generated_{key}"):
                generated_area = st.empty()
    else:
        st.session_state[f"{key}_current_value"] = current
        generated_area = None

    return generated_area


def update_generated(generated_area, original: str, generated: str, show_diff=True):
    if generated_area != None:
        if show_diff:
            output = string_diff(
                original,
                generated,
                lambda x: f":red[~{x}~]",
                lambda x: f":blue[{x}]",
            )
        else:
            output = generated

        generated_area.write(output)


def create_list(
    key: str,
    prefix: str,
    current: List[str],
    show_section=True,
):
    text = "\n".join([f"{prefix}{x}" for x in current])
    if show_section:
        c = st.columns(2)
        with c[0]:
            with st.container(key=f"current_{key}"):
                st.text_area(
                    key.title(),
                    text,
                    height=default_height(text),
                    key=f"{key}_current_value",
                    label_visibility="collapsed",
                    on_change=update_from_session_state,
                    placeholder=f"Enter {key} here",
                )
        with c[1]:
            with st.container(key=f"generated_{key}"):
                generated_area = st.empty()
    else:
        st.session_state[f"{key}_current_value"] = text
        generated_area = None

    return generated_area


def update_generated_list(
    generated_area,
    prefix: str,
    original: List[str],
    generated: List[str],
    code=False,
    show_diff=True,
):

    def skip_markdown(tag, strikethrough=False):
        return lambda x: (
            re.sub(
                r"^(\s*(?:[-+*]|[0-9]+\.)\s+)?(.*)",
                lambda m: f"{m.group(1) or ''}:{tag}[{'~' if strikethrough else ''}{m.group(2)}{'~' if strikethrough else ''}]",
                x,
            )
            if x
            else x
        )

    if generated_area != None:
        if show_diff:
            if code:
                output = string_lists_diff(
                    original,
                    generated,
                    lambda x: f"- {x}",
                    lambda x: f"+ {x}",
                    lambda x: f"  {x}",
                )
                generated_area.code("\n".join([f"{prefix}{x}" for x in output]))
            else:
                output = string_lists_diff(
                    original,
                    generated,
                    skip_markdown("red", strikethrough=True),
                    skip_markdown("blue"),
                )
                generated_area.write("\n".join([f"{prefix}{x}" for x in output]))
        else:
            if code:
                generated_area.code("\n".join([f"{prefix}{x}" for x in generated]))
            else:
                generated_area.write("\n".join([f"{prefix}{x}" for x in generated]))


def clear_generated(generated_area):
    if generated_area != None:
        generated_area.empty()


class EditResponse(BaseModel):
    label_differences: str
    label: str = Field(
        description="A short summary of what this stage of the computation does.",
    )
    requirements_differences: str
    requirements: List[str] = Field(
        description="A list of requirements that must be true of the return value for the function.  Describe the representation of the return value as well."
    )

    function_return_type: ExtendedType = Field(
        description="The type of the return value of the function implementing this computation stage.",
    )

    function_computed_value: str = Field(
        description="A description of computed value of the function implementing this computation stage.",
    )

    algorithm_differences: str
    algorithm: List[str] = Field(
        description="The algorithm used to generate the output.  Use Markdown text to describe the algorithm.",
    )
    code_differences: str
    code: List[str] = Field(
        description="The function for this computation stage of the data flow graph, stored as a list of source lines.  The signature should match the function_name, function_parameters, and function_return_type fields.",
    )


NodePassStreamFunction = Callable[
    [PassConfig, DataFlowGraph, Node], Tuple[OpenAIAssistant, BaseModel]
]


@dataclass
class AssistantState:
    message: str
    assistant: OpenAIAssistant
    completion_model: BaseModel
    assistant_history: List[dict[str, Any]]


StepFunction = Callable[[], Tuple[OpenAIAssistant, BaseModel]]


@dataclass
class PendingStep:
    message: str
    generate: StepFunction


@dataclass
class EditState:
    original: Node
    current: Node
    generated: Node

    pending_steps: List[PendingStep] = field(default_factory=list)

    assistant_state: AssistantState | None = None

    def add_step(self, prompt: PendingStep):
        self.pending_steps.append(prompt)

    def pop_step(self) -> PendingStep:
        return self.pending_steps.pop(0)

    def has_pending_steps(self) -> bool:
        return len(self.pending_steps) > 0

    def stage_assistant(self) -> AssistantState | None:
        if self.has_pending_steps():
            step = self.pop_step()
            assistant, model = step.generate()
            return AssistantState(
                message=step.message,
                assistant=assistant,
                completion_model=model,
                assistant_history=[],
            )
        else:
            return None

    def copy_generated(self):
        self.current = self.generated

    def update_current(self, **kwargs):
        self.current = self.current.update(**kwargs)

    def first_none_field(self, model: BaseModel, response):
        for key, field in model.model_fields.items():
            if key not in response:
                return field.description
        return None

    def drain_stream(self, parts, placeholder):
        assistant_state = self.assistant_state
        if assistant_state is None:
            self.assistant_state = self.stage_assistant()
            assistant_state = self.assistant_state
        if assistant_state is not None:
            message = assistant_state.message
            with placeholder:
                with st.status(message) as status:
                    model = assistant_state.completion_model
                    stream = assistant_state.assistant.stream(model)
                    response = None
                    for response in stream:
                        if type(response) == model:
                            response = response.model_dump()

                        first_none = self.first_none_field(model, response)
                        status.update(label=f"Generating {first_none}...")

                        # remove the differences fields
                        for key in list(response.keys()):
                            if key.endswith("_differences"):
                                del response[key]
                            if key in ["id", "predecessors"]:
                                del response[key]

                        if "function_return_type" in response:
                            try:
                                response["function_return_type"] = ExtendedType(
                                    **response["function_return_type"]
                                )
                            except:
                                pass

                        self.generated = self.generated.update(**response)
                        parts.update(self)

    # The next three must use generated, not current for the current requirements,
    # algorithm, and code to be updated.  Because the lifting step leaves the
    # newly generated (and hopefully consistent) values in the generated fields.

    def gen_requirements_assistant(self) -> StepFunction:
        config = st.session_state.ui_page.page().base_build_config(False)

        return lambda: interactive_requirements_assistant(
            config,
            st.session_state.ui_page.dfg(),
            self.generated,
            self.generated.requirements or [],
        )

    def gen_algorithm_assistant(self) -> StepFunction:
        config = st.session_state.ui_page.page().base_build_config(False)
        return lambda: interactive_algorithm_assistant(
            config,
            st.session_state.ui_page.dfg(),
            self.generated,
            self.generated.algorithm or [],
        )

    def gen_code_assistant(self) -> StepFunction:
        config = st.session_state.ui_page.page().base_build_config(False)
        return lambda: interactive_code_assistant(
            config,
            st.session_state.ui_page.dfg(),
            self.generated,
            self.generated.code or [],
        )

    def gen_lift_up_changes_assistant(self):

        def lift_up_changes() -> Tuple[OpenAIAssistant, BaseModel]:
            original: Node = self.original
            current: Node = self.current

            self.generated.code = current.code or []

            if show_code():
                model = create_model(
                    "LiftUpModel",
                    code_differences=(str, Field(description="Code Differences")),
                    algorithm_differences=(
                        str,
                        Field(description="Algorithm Differences"),
                    ),
                    requirements_differences=(
                        str,
                        Field(description="Requirements Differences"),
                    ),
                    algorithm=(List[str], Field(description="Algorithm Changes")),
                    requirements=(List[str], Field(description="Requirements Changes")),
                    label=(str, Field(description="Label Changes")),
                )
                assistant = OpenAIAssistant(
                    config.model,
                    False,
                    "lift_up_changes_from_code",
                    old_code=json.dumps(original.code, indent=2),
                    new_code=json.dumps(current.code, indent=2),
                    old_algorithm=json.dumps(original.algorithm, indent=2),
                    new_algorithm=json.dumps(current.algorithm, indent=2),
                    old_requirements=json.dumps(original.requirements, indent=2),
                    new_requirements=json.dumps(current.requirements, indent=2),
                    new_label=current.label,
                )
                return assistant, model

            elif show_algorithm():
                model = create_model(
                    "LiftUpModel",
                    algorithm_differences=(
                        str,
                        Field(description="Algorithm Differences"),
                    ),
                    requirements_differences=(
                        str,
                        Field(description="Requirements Differences"),
                    ),
                    requirements=(List[str], Field(description="Requirements Changes")),
                    label=(str, Field(description="Label Changes")),
                )

                assistant = OpenAIAssistant(
                    config.model,
                    False,
                    "lift_up_changes_from_algorithm",
                    old_algorithm=json.dumps(original.algorithm, indent=2),
                    new_algorithm=json.dumps(current.algorithm, indent=2),
                    old_requirements=json.dumps(original.requirements, indent=2),
                    new_requirements=json.dumps(current.requirements, indent=2),
                    new_label=current.label,
                )
                return assistant, model

            else:
                model = create_model(
                    "LiftUpModel",
                    requirements=(List[str], Field(description="Requirements Changes")),
                    label=(str, Field(description="Label Changes")),
                )

                assistant = OpenAIAssistant(
                    config.model,
                    False,
                    "lift_up_changes_from_requirements",
                    old_requirements=json.dumps(original.requirements, indent=2),
                    new_requirements=json.dumps(current.requirements, indent=2),
                    new_label=current.label,
                )
                return assistant, model

        return lift_up_changes

    def generate_validated(self, graph: DataFlowGraph, parts, placeholder):

        parts.clear()

        self.add_step(
            PendingStep(
                message="Lifting Up Changes...",
                generate=self.gen_lift_up_changes_assistant(),
            )
        )

        if show_requirements():
            self.add_step(
                PendingStep(
                    message="Validating Requirements...",
                    generate=self.gen_requirements_assistant(),
                )
            )
        if show_algorithm():
            self.add_step(
                PendingStep(
                    message="Validating Algorithm...",
                    generate=self.gen_algorithm_assistant(),
                )
            )

        if show_code():
            self.add_step(
                PendingStep(
                    message="Validating Code...", generate=self.gen_code_assistant()
                )
            )

        self.drain_stream(parts, placeholder)

    def generate_from_command(self, graph: DataFlowGraph, parts, placeholder):

        parts.clear()

        def gen() -> Tuple[OpenAIAssistant, BaseModel]:
            current: Node = self.current

            self.generated.code = current.code or []

            if show_code():
                model = create_model(
                    "ChangeModel",
                    node_differences=(str, Field(description="Differences")),
                    label=(str, Field(description="Label")),
                    requirements=(List[str], Field(description="Requirements")),
                    description=(str, Field(description="Description")),
                    function_return_type=(
                        ExtendedType,
                        Field(description="Computed Type"),
                    ),
                    function_computed_value=(
                        str,
                        Field(description="Computed Value Desription"),
                    ),
                    algorithm=(List[str], Field(description="Algorithm")),
                    code=(List[str], Field(description="Code")),
                )
            elif show_algorithm():
                model = create_model(
                    "ChangeModel",
                    node_differences=(str, Field(description="Differences")),
                    label=(str, Field(description="Label")),
                    requirements=(List[str], Field(description="Requirements")),
                    description=(str, Field(description="Description")),
                    function_return_type=(
                        ExtendedType,
                        Field(description="Computed Type"),
                    ),
                    function_computed_value=(
                        str,
                        Field(description="Computed Value Desription"),
                    ),
                    algorithm=(List[str], Field(description="Algorithm")),
                )
            else:
                model = create_model(
                    "ChangeModel",
                    node_differences=(str, Field(description="Differences")),
                    label=(str, Field(description="Label")),
                    requirements=(List[str], Field(description="Requirements")),
                    description=(str, Field(description="Description")),
                    function_return_type=(
                        ExtendedType,
                        Field(description="Computed Type"),
                    ),
                    function_computed_value=(
                        str,
                        Field(description="Computed Value Desription"),
                    ),
                )

            assistant = OpenAIAssistant(
                config.model,
                False,
                "change_command",
                command=st.session_state.chat_command,
                current=current.model_dump_json(
                    include=set(
                        [
                            "label",
                            "parameters",
                            "preconditions",
                            "requirements",
                            "description",
                            "function_return_type",
                            "function_computed_value",
                            "function_result_var",
                        ]
                        + (["algorithm"] if show_algorithm() else [])
                        + (["code"] if show_code() else [])
                    ),
                    indent=2,
                ),
            )
            return assistant, model

        self.add_step(PendingStep(message="Making changes...", generate=gen))

        self.drain_stream(parts, placeholder)

    def field_questions(self, parts, placeholder):
        assistant_state: AssistantState | None = self.assistant_state
        if assistant_state is None:
            return
        assistant = assistant_state.assistant
        if assistant.has_questions():
            question = assistant.peek_question()
            container = st.empty()
            with container.container(border=True):
                with st.chat_message("ai"):
                    st.markdown(question)

                answer = st.chat_input("Answer")
            if answer is not None:
                if answer:
                    assistant.answer_question(answer)
                    assistant_state.assistant_history.append(
                        {"role": "ai", "content": question}
                    )
                    assistant_state.assistant_history.append(
                        {"role": "user", "content": answer}
                    )
                    container.empty()
                    if assistant.has_questions():
                        st.rerun(scope="fragment")
                    else:
                        self.drain_stream(parts, placeholder)
                        st.rerun(scope="fragment")
        else:
            self.assistant_state = None
            self.drain_stream(parts, placeholder)
            st.rerun(scope="fragment")


def all_differences(response) -> str:
    return "".join(
        [
            (
                f"* **Label**: {response.get('label_differences', '')}\n"
                if "label_differences" in response
                else ""
            ),
            (
                f"* **Requirements**: {response.get('requirements_differences', '')}\n"
                if "requirements_differences" in response
                else ""
            ),
            (
                f"* **Algorithm**: {response.get('algorithm_differences', '')}\n"
                if "algorithm_differences" in response
                else ""
            ),
            (
                f"* **Code**: {response.get('code_differences', '')}\n"
                if "code_differences" in response
                else ""
            ),
        ]
    )


class Parts:
    generated_label: Any
    generated_requirements: Any
    generated_algorithm: Any
    generated_code: Any

    def __init__(self, edit_state: EditState):
        current = edit_state.current
        c = st.columns(2)
        with c[0]:
            st.write("### Node Edits")

        with c[1]:
            st.write("### Validated")

        with st.container(border=False):
            self.generated_label = create_string(
                "label", current.label, show_section=True
            )
        with st.container(border=False):
            self.generated_requirements = create_list(
                "requirements",
                "* ",
                current.requirements or [],
                show_section=show_requirements(),
            )
        with st.container(border=False):
            self.generated_algorithm = create_list(
                "algorithm",
                "",
                current.algorithm or [],
                show_section=show_algorithm(),
            )
        with st.container(border=False):
            self.generated_code = create_list(
                "code", "", current.code or [], show_section=show_code()
            )
        self.update(edit_state=edit_state)

    def clear(self):
        clear_generated(self.generated_label)
        clear_generated(self.generated_requirements)
        clear_generated(self.generated_algorithm)
        clear_generated(self.generated_code)

    def update(
        self,
        edit_state,
    ):
        generated = edit_state.generated
        original = edit_state.original
        if generated.label is not None:
            update_generated(
                self.generated_label,
                original.label or "",
                generated.label,
            )
        if generated.requirements is not None:
            update_generated_list(
                self.generated_requirements,
                "* ",
                original.requirements or [],
                generated.requirements,
            )
        if generated.algorithm is not None:
            update_generated_list(
                self.generated_algorithm,
                "",
                original.algorithm or [],
                generated.algorithm,
            )
        if generated.code is not None:
            update_generated_list(
                self.generated_code,
                "",
                original.code or [],
                generated.code,
                code=True,
            )


def edit_node(node_id: str, edits: Optional[Node] = None):
    graph: DataFlowGraph = st.session_state.ui_page.dfg()
    node: Node = st.session_state.ui_page.node(node_id)

    if node.phase < phase_for_last_shown_part():
        pass

    if edits is None:
        edits = node.model_copy()

    st.session_state.edit_state = EditState(
        original=node,
        current=edits,
        generated=edits,
    )

    @st.dialog(node.pill, width="large")
    def edit_dialog():
        with st.container(key="edit_dialog"):

            controls = st.empty()
            edit_state: EditState = st.session_state.edit_state
            spinner_placeholder = st.empty()
            top_placeholder = st.empty()

            if st.session_state.chat_command:
                parts = Parts(edit_state)
                edit_state.generate_from_command(graph, parts, spinner_placeholder)
            elif st.session_state.generate:
                parts = Parts(edit_state)
                edit_state.generate_validated(graph, parts, spinner_placeholder)
            elif st.session_state.copy_generated:
                edit_state.copy_generated()
                parts = Parts(edit_state)
            else:
                parts = Parts(edit_state)

            assistant_state: AssistantState | None = edit_state.assistant_state
            with top_placeholder.container():
                if assistant_state is not None:
                    with st.expander("Chat History"):
                        for item in assistant_state.assistant_history:
                            with st.chat_message(item["role"]):
                                st.write(item["content"])
                    edit_state.field_questions(parts, spinner_placeholder)

            if assistant_state is None:
                with controls.container():
                    st.chat_input("Let me make your changes!", key="chat_command")
                    c = st.columns(2)
                    with c[0]:
                        st.button("Validate", key="generate", help="Ensure the changes are consistent.")

                    with c[1]:
                        with st.container(key="edit_node_commands"):
                            d = st.columns(6)
                            with d[0]:
                                st.button("< Edit Validated", key="copy_generated", help="Continue to edit the validated changes.")
                            with d[1]:
                                if st.button("Save Validated", help="Save the validated changes to the node."):
                                    apply_edit_to_dfg(edit_state)
                                    st.session_state.force_update = True
                                    st.rerun(scope="app")
                            with d[5]:
                                if st.button(":material/clear_all: Clear Contents", help="Clear all generated content from the node (except requirements)."):
                                    reset()
                                    st.session_state.force_update = True
                                    st.rerun(scope="app")



    def reset():
        ui_page: UIPage = st.session_state.ui_page
        dfg = ui_page.dfg()

        log(f"Resetting Node {node_id}")
        new_node = node.reset(reset_requirements=False)  # keep requirements...
        dfg = dfg.with_node(new_node)
        ui_page.update_dfg(dfg)



    def apply_edit_to_dfg(edit_state):
        ui_page: UIPage = st.session_state.ui_page
        new_phase = phase_for_last_shown_part()
        generated = edit_state.generated
        dfg = ui_page.dfg()

        if generated.requirements != node.requirements or generated.label != node.label:
            if generated.label != node.label:
                generated = dfg.update_node_pill(generated)

            page: Page = ui_page.page()
            page.clean(node_id)  # !!! this can change the page.dfg

        dfg = ui_page.dfg()
        new_node = generated.update(phase=new_phase)
        for phase in visible_phases():
            new_node = new_node.update(
                cache=new_node.cache.update(phase=phase, node=new_node)
            )

        log("Updating node", new_node.formatted_str())

        dfg = dfg.with_node(new_node)
        ui_page.update_dfg(dfg)

    edit_dialog()

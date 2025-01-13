from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional, Tuple

from code_editor import code_editor
import deepdiff
import streamlit as st

from flowco.builder.cache import BuildCache
from flowco.dataflow.dfg import DataFlowGraph, Node
from flowco.page.ama_node import AskMeAnythingNode
from flowco.page.page import Page
from flowco.ui.ui_page import UIPage
from flowco.ui.ui_util import (
    phase_for_last_shown_part,
    show_code,
    visible_phases,
)
from flowco.util.config import config
from flowco.util.output import log
from openai import OpenAI


@dataclass
class NodeComponents:
    label: str
    requirements: List[str]
    code: List[str]

    @staticmethod
    def from_node(node: Node):
        return NodeComponents(
            label=node.label,
            requirements=node.requirements or [],
            code=node.code or [],
        )

    def update_node(self, node: Node):
        return node.update(
            label=self.label,
            requirements=self.requirements,
            code=self.code,
        )


@dataclass
class Response:
    text: str
    is_submit: bool


@dataclass
class PendingAMA:
    prompt: str
    show_prompt: bool


class NodeEditor:
    original_node: Node
    node: Node
    pending_ama: PendingAMA | None = None
    ama: AskMeAnythingNode

    last_update: Dict[str, Dict[str, Any]] = {}

    def __init__(self, dfg: DataFlowGraph, node: Node):
        self.dfg = dfg
        self.node = node
        self.original_node = node
        self.ama = AskMeAnythingNode(dfg, show_code())

    def others(self, title):
        if show_code():
            l = {
                "Label": ["Requirements", "Code"],
                "Requirements": ["Label", "Code"],
                "Code": ["Label", "Requirements"],
            }
        else:
            l = {
                "Label": ["Requirements"],
                "Requirements": ["Label"],
            }
        return " and ".join(l[title])

    def component_editor(
        self,
        title: str,
        value: str,
        language: str,
        height: int,
        focus=False,
    ) -> Response | None:
        props = {
            "showGutter": False,
            "highlightActiveLine": False,
            "enableBasicAutocompletion": False,
            "enableLiveAutocompletion": False,
        }

        options = {
            "wrap": True,
            "showLineNumbers": False,
            "highlightActiveLine": False,
            "enableBasicAutocompletion": False,
            "enableLiveAutocompletion": False,
        }

        info_bar = {
            "name": "language info",
            "css": "",
            "style": {
                "height": "1.5rem",
                "padding": "0rem",
            },
            "info": [
                {
                    "name": title,
                    "style": {
                        "font-size": "14px",
                        "font-family": '"Source Sans Pro", sans-serif',
                    },
                }
            ],
        }

        buttons = [
            {
                #                "name": f"Synchronize {self.others(title)} to {title}",
                "name": f"{title} → {self.others(title)}",
                "feather": "RefreshCw",
                "hasText": True,
                "alwaysOn": True,
                "commands": [
                    "submit",
                ],
                "style": {"top": "1rem", "right": "0.4rem"},
            },
        ]

        response = code_editor(
            value,
            lang=language,
            key=f"editor_{title}",
            height=height,
            allow_reset=True,
            response_mode="blur",
            props=props,
            options=options,
            info=info_bar,
            buttons=buttons,
            focus=focus,
        )
        last_response = self.last_update.get(title, None)
        if (
            (last_response is None and response["id"] != "")
            or last_response is not None
            and last_response["id"] != response["id"]
        ):
            self.last_update[title] = response
            return Response(response["text"], response["type"] == "submit")
        else:
            return None

    def register_pending_ama(self, prompt: str, show_prompt: bool):
        self.pending_ama = PendingAMA(prompt, show_prompt)

    def register_pending_voice(self, container):
        voice = st.session_state.voice_input
        client = OpenAI()
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=voice,
        )
        self.pending_ama = PendingAMA(transcription.text, True)

    def node_component_editors(self) -> None:
        label_response = self.component_editor(
            "Label", self.node.label, "markdown", 1, True
        )
        if label_response is not None:
            self.node = self.node.update(label=label_response.text)
            if label_response.is_submit:
                self.pending_ama = PendingAMA(
                    f"The label has been modified.  Propagate those changes to the {self.others('Label')}.",
                    False,
                )

        requirements_md = "\n".join(f"* {x}" for x in self.node.requirements or [])
        requirements_response = self.component_editor(
            "Requirements", requirements_md, "markdown", 30
        )
        if requirements_response is not None:
            self.node = self.node.update(
                requirements=[
                    x.lstrip("* ") for x in requirements_response.text.split("\n")
                ]
            )
            if requirements_response.is_submit:
                self.pending_ama = PendingAMA(
                    f"The requirements have been modified.  Propagate those changes to the {self.others('Requirements')}.",
                    False,
                )

        if show_code():
            code_python = "\n".join(self.node.code or [])
            code_response = self.component_editor("Code", code_python, "python", 30)
            if code_response is not None:
                self.node = self.node.update(code=code_response.text.split("\n"))
                if code_response.is_submit:
                    self.pending_ama = PendingAMA(
                        f"The code has been modified.  Propagate those changes to the {self.others('Code')}.",
                        False,
                    )

    def save(self):
        ui_page: UIPage = st.session_state.ui_page
        dfg = ui_page.dfg()
        original_node = self.original_node
        node = self.node

        if (
            original_node.requirements != node.requirements
            or original_node.label != node.label
        ):
            page: Page = ui_page.page()
            page.clean(node.id)  # !!! this can change the page.dfg

        dfg = ui_page.dfg()  # must reload

        for phase in visible_phases():
            node = node.update(cache=node.cache.update(phase=phase, node=node))

        log(
            "Updating node",
            node.update(cache=BuildCache()).model_dump_json(indent=2),
        )

        dfg = dfg.with_node(node)

        if original_node.label != node.label:
            node = dfg.update_node_pill(node)
            dfg = dfg.with_node(node)

        ui_page.update_dfg(dfg)

    @st.fragment
    def edit(self):

        top = st.empty()

        with st.container(key="edit_dialog"):
            self.node_component_editors()

        print(
            self.node.model_dump_json(
                include=set(["label", "requirements", "return-type", "code"]), indent=2
            )
        )

        with top.container():
            left, right = st.columns(2)
            with left:
                if st.button("Save", disabled=self.pending_ama is not None):
                    self.save()
                    st.session_state.force_update = True
                    st.rerun(scope="app")
            with right:
                if st.button(
                    ":material/cached: Check Changes",
                    disabled=self.pending_ama is not None,
                ):
                    self.pending_ama = PendingAMA(
                        config.get_prompt(
                            (
                                "ama_node_editor_sync"
                                if show_code()
                                else "ama_node_editor_sync_no_code"
                            ),
                        ),
                        False,
                    )

            container = st.container(height=200, border=True, key="chat_container_node")
            with container:
                for message in self.ama.messages():
                    with st.chat_message(message.role):
                        st.markdown(message.content, unsafe_allow_html=True)

                if self.pending_ama:
                    pending_ama = self.pending_ama
                    if pending_ama.show_prompt:
                        with st.chat_message("user"):
                            st.markdown(self.pending_ama.prompt)
                    with st.chat_message("assistant"):
                        response = st.write_stream(
                            self.ama.complete(
                                pending_ama.prompt,
                                self.node,
                                pending_ama.show_prompt,
                            )
                        )
                    self.node = self.ama.updated_node() or self.node
                    self.pending_ama = None
                    st.rerun(scope="fragment")

            st.audio_input(
                "Record a voice message",
                label_visibility="collapsed",
                key="voice_input_node",
                on_change=lambda: self.register_pending_voice(container),
                # disabled=not self.graph_is_editable(),
            )

            with st.container(key="ama_columns_node"):
                if prompt := st.chat_input(
                    "Ask me to modify this node, or edit its components directly below.",
                    key="ama_input_node",
                    # on_submit=lambda: self.register_pending_ama(prompt),
                    # on_submit=lambda: toggle("ama_responding"),
                    # disabled=not self.graph_is_editable(),
                ):
                    self.register_pending_ama(prompt, True)
                    st.rerun(scope="fragment")

            # self.ama_completion(container, prompt)

            # if st.button(
            #     "Save Validated",
            #     help="Save the validated changes to the node.",
            # ):
            #     apply_edit_to_dfg(edit_state)
            #     st.session_state.force_update = True
            #     st.rerun(scope="app")
            # if st.button(
            #     ":material/clear_all: Clear Contents",
            #     help="Clear all generated content from the node (except requirements).",
            # ):
            #     reset()
            #     st.session_state.force_update = True
            #     st.rerun(scope="app")


def edit_node(node_id: str):
    ui_page = st.session_state.ui_page
    node: Node = ui_page.node(node_id)
    st.session_state.node_editor = NodeEditor(ui_page.dfg(), node)

    if node.phase < phase_for_last_shown_part():
        pass

    @st.dialog(node.pill, width="large")
    def edit_dialog():
        st.session_state.node_editor.edit()

    edit_dialog()


# @dataclass
# class UndoStack:
#     undo_stack: List[NodeComponents] = field(default_factory=list)
#     redo_stack: List[NodeComponents] = field(default_factory=list)

#     def push(self, components: NodeComponents) -> None:
#         if not self.undo_stack or NodeComponents != self.undo_stack[-1]:
#             log(f"Pushing to undo stack: {components}")
#             self.undo_stack.append(components)
#             self.redo_stack.clear()

#     def can_undo(self) -> bool:
#         return len(self.undo_stack) > 0

#     def can_redo(self) -> bool:
#         return len(self.redo_stack) > 0

#     def undo(self, current: NodeComponents) -> NodeComponents:
#         if self.undo_stack:
#             self.redo_stack.append(current)
#             return self.undo_stack.pop()
#         else:
#             return current

#     def redo(self, current: NodeComponents) -> NodeComponents:
#         if self.redo_stack:
#             self.undo_stack.append(current)
#             return self.redo_stack.pop()
#         else:
#             return current


# def ama_voice_input(self, container):
#     toggle("ama_responding")
#     voice = st.session_state.voice_input
#     client = OpenAI()
#     transcription = client.audio.transcriptions.create(
#         model="whisper-1",
#         file=voice,
#     )
#     self.ama_completion(container, transcription.text)

# def ama_completion(self, container, prompt):
#     page = st.session_state.ui_page.page()
#     dfg = page.dfg
#     with container:
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         empty = st.empty()
#         try:
#             with empty.chat_message("assistant"):
#                 response = st.write_stream(
#                     st.session_state.ama.complete(
#                         prompt, st.session_state.selected_node
#                     )
#                 )

#             with empty.chat_message("assistant"):
#                 st.markdown(
#                     st.session_state.ama.last_message().content,
#                     unsafe_allow_html=True,
#                 )
#         except Exception as e:
#             with empty.chat_message("assistant"):
#                 st.error(f"An error occurred in AMA: {e}")
#                 st.stop()
#         finally:
#             time.sleep(1)
#             st.session_state.ama_responding = False
#             if dfg != page.dfg:
#                 st.session_state.force_update = True
#                 self.auto_update()


#     #             st.rerun()  # TODO: This could be in a callback!

#     # def reset():
#     #     ui_page: UIPage = st.session_state.ui_page
#     #     dfg = ui_page.dfg()

#     #     log(f"Resetting Node {node_id}")
#     #     new_node = node.reset(reset_requirements=False)  # keep requirements...
#     #     dfg = dfg.with_node(new_node)
#     #     ui_page.update_dfg(dfg)

#     # def apply_edit_to_dfg(edit_state):
#     #     ui_page: UIPage = st.session_state.ui_page
#     #     new_phase = phase_for_last_shown_part()
#     #     generated = edit_state.generated
#     #     dfg = ui_page.dfg()

#     #     if generated.requirements != node.requirements or generated.label != node.label:
#     #         if generated.label != node.label:
#     #             generated = dfg.update_node_pill(generated)

#     #         page: Page = ui_page.page()
#     #         page.clean(node_id)  # !!! this can change the page.dfg

#     #     dfg = ui_page.dfg()
#     #     new_node = generated.update(phase=new_phase)
#     #     for phase in visible_phases():
#     #         new_node = new_node.update(
#     #             cache=new_node.cache.update(phase=phase, node=new_node)
#     #         )

#     #     log(
#     #         "Updating node",
#     #         new_node.update(cache=BuildCache()).model_dump_json(indent=2),
#     #     )

#     #     dfg = dfg.with_node(new_node)
#     #     ui_page.update_dfg(dfg)

#     # edit_dialog()


# def default_height(text):
#     return max(68, int(len(text.split("\n")) * 18 + 30))


# def update_from_session_state():
#     st.session_state.edit_state.update_current(
#         label=st.session_state.label_current_value,
#         requirements=[
#             x[2:] for x in st.session_state.requirements_current_value.split("\n")
#         ],
#         code=st.session_state.code_current_value.split("\n"),
#     )


# def create_string(key: str, current: str, show_section=True):
#     if show_section:
#         c = st.columns(2)
#         with c[0]:
#             with st.container(key=f"current_{key}"):
#                 st.text_area(
#                     key.title(),
#                     current,
#                     height=default_height(current),
#                     key=f"{key}_current_value",
#                     label_visibility="collapsed",
#                     on_change=update_from_session_state,
#                     placeholder=f"Enter {key} here",
#                 )
#         with c[1]:
#             with st.container(key=f"generated_{key}"):
#                 generated_area = st.empty()
#     else:
#         st.session_state[f"{key}_current_value"] = current
#         generated_area = None

#     return generated_area


# def update_generated(generated_area, original: str, generated: str, show_diff=True):
#     if generated_area != None:
#         if show_diff:
#             output = string_diff(
#                 original,
#                 generated,
#                 lambda x: f":red[~{x}~]",
#                 lambda x: f":blue[{x}]",
#             )
#         else:
#             output = generated

#         generated_area.write(output)


# def create_list(
#     key: str,
#     prefix: str,
#     current: List[str],
#     show_section=True,
# ):
#     text = "\n".join(
#         [
#             f"{prefix}{x}" if not x.startswith("- ") else f"{' ' * len(prefix)}{x}"
#             for x in current
#         ]
#     )
#     if show_section:
#         c = st.columns(2)
#         with c[0]:
#             with st.container(key=f"current_{key}"):
#                 st.text_area(
#                     key.title(),
#                     text,
#                     height=default_height(text),
#                     key=f"{key}_current_value",
#                     label_visibility="collapsed",
#                     on_change=update_from_session_state,
#                     placeholder=f"Enter {key} here",
#                 )
#         with c[1]:
#             with st.container(key=f"generated_{key}"):
#                 generated_area = st.empty()
#     else:
#         st.session_state[f"{key}_current_value"] = text
#         generated_area = None

#     return generated_area


# def update_generated_list(
#     generated_area,
#     prefix: str,
#     original: List[str],
#     generated: List[str],
#     code=False,
#     show_diff=True,
# ):

#     def skip_markdown(tag, strikethrough=False):
#         return lambda x: (
#             re.sub(
#                 r"^(\s*(?:[-+*]|[0-9]+\.)\s+)?(.*)",
#                 lambda m: f"{m.group(1) or ''}:{tag}[{'~' if strikethrough else ''}{m.group(2)}{'~' if strikethrough else ''}]",
#                 x,
#             )
#             if x
#             else x
#         )

#     if generated_area != None:
#         if show_diff:
#             if code:
#                 output = string_lists_diff(
#                     original,
#                     generated,
#                     lambda x: f"- {x}",
#                     lambda x: f"+ {x}",
#                     lambda x: f"  {x}",
#                 )
#                 generated_area.code("\n".join([f"{prefix}{x}" for x in output]))
#             else:
#                 output = string_lists_diff(
#                     original,
#                     generated,
#                     skip_markdown("red", strikethrough=True),
#                     skip_markdown("blue"),
#                 )
#                 generated_area.write("\n".join([f"{prefix}{x}" for x in output]))
#         else:
#             if code:
#                 generated_area.code("\n".join([f"{prefix}{x}" for x in generated]))
#             else:
#                 generated_area.write("\n".join([f"{prefix}{x}" for x in generated]))


# def clear_generated(generated_area):
#     if generated_area != None:
#         generated_area.empty()


# class EditResponse(BaseModel):
#     label_differences: str
#     label: str = Field(
#         description="A short summary of what this stage of the computation does.",
#     )
#     requirements_differences: str
#     requirements: List[str] = Field(
#         description="A list of requirements that must be true of the return value for the function.  Describe the representation of the return value as well."
#     )

#     function_return_type: ExtendedType = Field(
#         description="The type of the return value of the function implementing this computation stage.",
#     )

#     function_computed_value: str = Field(
#         description="A description of computed value of the function implementing this computation stage.",
#     )

#     code_differences: str
#     code: List[str] = Field(
#         description="The function for this computation stage of the data flow graph, stored as a list of source lines.  The signature should match the function_name, function_parameters, and function_return_type fields.",
#     )


# NodePassStreamFunction = Callable[
#     [PassConfig, DataFlowGraph, Node], Tuple[OpenAIAssistant, BaseModel]
# ]


# @dataclass
# class AssistantState:
#     message: str
#     assistant: OpenAIAssistant
#     completion_model: BaseModel
#     assistant_history: List[dict[str, Any]]


# StepFunction = Callable[[], Tuple[OpenAIAssistant, BaseModel]]


# @dataclass
# class PendingStep:
#     message: str
#     generate: StepFunction


# @dataclass
# class EditState:
#     original: Node
#     current: Node
#     generated: Node

#     pending_steps: List[PendingStep] = field(default_factory=list)

#     assistant_state: AssistantState | None = None

#     def add_step(self, prompt: PendingStep):
#         self.pending_steps.append(prompt)

#     def pop_step(self) -> PendingStep:
#         return self.pending_steps.pop(0)

#     def has_pending_steps(self) -> bool:
#         return len(self.pending_steps) > 0

#     def stage_assistant(self) -> AssistantState | None:
#         if self.has_pending_steps():
#             step = self.pop_step()
#             assistant, model = step.generate()
#             return AssistantState(
#                 message=step.message,
#                 assistant=assistant,
#                 completion_model=model,
#                 assistant_history=[],
#             )
#         else:
#             return None

#     def copy_generated(self):
#         self.current = self.generated

#     def update_current(self, **kwargs):
#         self.current = self.current.update(**kwargs)

#     def first_none_field(self, model: BaseModel, response):
#         for key, field in model.model_fields.items():
#             if key not in response:
#                 return field.description
#         return None

#     def drain_stream(self, parts, placeholder):
#         assistant_state = self.assistant_state
#         if assistant_state is None:
#             self.assistant_state = self.stage_assistant()
#             assistant_state = self.assistant_state
#         if assistant_state is not None:
#             message = assistant_state.message
#             with placeholder:
#                 with st.status(message) as status:
#                     model = assistant_state.completion_model
#                     stream = assistant_state.assistant.stream(model)
#                     response = None
#                     for response in stream:
#                         if type(response) == model:
#                             response = response.model_dump()

#                         first_none = self.first_none_field(model, response)
#                         status.update(label=f"Generating {first_none}...")

#                         # remove the differences fields
#                         for key in list(response.keys()):
#                             if key.endswith("_differences"):
#                                 del response[key]
#                             if key in ["id", "predecessors"]:
#                                 del response[key]

#                         if "function_return_type" in response:
#                             try:
#                                 response["function_return_type"] = ExtendedType(
#                                     **response["function_return_type"]
#                                 )
#                             except:
#                                 pass

#                         self.generated = self.generated.update(**response)
#                         parts.update(self)

#     # The next three must use generated, not current for the current requirements,
#     # algorithm, and code to be updated.  Because the lifting step leaves the
#     # newly generated (and hopefully consistent) values in the generated fields.

#     def gen_requirements_assistant(self) -> StepFunction:
#         config = st.session_state.ui_page.page().base_build_config(False)

#         return lambda: interactive_requirements_assistant(
#             config,
#             st.session_state.ui_page.dfg(),
#             self.generated,
#             self.generated.requirements or [],
#         )

#     def gen_code_assistant(self) -> StepFunction:
#         config = st.session_state.ui_page.page().base_build_config(False)
#         return lambda: interactive_code_assistant(
#             config,
#             st.session_state.ui_page.dfg(),
#             self.generated,
#             self.generated.code or [],
#         )

#     def gen_lift_up_changes_assistant(self):

#         def lift_up_changes() -> Tuple[OpenAIAssistant, BaseModel]:
#             original: Node = self.original
#             current: Node = self.current

#             self.generated.code = current.code or []

#             if show_code():
#                 model = create_model(
#                     "LiftUpModel",
#                     code_differences=(str, Field(description="Code Differences")),
#                     requirements_differences=(
#                         str,
#                         Field(description="Requirements Differences"),
#                     ),
#                     requirements=(List[str], Field(description="Requirements Changes")),
#                     label=(str, Field(description="Label Changes")),
#                 )
#                 assistant = OpenAIAssistant(
#                     config.model,
#                     False,
#                     "lift_up_changes_from_code_no_alg",
#                     old_code=json.dumps(original.code, indent=2),
#                     new_code=json.dumps(current.code, indent=2),
#                     old_requirements=json.dumps(original.requirements, indent=2),
#                     new_requirements=json.dumps(current.requirements, indent=2),
#                     new_label=current.label,
#                 )
#                 return assistant, model

#             else:
#                 model = create_model(
#                     "LiftUpModel",
#                     requirements=(List[str], Field(description="Requirements Changes")),
#                     label=(str, Field(description="Label Changes")),
#                 )

#                 assistant = OpenAIAssistant(
#                     config.model,
#                     False,
#                     "lift_up_changes_from_requirements_no_alg",
#                     old_requirements=json.dumps(original.requirements, indent=2),
#                     new_requirements=json.dumps(current.requirements, indent=2),
#                     new_label=current.label,
#                 )
#                 return assistant, model

#         return lift_up_changes

#     def generate_validated(self, graph: DataFlowGraph, parts, placeholder):

#         parts.clear()

#         self.add_step(
#             PendingStep(
#                 message="Lifting Up Changes...",
#                 generate=self.gen_lift_up_changes_assistant(),
#             )
#         )

#         if show_requirements():
#             self.add_step(
#                 PendingStep(
#                     message="Validating Requirements...",
#                     generate=self.gen_requirements_assistant(),
#                 )
#             )

#         if show_code():
#             self.add_step(
#                 PendingStep(
#                     message="Validating Code...", generate=self.gen_code_assistant()
#                 )
#             )

#         self.drain_stream(parts, placeholder)

#     def generate_from_command(self, graph: DataFlowGraph, parts, placeholder):

#         parts.clear()

#         def gen() -> Tuple[OpenAIAssistant, BaseModel]:
#             current: Node = self.current

#             self.generated.code = current.code or []

#             if show_code():
#                 model = create_model(
#                     "ChangeModel",
#                     node_differences=(str, Field(description="Differences")),
#                     label=(str, Field(description="Label")),
#                     requirements=(List[str], Field(description="Requirements")),
#                     description=(str, Field(description="Description")),
#                     function_return_type=(
#                         ExtendedType,
#                         Field(description="Computed Type"),
#                     ),
#                     function_computed_value=(
#                         str,
#                         Field(description="Computed Value Desription"),
#                     ),
#                     code=(List[str], Field(description="Code")),
#                 )
#             else:
#                 model = create_model(
#                     "ChangeModel",
#                     node_differences=(str, Field(description="Differences")),
#                     label=(str, Field(description="Label")),
#                     requirements=(List[str], Field(description="Requirements")),
#                     description=(str, Field(description="Description")),
#                     function_return_type=(
#                         ExtendedType,
#                         Field(description="Computed Type"),
#                     ),
#                     function_computed_value=(
#                         str,
#                         Field(description="Computed Value Desription"),
#                     ),
#                 )

#             assistant = OpenAIAssistant(
#                 config.model,
#                 False,
#                 "change_command_no_alg",
#                 command=st.session_state.chat_command,
#                 current=current.model_dump_json(
#                     include=set(
#                         [
#                             "label",
#                             "parameters",
#                             "preconditions",
#                             "requirements",
#                             "description",
#                             "function_return_type",
#                             "function_computed_value",
#                             "function_result_var",
#                         ]
#                         + (["code"] if show_code() else [])
#                     ),
#                     indent=2,
#                 ),
#             )
#             return assistant, model

#         self.add_step(PendingStep(message="Making changes...", generate=gen))

#         self.drain_stream(parts, placeholder)

#     def field_questions(self, parts, placeholder):
#         assistant_state: AssistantState | None = self.assistant_state
#         if assistant_state is None:
#             return
#         assistant = assistant_state.assistant
#         if assistant.has_questions():
#             question = assistant.peek_question()
#             container = st.empty()
#             with container.container(border=True):
#                 with st.chat_message("ai"):
#                     st.markdown(question)

#                 answer = st.chat_input("Answer")
#             if answer is not None:
#                 if answer:
#                     assistant.answer_question(answer)
#                     assistant_state.assistant_history.append(
#                         {"role": "ai", "content": question}
#                     )
#                     assistant_state.assistant_history.append(
#                         {"role": "user", "content": answer}
#                     )
#                     container.empty()
#                     if assistant.has_questions():
#                         st.rerun(scope="fragment")
#                     else:
#                         self.drain_stream(parts, placeholder)
#                         st.rerun(scope="fragment")
#         else:
#             self.assistant_state = None
#             self.drain_stream(parts, placeholder)
#             st.rerun(scope="fragment")


# def all_differences(response) -> str:
#     return "".join(
#         [
#             (
#                 f"* **Label**: {response.get('label_differences', '')}\n"
#                 if "label_differences" in response
#                 else ""
#             ),
#             (
#                 f"* **Requirements**: {response.get('requirements_differences', '')}\n"
#                 if "requirements_differences" in response
#                 else ""
#             ),
#             (
#                 f"* **Code**: {response.get('code_differences', '')}\n"
#                 if "code_differences" in response
#                 else ""
#             ),
#         ]
#     )


# class Parts:
#     generated_label: Any
#     generated_requirements: Any
#     generated_code: Any

#     def __init__(self, edit_state: EditState):
#         current = edit_state.current
#         c = st.columns(2)
#         with c[0]:
#             st.write("### Node Edits")

#         with c[1]:
#             st.write("### Validated")

#         with st.container(border=False):
#             self.generated_label = create_string(
#                 "label", current.label, show_section=True
#             )
#         with st.container(border=False):
#             self.generated_requirements = create_list(
#                 "requirements",
#                 "* ",
#                 current.requirements or [],
#                 show_section=show_requirements(),
#             )
#         with st.container(border=False):
#             self.generated_code = create_list(
#                 "code", "", current.code or [], show_section=show_code()
#             )
#         self.update(edit_state=edit_state)

#     def clear(self):
#         clear_generated(self.generated_label)
#         clear_generated(self.generated_requirements)
#         clear_generated(self.generated_code)

#     def update(
#         self,
#         edit_state,
#     ):
#         generated = edit_state.generated
#         original = edit_state.original
#         if generated.label is not None:
#             update_generated(
#                 self.generated_label,
#                 original.label or "",
#                 generated.label,
#             )
#         if generated.requirements is not None:
#             update_generated_list(
#                 self.generated_requirements,
#                 "* ",
#                 original.requirements or [],
#                 generated.requirements,
#             )
#         if generated.code is not None:
#             update_generated_list(
#                 self.generated_code,
#                 "",
#                 original.code or [],
#                 generated.code,
#                 code=True,
#             )

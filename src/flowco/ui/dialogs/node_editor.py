from dataclasses import dataclass
from typing import Any, Dict, List

import seaborn as sns

from code_editor import code_editor
import streamlit as st

from flowco.assistant.flowco_assistant import fast_transcription
from flowco.builder.cache import BuildCache
from flowco.builder.synthesize import algorithm, requirements, compile
from flowco.builder.synthesize_kinds import table_requirements
from flowco.dataflow.dfg import DataFlowGraph, Node, NodeKind
from flowco.dataflow.extended_type import ExtendedType, schema_to_text
from flowco.dataflow.phase import Phase
from flowco.page.ama_node import AskMeAnythingNode
from flowco.page.page import Page
from flowco.page.tables import file_path_to_table_name, table_df
from flowco.session.session_file_system import fs_glob, fs_write
from flowco.ui.ui_page import UIPage
from flowco.ui.ui_util import (
    phase_for_last_shown_part,
    show_code,
    visible_phases,
)
from flowco.util.config import config
from flowco.util.output import log, logger


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
        self.ama = AskMeAnythingNode(dfg, show_code(), node.kind == NodeKind.plot)

    def others(self, title):
        if show_code():
            l = {
                "Label": ["Requirements", "Return Type", "Code"],
                "Requirements": ["Label", "Code", "Return Type"],
                "Code": ["Label", "Requirements", "Return Type"],
            }
        else:
            l = {
                "Label": ["Requirements", "Return Type"],
                "Requirements": ["Label", "Return Type"],
            }
        fields = l[title]
        if self.node.kind == NodeKind.plot:
            fields.remove("Return Type")
        return " and ".join(fields)

    def component_editor(
        self,
        title: str,
        value: str,
        language: str,
        height: int,
        focus=False,
        prop_change_button=True,
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

        if prop_change_button:
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
        else:
            buttons = []

        response = code_editor(
            value,
            lang=language,
            key=f"editor_{title}",
            height=height,
            allow_reset=True,
            response_mode=["blur"],  # type: ignore
            props=props,
            options=options,
            info=info_bar,
            buttons=buttons,
            focus=focus,
        )
        last_response = self.last_update.get(title, None)
        if response["id"] != "" and (
            last_response is None
            or (last_response is not None and last_response["id"] != response["id"])
        ):
            print(f"Last response: {last_response}")
            print(f"Current response: {response}")
            self.last_update[title] = response
            return Response(response["text"], response["type"] == "submit")
        else:
            return None

    def register_pending_ama(self, prompt: str, show_prompt: bool):
        self.pending_ama = PendingAMA(prompt, show_prompt)

    def register_pending_voice(self, container):
        voice = st.session_state.voice_input_node
        text = fast_transcription(voice)
        self.pending_ama = PendingAMA(text, True)

    def node_component_editors(self) -> None:

        l, r = st.columns(2, vertical_alignment="center")
        with l:
            pill_response = self.component_editor(
                "Title", self.node.pill, "markdown", 1, True, False
            )
            if pill_response is not None:
                pill_text = pill_response.text
                pill_text = "".join(
                    "-" if not c.isalnum() else c for c in pill_text
                ).strip("-")

                if pill_text:
                    self.node = self.node.update(pill=pill_text)

        with r:
            with st.popover("Show Inputs"):
                for param in self.node.function_parameters:
                    st.write(f"### {param.name}")
                    extended_type = param.type
                    if extended_type is not None:
                        st.caption(extended_type.description)
                        st.code(schema_to_text(extended_type.type_schema()))

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

        with st.container(key="output_type_schema"):
            if self.node.kind is not NodeKind.plot:
                st.write("#### Output Type")
                extended_type = self.node.function_return_type
                if extended_type is not None:
                    st.caption(extended_type.description)
                    st.code(schema_to_text(extended_type.type_schema()))

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
            original_node.kind != node.kind
            or original_node.requirements != node.requirements
            or original_node.label != node.label
            or original_node.function_return_type != node.function_return_type
        ):
            page: Page = ui_page.page()
            page.clean(node.id)  # !!! this can change the page.dfg

            assert node.requirements is not None, "Requirements must be set"
            assert node.function_return_type is not None, "Return type must be set"

            if original_node.kind != node.kind:
                node = node.update(phase=Phase.clean)
            else:
                node = node.update(phase=Phase.requirements)
        elif original_node.code != node.code:
            node = node.update(phase=Phase.code)

        dfg = ui_page.dfg()  # must reload

        if original_node.kind == node.kind:
            for phase in visible_phases():
                node = node.update(cache=node.cache.update(phase=phase, node=node))

        dfg = dfg.with_node(node)  # .reduce_phases_to_below_target(node.id, node.phase)

        # gen new pill if label changed but pill did not.
        # if original_node.label != node.label and original_node.pill == node.pill:
        #     node = dfg.update_node_pill(node)
        #     dfg = dfg.with_node(node)

        ui_page.update_dfg(dfg)

    def regenerate(self):
        with st.status("Generating...", expanded=True) as status:
            cache = self.node.cache
            cache = cache.invalidate(Phase.requirements)
            cache = cache.invalidate(Phase.algorithm)

            if show_code():
                cache = cache.invalidate(Phase.code)

            node = self.node.update(cache=cache)
            page: Page = st.session_state.ui_page.page()
            dfg = page.dfg

            pass_config = page.base_build_config(False)

            st.write("Requirements")
            node = requirements(pass_config=pass_config, graph=dfg, node=node)

            # st.write("Algorithm")
            if node.phase == Phase.requirements:
                node = algorithm(pass_config=pass_config, graph=dfg, node=node)

            if show_code():
                st.write("Code")
                if node.phase == Phase.algorithm:
                    node = compile(pass_config=pass_config, graph=dfg, node=node)

            status.update(label="Done!", state="complete", expanded=False)

        if node.phase != Phase.code:
            for warning in (
                node.filter_messages(Phase.requirements)
                + node.filter_messages(Phase.algorithm)
                + node.filter_messages(Phase.code)
            ):
                st.warning(warning.message())

        self.node = node

    @st.fragment
    def edit(self):

        top = st.empty()

        kinds = {
            "Compute a value": NodeKind.compute,
            "Load a dataset": NodeKind.table,
            "Create a plot": NodeKind.plot,
        }

        def update():
            kind = kinds[st.session_state.new_node_type]
            if kind != self.node.kind:
                self.node = self.node.update(
                    kind=kind,
                    phase=Phase.clean,
                    # requirements=[],
                    # function_return_type=None,
                    # code=[],
                )

        kind_str = st.selectbox(
            "This node will:",
            options=kinds.keys(),
            index=self.node.kind,
            key="new_node_type",
            placeholder="Choose an option option",
            # label_visibility="collapsed",
            on_change=update,
        )
        assert kind_str in kinds
        kind = kinds[kind_str]

        if self.node.kind == NodeKind.table:
            error = self.table()
        else:
            error = self.compute_or_plot()

        with top.container(key="node_edit_top"):
            if error is not None:
                st.error(error)

            c = st.columns(4, vertical_alignment="bottom")
            with c[0]:
                if st.button(
                    "Save",
                    icon=":material/save:",
                    disabled=self.pending_ama is not None or error is not None,
                ):
                    with st.spinner("Saving..."):
                        self.save()
                        st.session_state.force_update = True
                        st.rerun(scope="app")
            with c[1]:
                if st.button("Cancel", icon=":material/close:"):
                    st.rerun(scope="app")

            if kind != NodeKind.table:
                with c[2]:
                    if st.button(
                        "Check Consistency",
                        icon=":material/check:",
                        disabled=self.pending_ama is not None,
                    ):
                        self.pending_ama = PendingAMA(
                            config().get_prompt(
                                (
                                    "ama_node_editor_sync"
                                    if show_code()
                                    else "ama_node_editor_sync_no_code"
                                ),
                                label=self.node.label,
                                requirements="\n".join(
                                    f"* {x}" for x in self.node.requirements or []
                                ),
                                code="\n".join(self.node.code or []),
                            ),
                            False,
                        )
                with c[3]:
                    rebuild = st.button(
                        "Regenerate",
                        icon=":material/manufacturing:",
                        disabled=self.pending_ama is not None,
                    )

                if rebuild:
                    self.regenerate()
                    st.rerun(scope="fragment")

    def table(self):
        if self.node.predecessors:
            return "Nodes that load datasets cannot have predecessors."

        uploaded_file = st.file_uploader(
            "Upload new dataset", type=["csv"], accept_multiple_files=False
        )
        if uploaded_file is not None:
            fs_write(uploaded_file.name, uploaded_file.getvalue().decode("utf-8"))
            label = uploaded_file.name
        else:
            files = [file for file in fs_glob("", "*.csv")] + sns.get_dataset_names()
            label = st.pills("Select existing dataset", files, key="new_node_file")

        if label is not None:

            df = table_df(label)
            st.write("**Preview**")
            st.dataframe(df)

            function_return_type = ExtendedType.from_value(df)
            function_return_type.description += f"The DataFrame for the {self.node.pill} dataset.  Here are the first few rows:\n```\n{df.head()}\n```\n"
            requirements = [
                f"The result is the dataframe for the `{self.node.pill}` data set.",
            ]

            self.node = self.node.update(
                pill=file_path_to_table_name(label),
                label=f"Load the `{label}` table",
                requirements=requirements,
                function_return_type=function_return_type,
            )
            return None
        else:
            return "Select a dataset."

    def compute_or_plot(self):
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
                with logger("ama updating pill"):
                    self.node = self.dfg.update_node_pill(self.node)

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
                "Ask me to make changes, fix problems, or suggest improvements. Or edit the node directly below.",
                key="ama_input_node",
            ):
                self.register_pending_ama(prompt, True)
                st.rerun(scope="fragment")

        with st.container(key="edit_dialog"):
            self.node_component_editors()

        if self.node.kind == NodeKind.plot:
            successors = self.dfg.successors(self.node.id)
            if successors:
                return "Nodes that make plots cannot have successors."

        if self.node.pill == "..." or self.node.pill == "":
            return "The node must have a title."

        if self.node.label == "":
            return "The node must have a label."

        return None


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

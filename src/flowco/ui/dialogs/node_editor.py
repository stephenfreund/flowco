from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional, Tuple

from code_editor import code_editor
import deepdiff
import streamlit as st

from flowco.builder.cache import BuildCache
from flowco.builder.synthesize import algorithm, requirements, compile
from flowco.dataflow.dfg import DataFlowGraph, Node
from flowco.dataflow.extended_type import schema_to_text
from flowco.dataflow.phase import Phase
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
                "Label": ["Requirements", "Return Type", "Code"],
                "Requirements": ["Label", "Code", "Return Type"],
                "Code": ["Label", "Requirements", "Return Type"],
            }
        else:
            l = {
                "Label": ["Requirements", "Return Type"],
                "Requirements": ["Label", "Return Type"],
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

        with st.container(key="output_type_schema"):
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

        with st.container(key="edit_dialog"):
            self.node_component_editors()

        with top.container():
            left, middle, right = st.columns(3)
            with left:
                if st.button(
                    "Save",
                    icon=":material/save:",
                    disabled=self.pending_ama is not None,
                ):
                    with st.spinner("Saving..."):
                        self.save()
                        st.session_state.force_update = True
                        st.rerun(scope="app")
            with middle:
                if st.button(
                    "Check",
                    icon=":material/check:",
                    disabled=self.pending_ama is not None,
                ):
                    self.pending_ama = PendingAMA(
                        config.get_prompt(
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
            with right:
                rebuild = st.button(
                    "Regenerate",
                    icon=":material/manufacturing:",
                    disabled=self.pending_ama is not None,
                )

            if rebuild:
                self.regenerate()
                st.rerun(scope="fragment")

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
                    "Ask me to make changes, fix problems, or suggest improvements. Or edit the node directly below.",
                    key="ama_input_node",
                ):
                    self.register_pending_ama(prompt, True)
                    st.rerun(scope="fragment")


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

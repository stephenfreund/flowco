import time
from openai import OpenAI
import uuid

from flowco.page.ama import AskMeAnything
from flowco.ui.authenticate import sign_out
from flowco.ui.dialogs.edit_node import edit_node
import numpy as np
import pandas as pd


from flowco.dataflow import dfg_update
from flowco.dataflow.dfg import Node
from flowco.dataflow.phase import Phase
from flowco.page.output import OutputType
from flowco.ui.ui_dialogs import settings
from flowco.ui.ui_init import st_abstraction_level
from flowco.ui.ui_util import (
    show_algorithm,
    show_code,
    show_requirements,
    toggle,
)
import streamlit as st

from mxgraph_component import mxgraph_component

from flowco import __main__
from flowco.ui.ui_page import UIPage
from flowco.util.config import config
from flowco.util.costs import total_cost
from flowco.util.config import AbstractionLevel


class FlowcoPage:

    def sidebar(self):
        ui_page: UIPage = st.session_state.ui_page
        node_id: str | None = st.session_state.selected_node

        if node_id is not None:
            possible_edge = ui_page.dfg().get_edge(node_id)
            if possible_edge is not None:
                # for edges, just show the source, which computes that edge value
                node_id = possible_edge.src

        if node_id is not None:
            node = ui_page.node(node_id)
        else:
            node = None

        with st.container(key="masthead"):
            self.masthead(node)
            self.button_bar()

        # st.select_slider(
        with st.container(key="controls"):
            c = st.columns(2)
            with c[0]:

                def fix():
                    if st.session_state.abstraction_level is None:
                        st.session_state.abstraction_level = "Requirements"

                st.segmented_control(
                    "Abstraction Level",
                    AbstractionLevel,
                    key="abstraction_level",
                    on_change=fix,
                    disabled=not self.graph_is_editable(),
                )
            with c[1]:
                st.pills(
                    "Show:",
                    ["Output"]
                    + (["Description"] if not config.x_no_descriptions else [])
                    + ["AMA"],
                    default=(["Output"] if st.session_state.show_output else [])
                    + (["Description"] if st.session_state.show_description else [])
                    + (["AMA"] if st.session_state.show_ama else []),
                    selection_mode="multi",
                    key="show_pills",
                    disabled=not self.graph_is_editable()
                )

        self.show_ama(node)

        if node is not None:
            self.node_sidebar(node)
        else:
            self.global_sidebar()

    def masthead(self, node: Node | None = None):
        if node is None:
            ui_page: UIPage = st.session_state.ui_page
            st.title(ui_page.page().file_name)
            st.caption(f"Total cost: {total_cost():.2f} USD")
        else:
            st.title(node.pill)
            st.caption(f"Status: {node.phase}")

    def show_messages(self, node: Node):
        for message in node.messages:
            if message.level == "error":
                st.error(message.text)
        for message in node.messages:
            if message.level == "warning":
                st.warning(message.text)
        for message in node.messages:
            if message.level == "info":
                st.success(message.text)

    def node_sidebar(self, node: Node):

        self.show_messages(node)

        self.show_node_details(node)

    def show_ama(self, node: Node):
        page = st.session_state.ui_page.page()
        if st.session_state.ama is None or st.session_state.ama.page != page:
            st.session_state.ama = AskMeAnything(page)

        with st.container():
            height = 400 if "AMA" in st.session_state.show_pills or st.session_state.ama_responding else 1
            container = st.container(height=height, border=True, key="chat_container")
            with container:
                for message in st.session_state.ama.messages():
                    with st.chat_message(message.role):
                        st.markdown(message.content, unsafe_allow_html=True)

            if "AMA" in st.session_state.show_pills:
                st.audio_input(
                    "Record a voice message",
                    label_visibility="collapsed",
                    key="voice_input",
                    on_change=lambda: self.ama_voice_input(container),
                    disabled=not self.graph_is_editable(),
                )

            if len(st.session_state.ama) > 0 and "AMA" not in st.session_state.show_pills:
                ama_prompt = "Ask Me Anything!  Click AMA to see our conversation."
            else:
                ama_prompt = "Ask Me Anything!"

            with st.container(key="ama_columns"):
                if prompt := st.chat_input(
                    ama_prompt,
                    key="ama_input",
                    on_submit=lambda: toggle("ama_responding"),
                    disabled=not self.graph_is_editable(),
                ):
                    self.ama_completion(container, prompt)

    def ama_voice_input(self, container):
        toggle("ama_responding")
        voice = st.session_state.voice_input
        client = OpenAI()
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=voice,
        )
        self.ama_completion(container, transcription.text)

    def ama_completion(self, container, prompt):
        page = st.session_state.ui_page.page()
        dfg = page.dfg
        try:
            with container:
                with st.chat_message("user"):
                    st.markdown(prompt)

                empty = st.empty()
                with empty.chat_message("assistant"):
                    response = st.write_stream(
                        st.session_state.ama.complete(
                            prompt, st.session_state.selected_node
                        )
                    )

                with empty.chat_message("assistant"):
                    st.markdown(
                        st.session_state.ama.last_message().content,
                        unsafe_allow_html=True,
                    )
        except Exception as e:
            with empty.chat_message("assistant"):
                st.error(f"An error occurred in AMA: {e}")
                st.stop()
        finally:
            time.sleep(1)
            st.session_state.ama_responding = False
            if dfg != page.dfg:
                st.session_state.force_update = True
                self.auto_update()
            st.rerun()

    def auto_update(self):
        pass

    def show_node_details(self, node):
        with st.container(key="node_sidepanel"):
            with st.container(key="node_output"):
                if "Output" in st.session_state.show_pills and node is not None:
                    st.write("#### Output")
                    if (
                        node.function_return_type is not None
                        and not node.function_return_type.is_None_type()
                    ):
                        st.caption(
                            f"{node.function_return_type.description}"
                        )
                    self.show_output(node)

            with st.container(key="node_description"):
                if "Description" in st.session_state.show_pills and node is not None:
                    if show_code():
                        st.write(f"**Return Type:** `{node.function_return_type}`")

                    st.write("#### Description")
                    description = (
                        node.description if node.description is not None else ""
                    )
                    function_computed_value = (
                        node.function_computed_value
                        if node.function_computed_value is not None
                        else ""
                    )
                    st.write(description + "\n\n" + function_computed_value)

            with st.container(key="node_requirements"):
                if show_requirements() and node is not None:
                    st.write("#### Requirements")
                    st.write(
                        "\n".join(
                            [
                                "* " + x
                                for x in (
                                    node.requirements
                                    if node.requirements is not None
                                    else []
                                )
                            ]
                        )
                    )

            with st.container(key="node_algorithm"):
                if show_algorithm() and node is not None:
                    st.write("#### Algorithm")
                    st.write(
                        "\n".join(node.algorithm) if node.algorithm is not None else ""
                    )

            with st.container(key="node_code"):
                if show_code() and node is not None:
                    st.write("#### Code")
                    if node.code is not None:
                        st.code("\n".join(node.code))

    def show_output(self, node: Node):
        if node is not None and node.result is not None:
            if (
                node.result.result is not None
                and node.function_return_type is not None
                and not node.function_return_type.is_None_type()
            ):
                value = node.result.result.to_value()
                if type(value) == np.ndarray or type(value) == list:
                    value = pd.DataFrame(value)
                st.write(value)
            elif node.result.output is not None:
                output = node.result.output
                if output is not None:
                    if output.output_type == OutputType.text:
                        st.write(output.data)
                    elif output.output_type == OutputType.image:
                        base64encoded = output.data.split(",", maxsplit=1)
                        image_data = base64encoded[0] + ";base64," + base64encoded[1]
                        st.image(image_data)

    def bottom_bar(self):
        ui_page: UIPage = st.session_state.ui_page
        with st.container(key="bottom_bar"):
            cols = st.columns(4)
            with cols[0]:
                if st.button(":material/settings:", help="Change settings", 
                                                 disabled=not self.graph_is_editable()
):
                    settings(ui_page)
            with cols[3]:
                if st.button("Logout"):
                    sign_out()
                    st.rerun()

    def button_bar(self):
        pass

    def global_sidebar(self):
        pass

    def graph_is_editable(self) -> bool:
        return not st.session_state.ama_responding

    def update_ui_page(self, update: dfg_update.mxDiagramUpdate):
        ui_page: UIPage = st.session_state.ui_page
        new_dfg = dfg_update.update_dataflow_graph(ui_page.dfg(), update)

        if new_dfg != ui_page.dfg():
            ui_page.update_dfg(new_dfg)

    def init(self):
        pass

    def fini(self):
        pass

    def edit_node(self, node_id: str):
        edit_node(node_id)

    def refresh_phase(self) -> Phase:
        level = st_abstraction_level()
        if level == "Requirements":
            return Phase.requirements
        elif level == "Algorithm":
            return Phase.algorithm
        elif level == "Code":
            return Phase.code
        else:
            return Phase.run_checked



    def main(self):

        self.init()

        # Could be <<<<<< or node id...
        selected_node = st.session_state.selected_node

        result = mxgraph_component(
            key=st.session_state.nonce,
            diagram=st.session_state.ui_page.dfg_as_mx_diagram().model_dump(),
            editable=self.graph_is_editable(),
            selected_node=selected_node,
            dummy=uuid.uuid4().hex if st.session_state.force_update else None,
            refresh_phase=self.refresh_phase().value,
            clear=st.session_state.clear_graph,
        )  # type: ignore

        if not st.session_state.force_update:
            self.update_ui_page(
                dfg_update.mxDiagramUpdate.model_validate_json(result["diagram"])
            )

        st.session_state.force_update = False
        st.session_state.clear_graph = False

        if result["command"] == "edit":
            if st.session_state.last_sequence_number != result["sequence_number"]:
                st.session_state.last_sequence_number = result["sequence_number"]
                self.edit_node(result["selected_node"])
        else:
            if selected_node == "<<<<<":
                st.session_state.selected_node = None
            else:
                st.session_state.selected_node = result["selected_node"]

        with st.sidebar:
            self.sidebar()
            st.divider()
            self.bottom_bar()

        self.fini()

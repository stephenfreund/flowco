import time
from typing import List, Tuple
from openai import OpenAI
import uuid

from streamlit_js_eval import streamlit_js_eval

from flowco.page.ama import AskMeAnything
from flowco.ui.authenticate import sign_out
import numpy as np
import pandas as pd


from flowco.dataflow import dfg_update
from flowco.dataflow.dfg import Node
from flowco.dataflow.phase import Phase
from flowco.page.output import OutputType
from flowco.ui.ui_dialogs import settings
from flowco.ui.ui_page import st_abstraction_level
from flowco.ui.ui_util import (
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
from flowco.util.output import log


if config.x_algorithm_phase:
    from flowco.ui.dialogs.edit_node import edit_node
else:
    from flowco.ui.dialogs.edit_node_no_alg import edit_node


class FlowcoPage:

    # override and call super in subclasses
    def pills(self) -> List[Tuple[str, bool]]:

        if config.x_algorithm_phase:
            pill_list = []
            if st.session_state.show_ama:
                pill_list.append(("AMA", True))
            if not config.x_no_descriptions:
                pill_list.append(("Description", True))
            if st.session_state.show_output:
                pill_list.append(("Output", True))
            p = pill_list
        else:
            p = [("AMA", True), ("Output", True), ("Code", False)]

        return p

    # override and call super in subclasses
    def node_fields_to_show(self) -> List[str]:
        fields = ["pill", "label", "requirements", "result", "messages"]
        if config.x_algorithm_phase:
            fields.append("algorithm")

        if (
            "show_pills" in st.session_state
            and "Code" in st.session_state.show_pills
            or "Code" in [pill for pill, show in self.pills() if show]
        ):
            fields.append("code")

        return fields

    def sidebar(self):
        with st.container(key="masthead"):
            self.masthead(node)
            self.button_bar()

        # st.select_slider(
        with st.container(key="controls"):
            pills = self.pills()
            all_pills = [pill for pill, _ in pills]
            default_pills = [pill for pill, show in pills if show]
            if config.x_algorithm_phase:
                c = st.columns(2)
                with c[0]:

                    def fix():
                        if st.session_state.abstraction_level is None:
                            st.session_state.abstraction_level = "Requirements"
                        st.session_state.force_update = True

                    st.segmented_control(
                        "Abstraction Level",
                        (
                            AbstractionLevel
                            if config.x_algorithm_phase
                            else [AbstractionLevel.spec, AbstractionLevel.code]
                        ),
                        key="abstraction_level",
                        on_change=fix,
                        disabled=not self.graph_is_editable(),
                    )
                with c[1]:
                    st.pills(
                        "Show:",
                        all_pills,
                        default=default_pills,
                        selection_mode="multi",
                        key="show_pills",
                        disabled=not self.graph_is_editable(),
                    )

            else:

                def fix():
                    if "Code" in st.session_state.show_pills:
                        st.session_state.abstraction_level = AbstractionLevel.code
                    else:
                        st.session_state.abstraction_level = AbstractionLevel.spec
                    st.session_state.force_update = True

                st.pills(
                    "Show:",
                    all_pills,
                    default=default_pills,
                    selection_mode="multi",
                    key="show_pills",
                    disabled=not self.graph_is_editable(),
                    on_change=fix,
                    label_visibility="collapsed",
                )

        self.show_ama(node)
        self.global_sidebar()

        # if node is not None:
        #     self.show_messages(node)

        # self.show_ama(node)

        # if node is not None:
        #     self.node_sidebar(node)
        # else:
        #     self.global_sidebar()

    def right_panel(self):
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

        with st.container(key="right-panel"):
            st.write("")
            st.write("")
            st.write("")
            if node is not None:
                st.title(node.pill)
                st.caption(f"Status: {node.phase}")

                self.show_messages(node)
                self.node_sidebar(node)

    def masthead(self, node: Node | None = None):
        if node is None:
            ui_page: UIPage = st.session_state.ui_page
            st.title(ui_page.page().file_name)
            st.caption(f"Total cost: {total_cost():.2f} USD")

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

        with st.container(key="node_sidepanel"):
            self.show_node_details(node)

    def show_ama(self, node: Node | None):
        page = st.session_state.ui_page.page()
        if st.session_state.ama is None or st.session_state.ama.page != page:
            st.session_state.ama = AskMeAnything(page)

        if "AMA" in st.session_state.show_pills:
            with st.container():
                height = 400
                container = st.container(
                    height=height, border=True, key="chat_container"
                )
                with container:
                    for message in st.session_state.ama.messages():
                        with st.chat_message(message.role):
                            st.markdown(message.content, unsafe_allow_html=True)

                if "AMA" in st.session_state.show_pills:
                    if st.audio_input(
                        "Record a voice message",
                        label_visibility="collapsed",
                        key="voice_input",
                        on_change=lambda: self.ama_voice_input(container),
                        disabled=not self.graph_is_editable(),
                    ):
                        st.rerun()

                with st.container(key="ama_columns"):
                    if prompt := st.chat_input(
                        "Ask Me Anything!",
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
        with container:
            with st.chat_message("user"):
                st.markdown(prompt)

            empty = st.empty()
            try:
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
                st.rerun()  # TODO: This could be in a callback!

    def auto_update(self):
        pass

    # override and call super in subclasses
    def show_node_details(self, node):
        if "Output" in st.session_state.show_pills and node is not None:
            st.write("#### Output")
            if (
                node.function_return_type is not None
                and not node.function_return_type.is_None_type()
            ):
                st.caption(f"{node.function_return_type.description}")
            self.show_output(node)

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
                if type(value) == pd.DataFrame:
                    st.dataframe(value, height=200)
                else:
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
        if st.button(
            ":material/settings: Settings",
            help="Change settings",
            disabled=not self.graph_is_editable(),
        ):
            settings(ui_page)

        if st.button(":material/logout: Logout", help="Sign out"):
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

        if st.session_state.ui_page is not None:

            left, right = st.columns([4, 1])

            with left:
                result = mxgraph_component(
                    key=st.session_state.nonce,
                    diagram=st.session_state.ui_page.dfg_as_mx_diagram(
                        self.node_fields_to_show()
                    ).model_dump(),
                    editable=self.graph_is_editable(),
                    selected_node=selected_node,
                    dummy=uuid.uuid4().hex if st.session_state.force_update else None,
                    refresh_phase=self.refresh_phase().value,
                    clear=st.session_state.clear_graph,
                )

                if not st.session_state.force_update:
                    self.update_ui_page(
                        dfg_update.mxDiagramUpdate.model_validate_json(
                            result["diagram"]
                        )
                    )

                st.session_state.force_update = False
                st.session_state.clear_graph = False

                if result["command"] == "edit":
                    if (
                        st.session_state.last_sequence_number
                        != result["sequence_number"]
                    ):
                        st.session_state.last_sequence_number = result[
                            "sequence_number"
                        ]
                        self.edit_node(result["selected_node"])
                else:
                    if selected_node == "<<<<<":
                        st.session_state.selected_node = None
                    else:
                        st.session_state.selected_node = result["selected_node"]

            with right:
                self.right_panel()

        with st.sidebar:
            self.sidebar()
            st.divider()
            self.bottom_bar()

        self.fini()

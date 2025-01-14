import json
from pprint import pformat
import time
from flowco.session.session import session
from openai import OpenAI
import uuid

from flowco.page.ama import AskMeAnything
from flowco.session.session_file_system import fs_write
from flowco.ui import ui_help
from flowco.ui import ui_page
from flowco.ui.authenticate import sign_out
import numpy as np
import pandas as pd


from flowco.dataflow import dfg_update
from flowco.dataflow.dfg import Node
from flowco.dataflow.phase import Phase
from flowco.page.output import OutputType
from flowco.ui.dialogs.node_editor import edit_node
from flowco.ui.ui_dialogs import settings
from flowco.ui.ui_page import st_abstraction_level
from flowco.ui.ui_util import (
    toggle,
)
import streamlit as st

from mxgraph_component import mxgraph_component

from flowco import __main__
from flowco.ui.ui_page import UIPage
from flowco.util.config import config
from flowco.util.costs import total_cost
from flowco.util.config import AbstractionLevel
from flowco.util.files import create_zip_in_memory
from flowco.util.output import Output, error, log_timestamp


class FlowcoPage:

    def sidebar(self, node: Node | None = None):
        with st.container(key="masthead"):
            self.masthead()
            self.button_bar()

            def fix():
                if st.session_state.al is None:
                    st.session_state.abstraction_level = "Requirements"
                else:
                    st.session_state.abstraction_level = st.session_state.al
                st.session_state.force_update = True

            with st.container(key="controls"):
                st.session_state.abstraction_level = st.segmented_control(
                    "Abstraction Level",
                    (
                        AbstractionLevel
                        if config.x_algorithm_phase
                        else [AbstractionLevel.spec, AbstractionLevel.code]
                    ),
                    key="al",
                    default=st.session_state.abstraction_level,
                    on_change=fix,
                    disabled=not self.graph_is_editable(),
                )

        self.show_ama()

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

        st.write("")
        st.write("")
        st.write("")
        with st.container(key="right-panel", border=True):
            symbol = (
                ":material/chevron_right:"
                if st.session_state.wide_right_panel
                else ":material/chevron_left:"
            )
            if st.button(symbol, key="right_panel_width"):
                st.session_state.wide_right_panel = (
                    not st.session_state.wide_right_panel
                )
                st.rerun()
            if node is None:
                self.global_sidebar()
            else:
                self.node_header(node)
                self.show_messages(node)
                self.node_sidebar(node)

    def node_header(self, node):
        ui_page: UIPage = st.session_state.ui_page
        with st.container(key="node_header"):
            with st.container(key="lock"):
                left, right = st.columns([1, 8], vertical_alignment="bottom")
                with left:
                    pressed = st.segmented_control(
                        ":material/lock:",
                        [":material/lock:"],
                        default=[":material/lock:"] if node.is_locked else None,
                        label_visibility="collapsed",
                        disabled=not self.graph_is_editable(),
                    )
                    if pressed and not node.is_locked:
                        dfg = ui_page.dfg()
                        ui_page.update_dfg(
                            dfg.with_node(dfg[node.id].update(is_locked=True))
                        )
                        st.session_state.force_update = True
                        st.rerun()
                    elif not pressed and node.is_locked:
                        dfg = ui_page.dfg()
                        ui_page.update_dfg(
                            dfg.with_node(
                                dfg[node.id].update(is_locked=False, phase=Phase.clean)
                            )
                        )
                        st.session_state.force_update = True
                        st.rerun()

                with right:
                    st.subheader(node.pill)
            st.caption(f"Status: {node.phase}")

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

    def show_ama(self):
        page = st.session_state.ui_page.page()
        if st.session_state.ama is None or st.session_state.ama.page != page:
            st.session_state.ama = AskMeAnything(page)

        with st.container():
            height = 400
            container = st.container(height=height, border=True, key="chat_container")
            with container:
                for message in st.session_state.ama.messages():
                    with st.chat_message(message.role):
                        st.markdown(message.content, unsafe_allow_html=True)

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
                    error(e)
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
        st.write("**Output**")
        if (
            node.function_return_type is not None
            and not node.function_return_type.is_None_type()
        ):
            st.caption(f"{node.function_return_type.description}")
        self.show_output(node)

        st.write("**Requirements**")
        with st.container(key="node_requirements"):
            st.write(
                "\n".join(
                    [
                        "* " + x
                        for x in (
                            node.requirements if node.requirements is not None else []
                        )
                    ]
                )
            )

        if AbstractionLevel.show_code(st.session_state.abstraction_level):
            st.write("**Code**")
            with st.container(key="node_code"):
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
        cols = st.columns([2, 2.1, 3])
        with cols[0]:
            if st.button(":material/help: Help", disabled=not self.graph_is_editable()):
                ui_help.help_dialog()

        with cols[1]:
            if st.button(
                ":material/settings: Settings",
                help="Change settings",
                disabled=not self.graph_is_editable(),
            ):
                settings(ui_page)

        with cols[2]:
            if st.button(":material/bug_report: Report Bug", key="report_bug"):
                self.report_bug()

        if st.button(":material/logout: Logout", help="Sign out"):
            sign_out()
            st.rerun()

    def report_bug(self):
        ui_page = st.session_state.ui_page
        flowco_name = ui_page.page().file_name
        data_files = [ file for file in ui_page.page().tables.all_files() if file.endswith(".csv") ]
        time_stamp = log_timestamp()
        file_name = f"flowco-{time_stamp}.zip"

        @st.dialog("Report Bug", width="small")
        def download_files():

            text = st.text_input(
                "Bug",
                placeholder = "Enter a description of the issue",
            )

            if text:
                with st.spinner("Creating ZIP file..."):
                    zip_data = create_zip_in_memory(
                        [flowco_name] + data_files,
                        additional_entries={
                            "description.txt": text,
                            "logging.txt": session.get("output", Output).get_full_output(),
                            "session_state.json" : pformat(dict(st.session_state))
                        },
                    )
                    fs_write(file_name, zip_data, "wb")

                st.write("ZIP ready for download!")

                if st.download_button(
                    label=":material/download:",
                    data=zip_data,
                    file_name=file_name,  # with timestamp
                    help="Download the project and log",
                ):
                    st.rerun()

        download_files()        

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

    def prepare_node_for_edit(self, node_id: str):
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

    def main_columns(self):
        if st.session_state.wide_right_panel:
            return st.columns([3, 2])
        else:
            return st.columns([4, 1])

    def main(self):

        self.init()

        # Could be <<<<<< or node id...
        selected_node = st.session_state.selected_node

        if st.session_state.ui_page is not None:

            left, right = self.main_columns()

            with left:
                result = mxgraph_component(
                    key=st.session_state.nonce,
                    diagram=st.session_state.ui_page.dfg_as_mx_diagram().model_dump(),
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
                        self.prepare_node_for_edit(result["selected_node"])
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

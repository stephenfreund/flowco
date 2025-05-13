from uuid import uuid4
from flowco.util.text import pill_to_function_name, pill_to_result_var_name
from streamlit_flow.state import StreamlitFlowState
from typing import List
from flowco.dataflow.extended_type import schema_to_text

from streamlit_flow import streamlit_flow
from streamlit_flow.layouts import LayeredLayout, ManualLayout

from flowco.page.ama import AskMeAnything, VisibleMessage
from flowco.ui.authenticate import sign_out
import numpy as np
import pandas as pd


from flowco.dataflow import dfg_update
from flowco.dataflow.dfg import Geometry, Node, NodeKind, NodeMessage
from flowco.dataflow.phase import Phase
from flowco.page.output import OutputType
from flowco.ui.dialogs.node_creator import new_node_dialog
from flowco.ui.dialogs.node_editor import edit_node
from flowco.ui.flow_diagram import diff_state, update_dfg, update_state
from flowco.ui.ui_dialogs import inspect_node, settings
from flowco.ui.ui_page import st_abstraction_level
from flowco.ui.ui_util import (
    show_code,
    toggle,
    zip_bug,
)
import streamlit as st


from flowco import __main__
from flowco.ui.ui_page import UIPage
from flowco.util.config import config
from flowco.util.costs import inflight, total_cost
from flowco.util.config import AbstractionLevel
from flowco.util.output import error, log
from flowco.llm.assistant import AssistantError
from flowco.assistant.flowco_assistant import fast_transcription


class FlowcoPage:

    def sidebar(self, node: Node | None = None):
        with st.container(key="masthead"):
            self.masthead()
            self.button_bar()
            self.second_bar()

        try:
            self.show_ama()
        except AssistantError as e:
            error(e)
            st.error(e.message)

    def second_bar(self):

        def fix():
            if st.session_state.show_code:
                st.session_state.abstraction_level = AbstractionLevel.code
            else:
                st.session_state.abstraction_level = AbstractionLevel.spec
            st.session_state.force_update = True

        with st.container(key="show_code_bar"):
            st.write("")
            st.toggle(
                "Show Code",
                value=show_code(),
                key="show_code",
                on_change=fix,
                disabled=not self.graph_is_editable(),
            )

            # st.session_state.abstraction_level = st.segmented_control(
            #     "Abstraction Level",
            #     (
            #         AbstractionLevel
            #         if config().x_algorithm_phase
            #         else [AbstractionLevel.spec, AbstractionLevel.code]
            #     ),
            #     key="al",
            #     default=st.session_state.abstraction_level,
            #     on_change=fix,
            #     disabled=not self.graph_is_editable(),
            # )

        # with st.container(key="zoom_button_bar"):
        #     c1, spacer, c2, c3, c4 = st.columns(5, vertical_alignment="bottom")
        #     with c1.container(key="controls"):
        #         st.session_state.abstraction_level = st.segmented_control(
        #             "Abstraction Level",
        #             (
        #                 AbstractionLevel
        #                 if config().x_algorithm_phase
        #                 else [AbstractionLevel.spec, AbstractionLevel.code]
        #             ),
        #             key="al",
        #             default=st.session_state.abstraction_level,
        #             on_change=fix,
        #             disabled=not self.graph_is_editable(),
        #         )

        #     def zoom(cmd):
        #         log("Zoom", cmd)
        #         st.session_state.zoom = cmd
        #         st.session_state.force_update = True

        #     spacer.write(
        #         "<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>",
        #         unsafe_allow_html=True,
        #     )

        #     c2.button(
        #         label="",
        #         icon=":material/zoom_in:",
        #         key="zoom_in",
        #         disabled=not self.graph_is_editable(),
        #         on_click=lambda cmd: zoom(cmd),
        #         args=("in",),
        #         help="Zoom in",
        #     )
        #     c3.button(
        #         label="",
        #         icon=":material/zoom_out:",
        #         disabled=not self.graph_is_editable(),
        #         on_click=lambda cmd: zoom(cmd),
        #         args=("out",),
        #         help="Zoom out",
        #     )
        #     c4.button(
        #         label="",
        #         icon=":material/zoom_out_map:",
        #         disabled=not self.graph_is_editable(),
        #         on_click=lambda cmd: zoom(cmd),
        #         args=("reset",),
        #         help="Reset zoom",
        #     )

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
            with st.container(key="right-panel-size-button"):
                if st.button(label="", icon=symbol, key="right_panel_width"):
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
            # with st.container(key="lock"):
            # left, right = st.columns([1, 8], vertical_alignment="bottom")
            # with left:
            #     pressed = st.segmented_control(
            #         ":material/lock:",
            #         [":material/lock:"],
            #         default=[":material/lock:"] if node.is_locked else None,
            #         label_visibility="collapsed",
            #         disabled=not self.graph_is_editable(),
            #     )
            #     if pressed and not node.is_locked:
            #         dfg = ui_page.dfg()
            #         ui_page.update_dfg(
            #             dfg.with_node(dfg[node.id].update(is_locked=True))
            #         )
            #         st.session_state.force_update = True
            #         st.rerun()
            #     elif not pressed and node.is_locked:
            #         dfg = ui_page.dfg()
            #         ui_page.update_dfg(
            #             dfg.with_node(
            #                 dfg[node.id].update(is_locked=False, phase=Phase.clean)
            #             )
            #         )
            #         st.session_state.force_update = True
            #         st.rerun()

            # with right:
            st.subheader(node.pill)
            st.caption(f"Status: {node.phase}")

    def masthead(self, node: Node | None = None):
        if node is None:
            ui_page: UIPage = st.session_state.ui_page
            st.title(ui_page.page().file_name)
            st.caption(
                f"Total cost: {total_cost():.2f} USD &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;        {':gray[:material/bigtop_updates:]' * inflight()}"
            )

    def filter_messages(self, node: Node) -> List[NodeMessage]:
        return node.messages

    def show_messages(self, node: Node):
        messages = self.filter_messages(node)
        for message in messages:
            if message.level == "error":
                st.error(message.text)
        for message in messages:
            if message.level == "warning":
                st.warning(message.text)
        for message in messages:
            if message.level == "info":
                st.success(message.text)

    def node_sidebar(self, node: Node):

        with st.container(key="node_sidepanel"):
            self.show_node_details(node)

    def write_ama_message(self, message: VisibleMessage):
        with st.chat_message(message.role):
            if not message.is_error:
                st.markdown(message.content, unsafe_allow_html=True)
            else:
                st.error(message.content)

    def show_ama(self):
        page = st.session_state.ui_page.page()
        if st.session_state.ama is None or st.session_state.ama.page != page:
            st.session_state.ama = AskMeAnything(page)

        with st.container():
            height = 400 if not config().x_no_right_panel else 200
            container = st.container(height=height, border=True, key="chat_container")
            with container:
                for message in st.session_state.ama.messages():
                    self.write_ama_message(message)

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

            if st.session_state.pending_ama:
                prompt = st.session_state.pending_ama
                st.session_state.pending_ama = None
                self.ama_completion(container, prompt)

    def ama_voice_input(self, container):
        toggle("ama_responding")
        voice = st.session_state.voice_input
        transcription = fast_transcription(voice)
        self.ama_completion(container, transcription)

    def ama_completion(self, container, prompt):
        page = st.session_state.ui_page.page()
        dfg = page.dfg
        ama: AskMeAnything = st.session_state.ama
        with container:
            with st.chat_message("user"):
                st.markdown(prompt)

            empty = st.empty()
            try:
                with empty.chat_message("assistant"):
                    response = st.write_stream(
                        ama.complete(prompt, st.session_state.selected_node)
                    )
            except Exception as e:
                error(e)
            finally:
                # self.write_ama_message(ama.last_message())
                st.session_state.ama_responding = False
                if dfg != page.dfg:
                    st.session_state.force_update = True
                    self.auto_update()
                st.rerun()  # TODO: This could be in a callback!  But should be okay...

    def auto_update(self):
        pass

    # override and call super in subclasses
    def show_node_details(self, node: Node):
        with st.container(border=True):
            st.write("###### Output")
            self.show_output(node)

        if node.kind != NodeKind.table and node.requirements:
            with st.container(key="node_requirements", border=True):
                st.write("###### Requirements")
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

        if node.kind != NodeKind.plot and node.function_return_type is not None:
            with st.container(key="node_type", border=True):
                st.write("###### Output Type")
                function_return_type = node.function_return_type
                if function_return_type is not None:
                    if not function_return_type.is_None_type():
                        st.caption(f"{function_return_type.description}")
                    st.code(schema_to_text(function_return_type.type_schema()))

        if AbstractionLevel.show_code(st.session_state.abstraction_level) and node.code:
            with st.container(key="node_code", border=True):
                st.write("###### Code")
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
                if type(value) in [np.ndarray, list, pd.Series]:
                    value = pd.DataFrame(value)
                if type(value) == pd.DataFrame:
                    st.dataframe(value, hide_index=True, height=200)
                elif type(value) == dict:
                    for k, v in list(value.items())[0:10]:
                        st.write(f"**{k}**:")
                        if type(v) in [np.ndarray, list, pd.Series]:
                            v = pd.DataFrame(v)
                        if type(v) == pd.DataFrame:
                            st.dataframe(v, hide_index=True, height=200)
                        elif type(v) == dict:
                            st.json(v)
                        elif type(v) == str:
                            if v.startswith("{" or v.startswith("[")):
                                st.json(v)
                            else:
                                st.code(v)
                        else:
                            st.code(v)
                    if len(value) > 10:
                        st.write(f"And {len(value)-10} more...")
                elif type(value) == str:
                    if value.startswith("{" or value.startswith("[")):
                        st.json(value)
                    else:
                        st.code(value)
                else:
                    st.code(value)
            elif node.result.output is not None:
                output = node.result.output
                if output is not None:
                    if output.output_type == OutputType.text:
                        st.text(f"```{output.data}\n```")
                    elif output.output_type == OutputType.image:
                        base64encoded = output.data.split(",", maxsplit=1)
                        image_data = base64encoded[0] + ";base64," + base64encoded[1]
                        st.image(image_data)

    def bottom_bar(self):
        ui_page: UIPage = st.session_state.ui_page
        with st.container(key="bottom_bar"):
            cols = st.columns(3)
            with cols[0]:
                if st.button(
                    label="",
                    icon=":material/settings:",
                    help="Change settings",
                ):
                    settings(ui_page)

            with cols[1]:
                if st.button(
                    label="Report Bug", icon=":material/bug_report:", key="report_bug"
                ):
                    self.report_bug()

            with cols[2]:
                if st.button(
                    label="Logout",
                    icon=":material/logout:",
                    help=f"Sign out {st.session_state.user_email if 'user_email' in st.session_state else ''}",
                ):
                    sign_out()
                    st.rerun()

    def report_bug(self):
        @st.dialog("Report Bug", width="small")
        def download_files():

            text = st.text_input(
                "Bug",
                placeholder="Enter a description of the issue",
            )

            if text:
                with st.spinner("Creating ZIP file..."):
                    file_name, zip_data = zip_bug(text)

                st.write("Saved on server.  Click below to download bug files locally.")

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

    def run(self, node_id: str):
        pass

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

    def node_parts_for_diagram(self):
        keys = ["pill", "messages", "requirements", "function_return_type"]
        if show_code():
            keys += ["code"]
        return keys

    def edit_new_nodes(self, original_dfg):
        new_dfg = st.session_state.ui_page.dfg()
        for node in new_dfg.nodes:
            if node.id not in original_dfg.node_ids():
                log("Edit new node", node)
                new_node_dialog(node)
                return

    def main(self):

        self.init()

        if config().x_no_right_panel:
            left, right = st.container(), None
        else:
            left, right = self.main_columns()

        with left:
            curr_state = st.session_state.flow_state
            ui_page = st.session_state.ui_page
            dfg = ui_page.dfg()

            # if st.session_state.layout_graph:
            #     ui_page = st.session_state.ui_page
            #     with ui_page.page():
            #         log("Layout the diagram")
            #         dfg = ui_page.dfg()
            #         dfg = dfg.update(
            #             nodes=[
            #                 x.update(
            #                     geometry=Geometry(
            #                         x=0,
            #                         y=0,
            #                         width=x.geometry.width,
            #                         height=x.geometry.height,
            #                     )
            #                 )
            #                 for x in dfg.nodes
            #             ]
            #         )
            #         ui_page.update_dfg(dfg)

            change = update_state(
                curr_state,
                dfg,
                self.node_parts_for_diagram(),
                selected_id=curr_state.selected_id,
                reset_pos=st.session_state.layout_graph,
            )
            st.session_state.layout_graph = False

            new_state, command = streamlit_flow(
                "example_flow",
                curr_state,
                layout=ManualLayout(),
                fit_view=False,
                height=1000,
                enable_node_menu=True,
                enable_edge_menu=True,
                enable_pane_menu=True,
                get_edge_on_click=True,
                get_node_on_click=True,
                show_minimap=True,
                hide_watermark=True,
                allow_new_edges=True,
                min_zoom=0.1,
                disabled=not self.graph_is_editable()
            )
            ui_page = st.session_state.ui_page
            dfg = ui_page.dfg()

            if st.session_state.last_state_update != new_state.timestamp:

                selected_nodes = [node.id for node in new_state.nodes if node.selected]
                new_state.selected_id = selected_nodes[0] if selected_nodes else None
                st.session_state.selected_node = new_state.selected_id

                st.session_state.flow_state = new_state
                st.session_state.last_state_update = new_state.timestamp

                if diff_state(curr_state, new_state):
                    new_dfg = update_dfg(new_state, dfg)
                    ui_page.update_dfg(new_dfg)
                    if not command and dfg != new_dfg:
                        st.session_state.force_update = True
                        st.rerun()

                if command:
                    st.session_state.last_state_update = new_state.timestamp
                    old_dfg = dfg
                    self.do_command(dfg, command)

        if right:
            with right:
                self.right_panel()

        with st.sidebar:
            self.sidebar()
            st.divider()
            self.bottom_bar()

        self.fini()

    def do_command(self, dfg, command_dict):
        original_dfg = dfg
        command = command_dict["command"]
        count = 0
        while f"Step-{count}" in dfg.node_ids():
            count += 1
        pill = f"Step-{count}"
        if command == "new_node":
            new_node = Node(
                id=f"Step-{count}",
                kind=NodeKind.compute,
                pill=pill,
                label="",
                geometry=Geometry(
                    x=command_dict["position"]["x"],
                    y=command_dict["position"]["y"],
                    width=160,
                    height=80,
                ),
                predecessors=[],
                function_name=pill_to_function_name(pill),
                function_result_var=pill_to_result_var_name(pill),
            )
            dfg = dfg.with_new_node(new_node)
            st.session_state.ui_page.update_dfg(dfg)
            new_node_dialog(new_node)
        elif command == "sketch":
            dataURL = command_dict["dataUrl"]
            ama: AskMeAnything = st.session_state.ama
            ama.assistant.add_text("user", "Here is a marked up version of the diagram")
            ama.assistant.add_image("user", dataURL)
            st.session_state.pending_ama = "Change the diagram to match this sketch"

        else:
            node_id = command_dict["id"]
            node = dfg.get_node(node_id)
            print("COMMAND", command, node_id)
            if command == "edit":
                self.prepare_node_for_edit(node.id)
            elif command == "run":
                self.run(node.id)
            elif command == "inspect":
                inspect_node(node)
            elif command == "lock":
                dfg = dfg.with_node(node.update(is_locked=True, phase=Phase.clean))
            elif command == "unlock":
                dfg = dfg.with_node(node.update(is_locked=False, phase=Phase.clean))
            elif command == "show":
                dfg = dfg.with_node(node.update(force_show_output=True))
            elif command == "hide":
                dfg = dfg.with_node(node.update(force_show_output=False))
            else:
                error("Unknown command", command)
            st.session_state.ui_page.update_dfg(dfg)
            if original_dfg != dfg:
                st.rerun()

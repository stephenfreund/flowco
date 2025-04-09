from dataclasses import dataclass
import textwrap
from typing import List, Literal
import streamlit as st
from flowco.dataflow.dfg import DataFlowGraph, Geometry, Node, NodeMessage
from flowco.dataflow.phase import Phase
from flowco.page.ama import AskMeAnything, VisibleMessage
from flowco.ui.dialogs.data_files import data_files_dialog
from flowco.ui.ui_page import st_abstraction_level
from flowco.ui.ui_page import UIPage
from flowco.ui.ui_util import phase_for_last_shown_part, set_session_state
from flowco.util.config import AbstractionLevel, config
from flowco.util.costs import inflight
from flowco.util.output import debug, error, log
from flowco.ui.page_files.base_page import FlowcoPage

import queue
import time


from flowco.dataflow import dfg_update
from flowco.ui.ui_builder import Builder
import streamlit as st


from flowco import __main__
from flowco.ui.ui_page import UIPage

from code_editor import code_editor
from flowthon.flowthon import FlowthonProgram

from flowco.ui.dialogs.node_editor import edit_node


@dataclass
class BuildButton:
    label: str
    icon: str
    action: Literal["Run", "Stop", "Update"]
    passes_key: str | None = None
    repair: bool = True
    node_specific: bool = False


class BuildPage(FlowcoPage):

    # Override for other pages

    def update_button(self) -> BuildButton:
        return BuildButton(label="Update", icon=":material/refresh:", action="Update")

    def run_button(self) -> BuildButton:
        return BuildButton(label="  Run  ", icon=":material/play_circle:", action="Run")

    def stop_button(self) -> BuildButton:
        return BuildButton(label="Stop", icon=":material/stop_circle:", action="Stop")

    def build_target_phase(self) -> Phase:
        return Phase.run_checked

    # Below should all be untouched for subclasses

    def button_bar(self):
        if st.session_state.builder is not None:
            with st.container(border=True):
                st.progress(
                    st.session_state.builder_progress,
                    st.session_state.builder.get_message(),
                )

        with st.container():
            with st.container(key="button_bar"):
                if st.session_state.builder is None:
                    cols = st.columns(8)
                    with cols[0]:
                        if st.session_state.builder is None:
                            run_button = self.run_button()
                        else:
                            run_button = self.stop_button()
                        st.button(
                            run_button.label,
                            icon=run_button.icon,
                            on_click=lambda: set_session_state(
                                "trigger_build_toggle", run_button
                            ),
                            disabled=st.session_state.ama_responding,
                            help=(
                                "Build and run the diagram"
                                if st.session_state.builder is None
                                else "Stop building"
                            ),
                        )

                    with cols[1]:
                        st.write(
                            "<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>",
                            unsafe_allow_html=True,
                        )

                    with cols[2]:
                        st.write(
                            "<span>&nbsp;</span>",
                            unsafe_allow_html=True,
                        )

                    with cols[3]:
                        st.button(
                            label="",
                            icon=":material/undo:",
                            disabled=(
                                not self.graph_is_editable()
                                or not st.session_state.ui_page.can_undo()
                            ),
                            on_click=lambda: st.session_state.ui_page.undo(),
                            help="Undo the last change",
                        )
                    with cols[4]:
                        st.button(
                            label="",
                            icon=":material/redo:",
                            disabled=(
                                not self.graph_is_editable()
                                or not st.session_state.ui_page.can_redo()
                            ),
                            on_click=lambda: st.session_state.ui_page.redo(),
                            help="Redo the last change",
                        )
                    with cols[5]:
                        st.write(
                            "<span>&nbsp;&nbsp;&nbsp;&nbsp;</span>",
                            unsafe_allow_html=True,
                        )
                    with cols[6]:
                        if st.button(
                            label="",
                            icon=":material/network_node:",
                            help="Layout the diagram",
                            disabled=not self.graph_is_editable(),
                        ):
                            ui_page = st.session_state.ui_page
                            with ui_page.page():
                                dfg = ui_page.dfg()
                                dfg = dfg.update(
                                    nodes=[
                                        x.update(
                                            geometry=Geometry(
                                                x=0, y=0, width=0, height=0
                                            )
                                        )
                                        for x in dfg.nodes
                                    ]
                                )
                                ui_page.update_dfg(dfg)
                            st.session_state.force_update = True
                            st.rerun()
                    with cols[7]:
                        if st.button(
                            label="",
                            icon=":material/cancel_presentation:",
                            help="Clear all node outputs",
                            disabled=not self.graph_is_editable(),
                        ):
                            st.session_state.ui_page.page().clear_outputs()
                            st.session_state.force_update = True
                            st.rerun()
                else:
                    run_button = self.stop_button()
                    st.button(
                        run_button.label,
                        icon=run_button.icon,
                        on_click=lambda: set_session_state(
                            "trigger_build_toggle", run_button
                        ),
                        disabled=st.session_state.ama_responding,
                        help=(
                            "Build and run the diagram"
                            if st.session_state.builder is None
                            else "Stop building"
                        ),
                    )

    def second_bar(self):
        if st.session_state.builder is None:
            super().second_bar()

    def filter_messages(self, node: Node) -> List[NodeMessage]:
        return [x for x in node.messages if x.phase <= Phase.run_checked]

    def auto_update(self):
        self.toggle_building(force=False, repair=True)
        super().auto_update()

    def edit_node(self, node_id: str):
        edit_node(node_id)

    def prepare_node_for_edit(self, node_id: str):
        ui_page: UIPage = st.session_state.ui_page
        # Refresh before editing!
        if ui_page.dfg()[node_id].phase < phase_for_last_shown_part():
            if st.session_state.builder is None:
                st.session_state.builder = Builder(
                    ui_page.page(),
                    node_id,
                    target_phase=phase_for_last_shown_part(),
                    passes_key=None,
                    force=False,
                    repair=True,
                )
                st.session_state.builder_progress = 0
        else:
            self.edit_node(node_id)

    def toggle_building(
        self, node_specific=False, passes_key=None, force=True, repair=True
    ):
        ui_page: UIPage = st.session_state.ui_page
        builder: Builder = st.session_state.builder
        if builder is None:
            st.session_state.builder = Builder(
                ui_page.page(),
                None if not node_specific else st.session_state.selected_node,
                target_phase=self.build_target_phase(),
                passes_key=passes_key,
                force=force,
                repair=repair,
            )
            st.session_state.builder_progress = 0
        else:
            builder.stop()
            self.clear_builder_and_reset_state()

    def get_builder_updates(self):
        ui_page: UIPage = st.session_state.ui_page
        if st.session_state.builder is not None:
            while not st.session_state.builder.empty():
                builder: Builder = st.session_state.builder
                try:
                    build_update = builder.get()
                    if build_update.steps_total > 0:
                        st.session_state.builder_progress = (
                            1 - build_update.steps_remaining / build_update.steps_total
                        )
                    dfg = build_update.new_graph
                    ui_page.update_dfg(dfg)
                    builder.update_done()
                    # if build_update.updated_node is not None:
                    #     text = self.new_node_to_message(build_update.updated_node)
                    #     if text:
                    #         st.session_state.ama.add_visible_message(
                    #             VisibleMessage(role="assistant", content=text)
                    #         )

                except queue.Empty:
                    continue

    def graph_is_editable(
        self,
    ):
        return super().graph_is_editable() and not (
            st.session_state.builder and st.session_state.builder.is_alive()
        )

    def update_ui_page(self, update: dfg_update.mxDiagramUpdate):
        ui_page: UIPage = st.session_state.ui_page
        if ui_page.dfg().version != update.version:
            debug("Dataflow graph version mismatch on update -- this is fine...")
            st.session_state.force_update = True
        else:
            debug("Updating UI page")
            new_dfg = dfg_update.update_dataflow_graph(ui_page.dfg(), update)

            if new_dfg != ui_page.dfg():
                ui_page.update_dfg(new_dfg)

    def init(self):
        self.get_builder_updates()

    def messages_triggering_global_error_check(self):
        dfg: DataFlowGraph = st.session_state.ui_page.dfg()
        error_run_messages = dfg.filter_messages(
            [x for x in Phase],  #  if x.value <= Phase.run_checked.value],
            level="error",
        )
        return error_run_messages

    def global_error_check(self):
        @st.dialog("Errors", width="large")
        def global_error_fix(error_messages):
            st.write(
                "There are errors in the diagram that need to be fixed.  Proceed to try to fix them automatically?"
            )
            if st.button("Fix All"):
                errors_as_string = "\n".join(
                    [
                        f"* **{node.pill}**: {message.message()}\n"
                        for node, message in error_messages
                    ]
                )
                prompt = f"For each error, identify the root cause, working backwards from the node with the error if necessary.  Make minimal changes to the graph and nodes to fix each error.\n{errors_as_string}\n"

                st.session_state.pending_ama = prompt
                st.rerun()

            for node, message in error_messages:
                st.error(f"**{node.pill}**: {message.message()}")

        error_run_messages = self.messages_triggering_global_error_check()
        if error_run_messages:
            global_error_fix(error_run_messages)

    def fini(self):

        if st.session_state.trigger_build_toggle is not None:
            button = st.session_state.trigger_build_toggle
            st.session_state.trigger_build_toggle = None
            self.toggle_building(
                node_specific=button.node_specific,
                force=False,  # (button.action == "Run"),
                repair=button.repair,
                passes_key=button.passes_key,
            )
            st.session_state.force_update = True
            st.rerun()

        builder: Builder = st.session_state.builder
        debug(f"Builder is alive: {builder is not None and builder.is_alive()}")
        if builder is not None:
            if not builder.is_alive():
                self.get_builder_updates()
                self.clear_builder_and_reset_state()
                st.session_state.global_error_check = True
                st.rerun()
            else:
                time.sleep(0.25)
                st.rerun()

        if st.session_state.global_error_check:
            st.session_state.global_error_check = False
            self.global_error_check()

    def clear_builder_and_reset_state(self):
        st.session_state.force_update = True
        st.session_state.builder = None
        ui_page: UIPage = st.session_state.ui_page
        dfg = ui_page.dfg()
        dfg = dfg.update(
            nodes=[
                x.update(
                    build_status=None,
                )
                for x in dfg.nodes
            ]
        )
        ui_page.update_dfg(dfg)

    @st.dialog("Edit Description", width="large")
    def edit_description(self):
        ui_page: UIPage = st.session_state.ui_page
        text = st.text_area(
            "Description",
            key="graph_description",
            value=ui_page.dfg().description,
            height=300,
            label_visibility="collapsed",
        )
        if st.button("Ok"):
            ui_page.page().user_edit_graph_description(text)
            st.rerun()

    @st.dialog("Edit as Flowthon program", width="large")
    def edit_flowthon(self):
        def doit():
            updated_source = st.session_state.code_editor["text"]
            flowthon = FlowthonProgram.from_source(updated_source)
            ui_page.page().merge_flowthon(flowthon, rebuild=False, interactive=False)
            st.session_state.force_update = True

        ui_page: UIPage = st.session_state.ui_page
        source = ui_page.page().to_flowthon().to_source(st_abstraction_level())

        if st.button(
            "Save",
            on_click=doit,
            disabled=st.session_state.code_editor is None
            or st.session_state.code_editor["text"] == source,
        ):
            st.rerun()

        code_editor(
            source,
            key="code_editor",
            lang="python",
            response_mode="debounce",
            props={
                "showGutter": True,
            },
            options={
                "wrap": True,
                "showLineNumbers": True,
            },
        )

    def global_sidebar(self):
        ui_page: UIPage = st.session_state.ui_page

        with st.popover("Pinned Outputs"):
            nodes = ui_page.dfg().topological_sort()
            pills = [ui_page.dfg()[node].pill for node in nodes]
            st.pills(
                "Pills",
                nodes,
                default=st.session_state.pinned_nodes,
                key="pinned_nodes_pills",
                format_func=lambda id: ui_page.dfg()[id].pill,
                selection_mode="multi",
                label_visibility="collapsed",
                on_change=lambda: set_session_state(
                    "pinned_nodes", st.session_state.pinned_nodes_pills
                ),
            )

        if st.session_state.pinned_nodes:
            with st.container(border=True):
                for node_id in st.session_state.pinned_nodes:
                    if node_id in ui_page.dfg().node_ids():
                        node = ui_page.dfg()[node_id]
                        st.write(f"###### {node.pill}")
                        super().show_output(node)

        with st.container(border=True):
            st.write("###### Notes")
            description = ui_page.dfg().description
            if description:
                st.write(description)
            else:
                st.write("*Add notes here*")

        cols = st.columns(4)
        with cols[0]:
            if st.button(
                label="",
                icon=":material/edit_note:",
                disabled=not self.graph_is_editable(),
                help="Edit the description of the diagram",
                key="edit_description",
            ):
                self.edit_description()

        self.help_details()

    def help_details(self):
        st.divider()
        colors = [
            ("#dfedf7", "Modified"),
            ("#fac4b3", "Running"),
            ("#fef2d0", "Completed"),
            ("#ddf1da", "Checked"),
            ("#dedbec", "Tested"),
        ]
        with st.container():
            st.markdown("###### Node Colors and Kinds")
            left, right = st.columns(2)
            with left:
                lines = (
                    ["<div style='background-color:#FFFFFF;padding:0.5rem;'>"]
                    + [
                        f"<div style='background-color:{color};padding:0.2rem; font-size:12px;'>{label}</div>"
                        for color, label in colors
                    ]
                    + ["</div>"]
                )
                st.html("\n".join(lines))
            with right:
                with st.container(key="node_shapes"):
                    st.image("static/nodes.png")
            st.write(
                textwrap.dedent(
                    """\
                ###### Graph Operations
                * **Scroll canvas:** Drag around canvas
                * **Select node:** Click on it
                * **Create node:** Shift-click on canvas
                * **Move node:** Click on it and drag
                * **Edit node:** Hover over node and press pencil
                * **Delete node:** Hover over node and press trash can
                * **Add edge:** Hover over node and drag from the (+) icon to the other node
                * **Add edge and node:** Hover over node and drag from the (+) icon to the other node
                * **Delete edge:** Hover over edge and press trash can"""
                )
            )

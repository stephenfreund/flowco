import dis
from typing import List
import streamlit as st
from flowco.builder import build
from flowco.dataflow.dfg import Geometry
from flowco.ui.dialogs.edit_node import edit_node
from flowco.ui.dialogs.data_files import data_files_dialog
from flowco.ui.ui_page import UIPage
from flowco.ui.ui_util import phase_for_last_shown_part, set_session_state, toggle
from flowco.util.output import error, log, debug, warn
from flowco.ui.page_files.base_page import FlowcoPage

import queue
import time


from flowco.dataflow import dfg_update
from flowco.ui.ui_builder import Builder
import streamlit as st


from flowco import __main__
from flowco.ui.ui_page import UIPage


class BuildPage(FlowcoPage):

    def button_bar(self):
        if st.session_state.builder is not None:
            with st.container(border=True):
                st.progress(
                    st.session_state.builder_progress,
                    st.session_state.builder.get_message(),
                )
        with st.container(key="button_bar"):
            cols = st.columns(8)
            with cols[0]:
                st.button(
                    "Run All" if st.session_state.builder is None else " Stop ",
                    on_click=lambda: set_session_state(
                        "trigger_build_toggle",
                        "Run" if st.session_state.builder is None else "Stop",
                    ),
                    disabled=st.session_state.ama_responding,
                    help="Build and run the whole diagram" if st.session_state.builder is None else "Stop building",
                )
            with cols[2]:
                st.button(
                    "Run",
                    on_click=lambda: set_session_state(
                        "trigger_build_toggle", "Update"
                    ),
                    disabled=not self.graph_is_editable(),
                    help="Build and run any nodes that have changed since the last Run",
                )
            with cols[4]:
                st.button(
                    ":material/undo:",
                    disabled=(
                        not self.graph_is_editable()
                        or not st.session_state.ui_page.can_undo()
                    ),
                    on_click=lambda: st.session_state.ui_page.undo(),
                    help="Undo the last change",
                )
            with cols[5]:
                st.button(
                    ":material/redo:",
                    disabled=(
                        not self.graph_is_editable()
                        or not st.session_state.ui_page.can_redo()
                    ),
                    on_click=lambda: st.session_state.ui_page.redo(),
                    help="Redo the last change",
                )
            with cols[6]:
                if st.button(
                    ":material/network_node:",
                    help="Layout the diagram",
                    disabled=not self.graph_is_editable(),
                ):
                    ui_page = st.session_state.ui_page
                    dfg = ui_page.dfg()
                    dfg = dfg.update(
                        nodes=[
                            x.update(geometry=Geometry(x=0, y=0, width=0, height=0))
                            for x in dfg.nodes
                        ]
                    )
                    ui_page.update_dfg(dfg)
                    st.session_state.force_update = True
                    st.rerun()

    def auto_update(self):
        self.toggle_building(force=False)
        super().auto_update()

    def edit_node(self, node_id: str):
        ui_page: UIPage = st.session_state.ui_page
        if ui_page.dfg()[node_id].phase < phase_for_last_shown_part():
            if st.session_state.builder is None:
                st.session_state.builder = Builder(
                    ui_page.page(),
                    node_id,
                    target_phase=phase_for_last_shown_part(),
                    force=False,
                )
                st.session_state.builder_progress = 0
        else:
            edit_node(node_id)

    def toggle_building(self, force=True):
        ui_page: UIPage = st.session_state.ui_page
        builder: Builder = st.session_state.builder
        if builder is None:
            st.session_state.builder = Builder(ui_page.page(), None, force=force)
            st.session_state.builder_progress = 0
        else:
            builder.stop()

    def get_builder_updates(self):
        builder: Builder = st.session_state.builder
        ui_page: UIPage = st.session_state.ui_page
        if builder is not None:
            while not builder.empty():
                try:
                    build_update = builder.get()
                    if build_update.steps_total > 0:
                        st.session_state.builder_progress = (
                            1 - build_update.steps_remaining / build_update.steps_total
                        )
                    dfg = build_update.new_graph
                    ui_page.update_dfg(dfg)
                    builder.update_done()
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

    def fini(self):

        if st.session_state.trigger_build_toggle is not None:
            value = st.session_state.trigger_build_toggle
            st.session_state.trigger_build_toggle = None
            self.toggle_building(value == "Run")

        builder: Builder = st.session_state.builder
        debug(f"Builder is alive: {builder is not None and builder.is_alive()}")
        if builder is not None:
            if not builder.is_alive():

                self.get_builder_updates()

                st.session_state.force_update = True
                st.session_state.builder = None
                st.rerun()
            else:
                time.sleep(0.25)
                st.rerun()

    @st.dialog("Edit Description", width="wide")
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

    def global_sidebar(self):
        ui_page: UIPage = st.session_state.ui_page

        # st.text_area(
        #     "Description",
        #     key="graph_description",
        #     value=ui_page.dfg().description,
        #     height=200,
        #     on_change=lambda: ui_page.page().user_edit_graph_description(
        #         st.session_state.graph_description
        #     ),
        # )

        st.write("### Description")
        st.write(ui_page.dfg().description)

        with st.container(key="page_controls"):
            cols = st.columns(2)
            with cols[0]:
                if st.button("Edit Description", disabled=not self.graph_is_editable(), help="Edit the description of the diagram"):
                    self.edit_description()

            with cols[1]:
                if st.button("Data Files", disabled=not self.graph_is_editable(), help="Manage data files for the diagram"):
                    data_files_dialog()

import textwrap
from typing import List, Tuple
import pandas as pd
import streamlit as st
from flowco.builder.build import BuildEngine
from flowco.dataflow.dfg import DataFlowGraph, Geometry, Node
from flowco.dataflow.phase import Phase
from flowco.ui.dialogs.data_files import data_files_dialog
from flowco.ui.page_files.build_page import BuildPage
from flowco.ui.ui_page import st_abstraction_level
from flowco.ui.ui_page import UIPage
from flowco.ui.ui_util import (
    phase_for_last_shown_part,
    set_session_state,
    show_code,
    show_requirements,
)
from flowco.util.config import config
from flowco.util.output import debug
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

if config.x_algorithm_phase:
    from flowco.ui.dialogs.edit_node import edit_node
else:
    from flowco.ui.dialogs.edit_node_no_alg import edit_node


class CheckPage(BuildPage):

    # Override for other pages

    def update_button_label(self) -> str:
        return ":material/refresh: Update"

    def run_button_label(self) -> str:
        return ":material/play_circle: Check"

    def build_target_phase(self) -> Phase:
        return Phase.assertions_checked

    @st.dialog("Edit Checks", width="large")
    def edit_checks(self, node_id: str):

        ui_page: UIPage = st.session_state.ui_page
        node = st.session_state.tmp_dfg[node_id]
        buttons = st.empty()
        st.write("### Checks")
        editable_df = st.data_editor(
            pd.DataFrame({"checks": st.session_state.tmp_assertions}, dtype=str),
            key="edit_checks",
            num_rows="dynamic",
            use_container_width=True,
        )

        new_assertions = list([x for x in editable_df["checks"] if x])
        if new_assertions != node.assertions or node.phase < Phase.assertions_code:
            dfg = st.session_state.tmp_dfg
            node = dfg[node_id]
            dfg = dfg.with_node(node.update(assertions=new_assertions))
            dfg = dfg.reduce_phases_to_below_target(node.id, Phase.assertions_code)
            st.session_state.tmp_dfg = dfg

        dfg = st.session_state.tmp_dfg
        if show_code() and dfg[node_id].phase < Phase.assertions_code:
            with st.spinner("Generating validation steps..."):
                build_config = ui_page.page().base_build_config(repair=False)
                engine = BuildEngine.get_builder()
                for build_updated in engine.build_with_worklist(
                    build_config, dfg, Phase.assertions_code, node_id
                ):
                    dfg = build_updated.new_graph
                st.session_state.tmp_dfg = dfg

        with buttons.container():
            if st.button("Save"):
                ui_page.page().update_dfg(st.session_state.tmp_dfg)
                st.session_state.force_update = True
                st.rerun()

        if show_code():
            node = st.session_state.tmp_dfg[node_id]
            st.write("### Validation Steps")
            if node.assertion_checks:

                for message in node.messages or []:
                    if message.phase == Phase.assertions_code:
                        if message.level == "error":
                            st.error(f"{message.text}")
                        else:
                            st.warning(f"{message.text}")

                for assertion in node.assertions or []:
                    check = node.assertion_checks.get(assertion, None)
                    if check:
                        st.write(f"* **{assertion}**")
                        if check.type == "quantitative":
                            code = check.code
                            if code:
                                st.code(textwrap.indent("\n".join(code), "    "))
                            else:
                                st.write("    *Code not available*")
                        else:
                            st.write(f"    *{check.requirement}*")

    def edit_node(self, node_id: str):
        ui_page: UIPage = st.session_state.ui_page
        with ui_page.page():
            st.session_state.tmp_dfg = ui_page.dfg()
            st.session_state.tmp_assertions = ui_page.dfg()[node_id].assertions or []
            self.edit_checks(node_id)

    def show_node_details(self, node: Node):
        st.write("**Checks**")
        st.write("\n".join(["* " + x for x in (node.assertions or [])]))
        # if st.button(
        #     ":material/edit_note:",
        #     disabled=not self.graph_is_editable(),
        #     help="Edit node checks",
        # ):
        #     self.edit_checks(node)

        super().show_node_details(node)

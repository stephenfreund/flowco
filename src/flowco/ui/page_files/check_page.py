import textwrap
import pandas as pd
import streamlit as st
from flowco.builder.assertions import suggest_assertions
from flowco.builder.build import BuildEngine
from flowco.dataflow.dfg import Node
from flowco.dataflow.phase import Phase
from flowco.ui.page_files.build_page import BuildPage
from flowco.ui.ui_page import UIPage
from flowco.ui.ui_util import (
    show_code,
)
from flowco.util.config import config


import streamlit as st


from flowco import __main__
from flowco.ui.ui_page import UIPage


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

        def make_suggestions():
            st.session_state.make_suggestions = True

        node = st.session_state.tmp_dfg[node_id]
        buttons = st.empty()

        if st.session_state.make_suggestions:
            with st.spinner("Making suggestions..."):
                st.session_state.make_suggestions = False
                suggested_assertions = suggest_assertions(
                    st.session_state.tmp_dfg, st.session_state.tmp_dfg[node_id]
                )
                st.session_state.tmp_assertions = (
                    st.session_state.tmp_assertions + suggested_assertions
                )
                dfg = st.session_state.tmp_dfg
                node = dfg[node_id]
                dfg = dfg.reduce_phases_to_below_target(node.id, Phase.assertions_code)
                st.session_state.tmp_dfg = dfg

        st.write("### Checks")
        print(st.session_state.tmp_assertions)
        editable_df = st.data_editor(
            pd.DataFrame({"checks": st.session_state.tmp_assertions}, dtype=str),
            key="edit_checks",
            num_rows="dynamic",
            use_container_width=True,
        )

        new_assertions = list([x for x in editable_df["checks"] if x])
        dfg = st.session_state.tmp_dfg
        node = dfg[node_id]
        if new_assertions != node.assertions or node.phase < Phase.assertions_code:
            dfg = dfg.with_node(node.update(assertions=new_assertions))
            dfg = dfg.reduce_phases_to_below_target(node.id, Phase.assertions_code)
            st.session_state.tmp_dfg = dfg

        with buttons.container():
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("Save"):
                    ui_page: UIPage = st.session_state.ui_page
                    ui_page.page().update_dfg(st.session_state.tmp_dfg)
                    st.session_state.force_update = True
                    st.rerun(scope="app")

            with c2:
                st.button("Suggest", on_click=make_suggestions)

            with c3:
                if show_code():
                    if st.button("Regenerate"):
                        with st.spinner("Regenerating..."):
                            dfg = st.session_state.tmp_dfg
                            dfg = dfg.reduce_phases_to_below_target(
                                node_id, Phase.assertions_code
                            )
                            node = dfg[node_id]
                            dfg = dfg.with_node(
                                node.update(
                                    cache=node.cache.invalidate(Phase.assertions_code)
                                )
                            )
                            st.session_state.tmp_dfg = dfg

        dfg = st.session_state.tmp_dfg
        if show_code():
            if dfg[node_id].phase < Phase.assertions_code:
                with st.spinner("Generating validation steps..."):
                    ui_page: UIPage = st.session_state.ui_page
                    build_config = ui_page.page().base_build_config(repair=False)
                    engine = BuildEngine.get_builder()
                    for build_updated in engine.build_with_worklist(
                        build_config, dfg, Phase.assertions_code, node_id
                    ):
                        dfg = build_updated.new_graph
                    st.session_state.tmp_dfg = dfg

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

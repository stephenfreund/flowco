import textwrap
import pandas as pd
import streamlit as st
from flowco.builder.assertions import suggest_assertions
from flowco.builder.build import BuildEngine
from flowco.dataflow.dfg import Node
from flowco.dataflow.phase import Phase
from flowco.ui.page_files.build_page import BuildButton, BuildPage
from flowco.ui.ui_page import UIPage
from flowco.ui.ui_util import (
    set_session_state,
    show_code,
)


import streamlit as st


from flowco import __main__
from flowco.ui.ui_page import UIPage


class CheckPage(BuildPage):

    # Override for other pages

    def update_button(self) -> BuildButton:
        return BuildButton(
            label="Update",
            icon=":material/refresh:",
            action="Update",
            # passes_key="repair-checks-passes",
        )

    def run_button(self) -> BuildButton:
        return BuildButton(
            label="Check",
            icon=":material/play_circle:",
            action="Run",
            # passes_key="repair-checks-passes",
        )

    def fix_button(self) -> BuildButton:
        return BuildButton(
            label="Fix",
            icon=":material/build:",
            action="Run",
            passes_key="repair-checks-passes",
        )

    def build_target_phase(self) -> Phase:
        return Phase.assertions_checked

    def node_header(self, node: Node):
        super().node_header(node)
        assertion_failures = node.filter_messages(Phase.assertions_checked, "error")
        if assertion_failures:
            fix_button = self.fix_button()
            st.button(
                fix_button.label,
                on_click=lambda: set_session_state("trigger_build_toggle", fix_button),
                disabled=not self.graph_is_editable(),
                help="Fix any errors in the checks",
            )
        else:
            if node.phase >= Phase.assertions_checked:
                st.success("All checks passed")

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
        st.session_state.tmp_assertions = [
            x for x in st.session_state.tmp_assertions if x
        ]
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
                if st.button("Save", icon=":material/save:"):
                    ui_page: UIPage = st.session_state.ui_page
                    ui_page.page().update_dfg(st.session_state.tmp_dfg)
                    st.session_state.force_update = True
                    st.rerun(scope="app")

            with c2:
                st.button(
                    "Suggest",
                    icon=":material/format_list_bulleted:",
                    on_click=make_suggestions,
                )

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
        if node.assertion_checks:
            for assertion in node.assertions or []:
                st.divider()
                check = node.assertion_checks.get(assertion, None)
                if check:
                    st.write(f"**{assertion}**")
                    if check.warning:
                        st.warning(check.warning)

                    if show_code():
                        if check.type == "quantitative":
                            code = check.code
                            if code:
                                st.code(textwrap.indent("\n".join(code), "    "))
                            else:
                                st.write("    *Code not available*")
                        else:
                            st.write(f"    *{check.requirement}*")
        else:
            st.write("*No details available*")

    def edit_node(self, node_id: str):
        ui_page: UIPage = st.session_state.ui_page
        with ui_page.page():
            st.session_state.tmp_dfg = ui_page.dfg()
            st.session_state.tmp_assertions = ui_page.dfg()[node_id].assertions or []
            self.edit_checks(node_id)

    def show_node_details(self, node: Node):
        with st.container(key="node_code", border=True):
            st.write("###### Checks")
            assertions = node.assertions or []
            if assertions:
                st.write("\n".join(["* " + x for x in assertions]))
            else:
                st.write("*Edit node to add checks!*")

        super().show_node_details(node)

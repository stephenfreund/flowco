import textwrap
from typing import List
import pandas as pd
import streamlit as st
from flowco.builder.assertions import suggest_assertions
from flowco.builder.build import BuildEngine
from flowco.dataflow.dfg import Node, NodeMessage
from flowco.dataflow.phase import Phase
from flowco.ui.page_files.build_page import BuildButton, BuildPage
from flowco.ui.ui_builder import Builder
from flowco.ui.ui_page import UIPage
from flowco.ui.ui_util import (
    set_session_state,
    show_code,
)


import streamlit as st


from flowco import __main__
from flowco.ui.ui_page import UIPage
from flowco.llm.assistant import AssistantError


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

    def suggest_button(self) -> BuildButton:
        return BuildButton(
            label="Suggest Checks",
            icon=":material/format_list_bulleted:",
            action="Run",
            passes_key="suggest-assertions-passes",
        )

    def build_target_phase(self) -> Phase:
        return Phase.assertions_checked

    def messages_triggering_global_error_check(self):
        node_id = st.session_state.node_id_with_fix_button
        if node_id is None:
            return []
        ui_page: UIPage = st.session_state.ui_page
        node = ui_page.dfg()[node_id]
        error_run_messages = node.filter_messages(
            [x for x in Phase],  #  if x.value <= Phase.run_checked.value],
            level="error",
        )
        return [(node, m) for m in error_run_messages]

    def node_header(self, node: Node):
        super().node_header(node)
        assertion_failures = node.filter_messages(Phase.assertions_checked, "error")
        if assertion_failures:
            fix_button = self.fix_button()

            def fire():
                set_session_state("trigger_build_toggle", fix_button)
                st.session_state.node_id_with_fix_button = node.id

            st.button(
                fix_button.label,
                on_click=fire,
                disabled=not self.graph_is_editable(),
                help="Fix any errors in the checks",
            )
        else:
            if node.phase >= Phase.assertions_checked:
                st.success("All checks passed")

    def edit_checks(self, node_id: str):
        node = st.session_state.tmp_dfg[node_id]

        @st.dialog(f"Output Checks for {node.pill}", width="large")
        def for_real(node_id: str):
            top = st.empty()
            try:

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
                        dfg = dfg.reduce_phases_to_below_target(
                            node.id, Phase.assertions_code
                        )
                        st.session_state.tmp_dfg = dfg

                # st.write(f"### Output Checks for {node.pill}")
                st.session_state.tmp_assertions = [
                    x for x in st.session_state.tmp_assertions if x
                ]
                editable_df = st.data_editor(
                    pd.DataFrame(
                        {"checks": st.session_state.tmp_assertions}, dtype=str
                    ),
                    key="edit_checks",
                    num_rows="dynamic",
                    use_container_width=True,
                )

                new_assertions = list([x for x in editable_df["checks"] if x])
                st.session_state.tmp_assertions = new_assertions
                dfg = st.session_state.tmp_dfg
                node = dfg[node_id]
                if (
                    new_assertions != node.assertions
                    or node.phase < Phase.assertions_code
                ):
                    dfg = dfg.with_node(node.update(assertions=new_assertions))
                    dfg = dfg.reduce_phases_to_below_target(
                        node.id, Phase.assertions_code
                    )
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
                                            cache=node.cache.invalidate(
                                                Phase.assertions_code
                                            )
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
                        check = node.assertion_checks.get(assertion, None)
                        if check and (check.warning or show_code()):
                            st.divider()
                            st.write(f"**{assertion}**")
                            if check.warning:
                                st.warning(check.warning)

                            if show_code():
                                if check.type == "quantitative":
                                    code = check.code
                                    if code:
                                        st.code(
                                            textwrap.indent("\n".join(code), "    ")
                                        )
                                    else:
                                        st.write("    *Code not available*")
                                else:
                                    st.write(f"    *{check.requirement}*")
                else:
                    st.write("*No details available*")
            except AssistantError as e:
                top.error(e)

        for_real(node_id)

    def edit_node(self, node_id: str):
        ui_page: UIPage = st.session_state.ui_page
        with ui_page.page():
            st.session_state.tmp_dfg = ui_page.dfg()
            st.session_state.tmp_assertions = ui_page.dfg()[node_id].assertions or []
            self.edit_checks(node_id)

    def show_node_details(self, node: Node):
        with st.container(key="node_checks", border=True):
            st.write("###### Checks")
            assertions = node.assertions or []
            if assertions:
                st.write("\n".join(["* " + x for x in assertions]))
            else:
                st.write("*Edit node to add checks!*")

        super().show_node_details(node)

    def node_parts_for_diagram(self):
        keys = [
            "pill",
            "messages",
            "assertions",
            "requirements",
            "function_return_type",
        ]
        if show_code():
            keys += ["code"]
        return keys

    def global_sidebar(self):
        ui_page: UIPage = st.session_state.ui_page

        st.session_state.node_with_fix_button = None

        failed_assertions = [
            node.pill
            for node in ui_page.dfg().nodes
            if node.filter_messages(Phase.assertions_checked, "error")
        ]
        if failed_assertions:
            items = "\n".join([f"* {x}" for x in failed_assertions])
            st.error(f"**These nodes have failing checks:**\n{items}")

        warnings = [
            node.pill
            for node in ui_page.dfg().nodes
            if node.filter_messages(Phase.assertions_code, "warning")
        ]
        if warnings:
            items = "\n".join([f"* {x}" for x in warnings])
            st.warning(f"**These nodes have check warnings:**\n{items}")

        nodes_with_no_assertions = [
            node.pill for node in ui_page.dfg().nodes if not node.assertions
        ]
        nodes_with_no_assertions.sort()

        if nodes_with_no_assertions:
            items = "\n".join([f"* {x}" for x in nodes_with_no_assertions])
            st.info(f"**These nodes have no checks:**\n{items}")

            suggest_button = self.suggest_button()
            if st.button(
                suggest_button.label,
                icon=suggest_button.icon,
                disabled=st.session_state.ama_responding
                or st.session_state.builder is not None,
                help=("Suggest tests for nodes that have none. "),
            ):
                if st.session_state.builder is None:
                    st.session_state.builder = Builder(
                        ui_page.page(),
                        [x.id for x in ui_page.dfg().nodes if not x.assertions],
                        target_phase=Phase.assertions_code,
                        passes_key="suggest-assertions-passes",
                        force=True,
                        repair=False,
                    )
                    st.session_state.builder_progress = 0

        self.help_details()

    def filter_messages(self, node: Node) -> List[NodeMessage]:
        return [
            x
            for x in node.messages
            if x.phase <= Phase.assertions_checked and x.level == "error"
        ]

    def clear_builder_and_reset_state(self):
        super().clear_builder_and_reset_state()
        st.session_state.node_id_with_fix_button = None

import textwrap
from typing import List
import pandas as pd
import streamlit as st

# from flowco.builder.unit_tests import suggest_unit_tests
from flowco.builder.build import BuildEngine
from flowco.builder.unit_tests import suggest_unit_tests
from flowco.dataflow.dfg import Node, NodeMessage
from flowco.dataflow.phase import Phase
from flowco.dataflow.tests import UnitTest
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


class TestPage(BuildPage):

    # Override for other pages

    def update_button(self) -> BuildButton:
        return BuildButton(
            label="Update",
            icon=":material/refresh:",
            action="Update",
        )

    def run_button(self) -> BuildButton:
        return BuildButton(
            label="Test",
            icon=":material/play_circle:",
            action="Run",
        )

    def fix_button(self) -> BuildButton:
        return BuildButton(
            label="Fix",
            icon=":material/build:",
            action="Run",
            passes_key="repair-tests-passes",
        )

    def suggest_button(self) -> BuildButton:
        return BuildButton(
            label="Suggest Tests",
            icon=":material/format_list_bulleted:",
            action="Run",
            passes_key="suggest-unit-test-passes",
        )

    def build_target_phase(self) -> Phase:
        return Phase.unit_tests_checked

    def node_header(self, node: Node):
        super().node_header(node)
        test_failures = node.filter_messages(Phase.unit_tests_checked, "error")
        if test_failures:
            fix_button = self.fix_button()
            st.button(
                fix_button.label,
                on_click=lambda: set_session_state("trigger_build_toggle", fix_button),
                help="Fix any errors in the tests",
            )
        else:
            if node.phase >= Phase.unit_tests_checked:
                st.success("All tests passed")

    def edit_checks(self, node_id: str):
        node = st.session_state.tmp_dfg[node_id]

        @st.dialog(f"Unit Tests for {node.pill}", width="large")
        def for_real(node_id: str):

            top = st.empty()
            try:
                with st.container(key="unit-test-ui"):

                    def make_suggestions():
                        st.session_state.make_suggestions = True

                    node = st.session_state.tmp_dfg[node_id]
                    buttons = st.empty()

                    if st.session_state.make_suggestions:
                        with st.spinner("Making suggestions..."):
                            st.session_state.make_suggestions = False
                            suggested_unit_tests = suggest_unit_tests(
                                st.session_state.tmp_dfg,
                                st.session_state.tmp_dfg[node_id],
                            )
                            st.session_state.tmp_unit_tests = (
                                st.session_state.tmp_unit_tests + suggested_unit_tests
                            )
                            dfg = st.session_state.tmp_dfg
                            node = dfg[node_id]
                            dfg = dfg.reduce_phases_to_below_target(
                                node.id, Phase.unit_tests_code
                            )
                            st.session_state.tmp_dfg = dfg

                    st.session_state.tmp_unit_tests = [
                        x for x in st.session_state.tmp_unit_tests if x
                    ]
                    editable_df = st.data_editor(
                        pd.DataFrame(
                            [
                                {
                                    "description": x.description,
                                    "inputs": x.inputs,
                                    "expected outcome": x.expected,
                                }
                                for x in st.session_state.tmp_unit_tests
                            ],
                            columns=["description", "inputs", "expected outcome"],
                            dtype=str,
                        ),
                        key="edit_checks",
                        num_rows="dynamic",
                        use_container_width=True,
                    )

                    fixed = editable_df.fillna("")
                    new_unit_tests = [
                        UnitTest(
                            description=x.get("description", ""),
                            inputs=x.get("inputs", ""),
                            expected=x.get("expected outcome", ""),
                        )
                        for x in fixed.to_dict(orient="records")
                    ]
                    dfg = st.session_state.tmp_dfg
                    node = dfg[node_id]
                    if (
                        new_unit_tests != node.unit_tests
                        or node.phase < Phase.unit_tests_code
                    ):
                        dfg = dfg.with_node(node.update(unit_tests=new_unit_tests))
                        dfg = dfg.reduce_phases_to_below_target(
                            node.id, Phase.unit_tests_code
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
                                            node_id, Phase.unit_tests_code
                                        )
                                        node = dfg[node_id]
                                        dfg = dfg.with_node(
                                            node.update(
                                                cache=node.cache.invalidate(
                                                    Phase.unit_tests_code
                                                )
                                            )
                                        )
                                        st.session_state.tmp_dfg = dfg

                    dfg = st.session_state.tmp_dfg

                    if dfg[node_id].phase < Phase.unit_tests_code:
                        with st.spinner("Generating validation steps..."):
                            ui_page: UIPage = st.session_state.ui_page
                            build_config = ui_page.page().base_build_config(
                                repair=False
                            )
                            engine = BuildEngine.get_builder()
                            for build_updated in engine.build_with_worklist(
                                build_config, dfg, Phase.unit_tests_code, node_id
                            ):
                                dfg = build_updated.new_graph
                            st.session_state.tmp_dfg = dfg

                    node = st.session_state.tmp_dfg[node_id]
                    if node.unit_test_checks:
                        for unit_test in node.unit_tests or []:
                            st.divider()
                            check = node.unit_test_checks.get(str(unit_test), None)
                            if check:
                                st.write(f"**{unit_test.description}**")
                                st.write(
                                    f"* **Input**: {unit_test.inputs}\n* **Expected:** {unit_test.expected}"
                                )
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
                                    elif check.type == "qualitative-code":
                                        code = check.code
                                        if code:
                                            st.write("Setup code:")
                                            st.code(
                                                textwrap.indent("\n".join(code), "    ")
                                            )
                                        else:
                                            st.write("    *Code not available*")

                                        st.write(f"    *{check.requirement}*")
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
            st.session_state.tmp_unit_tests = ui_page.dfg()[node_id].unit_tests or []
            self.edit_checks(node_id)

    def show_node_details(self, node: Node):
        with st.container(key="node_checks", border=True):
            st.write("###### Tests")
            unit_tests = node.unit_tests or []
            if unit_tests:
                st.write("\n".join([f"* {x}" for x in unit_tests]))
            else:
                st.write("*Edit node to add tests!*")

        super().show_node_details(node)

    def node_parts_for_diagram(self):
        keys = [
            "pill",
            "messages",
            "unit_tests",
            "requirements",
            "function_return_type",
        ]
        if show_code():
            keys += ["code"]
        return keys

    def global_sidebar(self):
        ui_page: UIPage = st.session_state.ui_page

        # st.write("##### Tests")
        # data = [
        #     {"Node": node.pill, "Tests": len(node.unit_tests or [])}
        #     for node in ui_page.dfg().nodes
        # ]
        # data.sort(key=lambda x: x["Tests"])
        # st.table(data)

        failed_tests = [
            node.pill
            for node in ui_page.dfg().nodes
            if node.filter_messages(Phase.unit_tests_checked, "error")
        ]
        if failed_tests:
            items = "\n".join([f"* {x}" for x in failed_tests])
            st.error(f"**These nodes have failing tests:**\n{items}")

        warnings = [
            node.pill
            for node in ui_page.dfg().nodes
            if node.filter_messages(Phase.unit_tests_code, "warning")
        ]
        if warnings:
            items = "\n".join([f"* {x}" for x in warnings])
            st.warning(f"**These nodes have test warnings:**\n{items}")

        nodes_with_no_tests = [
            node.pill for node in ui_page.dfg().nodes if not node.unit_tests
        ]
        nodes_with_no_tests.sort()

        if nodes_with_no_tests:
            items = "\n".join([f"* {x}" for x in nodes_with_no_tests])
            st.info(f"**These nodes have no tests:**\n{items}")

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
                        [x.id for x in ui_page.dfg().nodes if not x.unit_tests],
                        target_phase=Phase.unit_tests_code,
                        passes_key="suggest-unit-test-passes",
                        force=True,
                        repair=False,
                    )
                    st.session_state.builder_progress = 0

        self.help_details()

    def filter_messages(self, node: Node) -> List[NodeMessage]:
        return [
            x
            for x in node.messages
            if x.phase <= Phase.run_checked or x.phase >= Phase.unit_tests_code
        ]

import os
import trace
import traceback
from regex import E
import streamlit as st

# from flowco.ui.page_files.ama_page import AMAPage
from flowco.dataflow.phase import Phase
from flowco.ui.page_files.help_page import HelpPage
from flowco.ui.page_files.v1.build_page import BuildPage as BuildPageV1
from flowco.ui.page_files.v1.check_page import CheckPage as CheckPageV1
from flowco.ui.page_files.v1.projects_page import ProjectsPage as ProjectsPageV1
from flowco.ui.page_files.v1.test_page import TestPage as TestPageV1

from flowco.ui.page_files.v2.build_page import BuildPage as BuildPageV2
from flowco.ui.page_files.v2.check_page import CheckPage as CheckPageV2
from flowco.ui.page_files.v2.projects_page import ProjectsPage as ProjectsPageV2
from flowco.ui.page_files.v2.test_page import TestPage as TestPageV2


from flowco.util.output import error


def st_pages():

    def version():
        return st.session_state.get("ui_version", 1)

    def build_main():
        st.session_state.current_page = "build"
        # st.session_state.image_cache.clear()
        # st.session_state.selected_node = "<<<<<"
        (BuildPageV1() if version() == 1 else BuildPageV2()).main()

    def check_main():
        st.session_state.current_page = "check"
        # st.session_state.image_cache.clear()
        # st.session_state.selected_node = "<<<<<"
        (CheckPageV1() if version() == 1 else CheckPageV2()).main()

    def test_main():
        st.session_state.current_page = "test"
        # st.session_state.image_cache.clear()
        # st.session_state.selected_node = "<<<<<"
        (TestPageV1() if version() == 1 else TestPageV2()).main()

    def projects_main():
        st.session_state.current_page = "projects"
        # st.session_state.image_cache.clear()
        st.session_state.selected_node = "<<<<<"
        (ProjectsPageV1() if version() == 1 else ProjectsPageV2()).main()

    def help_main():
        st.session_state.current_page = "help"
        # st.session_state.image_cache.clear()
        st.session_state.selected_node = "<<<<<"
        HelpPage().main()

    dfg = st.session_state.ui_page.dfg()
    warn_run_messages = dfg.filter_messages(
        [x for x in Phase if x.value <= Phase.run_checked.value],
        level="warning",
    )
    warn_check_messages = dfg.filter_messages(
        [Phase.assertions_code, Phase.assertions_checked],
        level="warning",
    )
    warn_test_messages = dfg.filter_messages(
        [Phase.unit_tests_code, Phase.unit_tests_checked],
        level="warning",
    )
    error_run_messages = dfg.filter_messages(
        [x for x in Phase if x.value <= Phase.run_checked.value],
        level="error",
    )
    error_check_messages = dfg.filter_messages(
        [Phase.assertions_code, Phase.assertions_checked],
        level="error",
    )
    error_test_messages = dfg.filter_messages(
        [Phase.unit_tests_code, Phase.unit_tests_checked],
        level="error",
    )

    if st.session_state.builder is None and not st.session_state.ama_responding:
        pages = [
            st.Page(
                projects_main,
                title="Projects",
                default=st.session_state.ui_page is None,
            ),
            st.Page(
                build_main,
                title="Edit",
                icon="⛔️" if error_run_messages else "⚠️" if warn_run_messages else None,
                default=st.session_state.ui_page is not None,
            ),
            st.Page(
                check_main,
                title="Check",
                icon=(
                    "⛔️"
                    if error_check_messages
                    else "⚠️" if warn_check_messages else None
                ),
            ),
            st.Page(
                test_main,
                title="Test",
                icon=(
                    "⛔️" if error_test_messages else "⚠️" if warn_test_messages else None
                ),
            ),
            st.Page(
                help_main,
                title="Help",
            ),
            # st.Page(test_main, title="Test"),
        ]
    else:
        if st.session_state.current_page == "build":
            pages = [
                st.Page(build_main, title="Edit"),
            ]
        elif st.session_state.current_page == "check":
            pages = [
                st.Page(check_main, title="Check"),
            ]
        elif st.session_state.current_page == "test":
            pages = [
                st.Page(test_main, title="Test"),
            ]
        else:
            assert (
                False
            ), f"Builder running from bad page {st.session_state.current_page}"

    pg = st.navigation(pages)
    return pg

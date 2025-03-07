import os
import trace
import traceback
from regex import E
import streamlit as st

# from flowco.ui.page_files.ama_page import AMAPage
from flowco.ui.page_files.build_page import BuildPage
from flowco.ui.page_files.check_page import CheckPage
from flowco.ui.page_files.help_page import HelpPage
from flowco.ui.page_files.projects_page import ProjectsPage
from flowco.ui.page_files.test_page import TestPage
from flowco.util.output import error


def st_pages():
    def build_main():
        st.session_state.current_page = "build"
        # st.session_state.image_cache.clear()
        # st.session_state.selected_node = "<<<<<"
        BuildPage().main()

    def check_main():
        st.session_state.current_page = "check"
        # st.session_state.image_cache.clear()
        # st.session_state.selected_node = "<<<<<"
        CheckPage().main()

    def test_main():
        st.session_state.current_page = "test"
        # st.session_state.image_cache.clear()
        # st.session_state.selected_node = "<<<<<"
        TestPage().main()

    def projects_main():
        st.session_state.current_page = "projects"
        # st.session_state.image_cache.clear()
        st.session_state.selected_node = "<<<<<"
        ProjectsPage().main()

    def help_main():
        st.session_state.current_page = "help"
        # st.session_state.image_cache.clear()
        st.session_state.selected_node = "<<<<<"
        HelpPage().main()

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
                default=st.session_state.ui_page is not None,
            ),
            st.Page(
                check_main,
                title="Check",
            ),
            st.Page(
                test_main,
                title="Test",
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

import trace
import traceback
from regex import E
import streamlit as st

# from flowco.ui.page_files.ama_page import AMAPage
from flowco.ui.page_files.build_page import BuildPage
from flowco.ui.page_files.projects_page import ProjectsPage


def st_pages():
    def build_main():
        st.session_state.current_page = "build"
        try:
            BuildPage().main()
        except Exception as e:
            print(e)
            traceback.print_exc()

    # def ama_main():
    #     if st.session_state.current_page != "ama":
    #         st.session_state.selected_node = "<<<<<"
    #     st.session_state.current_page = "ama"
    #     AMAPage().main()

    def test_main():
        st.write("Test")

    # build = st.Page(build_main, title="Build", default=st.session_state.ui_page is not None)

    def projects_main():
        st.session_state.current_page = "projects"
        st.session_state.selected_node = "<<<<<"
        ProjectsPage().main()

    if st.session_state.builder is None:
        pages = [
            st.Page(
                projects_main, title="Browse", default=st.session_state.ui_page is None
            ),
            st.Page(
                build_main, title="Edit", default=st.session_state.ui_page is not None
            ),
            st.Page(test_main, title="Test"),
            # st.Page(ama_main, title="Ask Me Anything"),
        ]
    else:
        pages = [
            st.Page(build_main, title="Edit"),
        ]

    pg = st.navigation(pages)
    return pg

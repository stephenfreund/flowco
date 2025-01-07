import os
import traceback
from flowco.pythonshell.shells import PythonShells
from flowco.session.session_file_system import SessionFileSystem
from flowco.ui.ui_args import parse_args
from flowco.ui.ui_init import st_init
from flowco.ui.ui_page import UIPage, set_ui_page
from flowco.ui.ui_st_pages import st_pages
from flowco.util.files import setup_flowco_files
import streamlit as st

from flowco.util.costs import CostTracker
from flowco.util.output import Output, error

from flowco.session.session import StreamlitSession, session
from flowco.util.stopper import Stopper


def init_service():
    if "service_initialized" not in st.session_state:

        global session
        session = StreamlitSession()

        st.session_state.args = parse_args()

        # if st.session_state.args.page is a directory...
        if os.path.isdir(st.session_state.args.page):
            page_path = os.path.abspath(st.session_state.args.page)
            page_file = None
        else:
            page_path = os.path.abspath(os.path.dirname(st.session_state.args.page))
            page_file = os.path.basename(st.session_state.args.page)

        session.set(
            output=Output(),
            costs=CostTracker(),
            stopper=Stopper(),
            shells=PythonShells(),
            filesystem=SessionFileSystem(f"file://{page_path}"),
        )
        setup_flowco_files()
        if page_file is not None:
            set_ui_page(UIPage(page_file))
        else:
            set_ui_page(UIPage("welcome.flowco"))

        if os.environ.get("OPENAI_API_KEY", None) is None:
            st.write(
                "No OpenAI API key found. Please set the OPENAI_API_KEY environment variable."
            )
            st.stop()

        st.session_state.service_initialized = True


try:
    st_init()
    init_service()
    pg = st_pages()
    pg.run()
except Exception as e:
    error(e)
    error(traceback.format_exc())
    st.error(e)
    st.exception(e)

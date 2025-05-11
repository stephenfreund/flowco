import os
import traceback
from flowco.assistant.flowco_keys import KeyEnv
from flowco.pythonshell.shells import PythonShells
from flowco.session.session_file_system import SessionFileSystem
from flowco.ui.ui_args import parse_args
from flowco.ui.ui_init import st_init
from flowco.ui.ui_page import UIPage, set_ui_page
from flowco.ui.ui_st_pages import st_pages

from flowco.util.files import copy_from_google_folder, setup_flowco_files
import streamlit as st

from flowco.util.costs import CostTracker
from flowco.util.output import Output, error, log_timestamp

from flowco.session.session import StreamlitSession


def init_service():
    if "service_initialized" not in st.session_state:

        global session
        session = StreamlitSession()

        config, args = parse_args()

        # if st.session_state.args.page is a directory...
        if os.path.isdir(args.page):
            page_path = os.path.abspath(args.page)
            page_file = None
        else:
            page_path = os.path.abspath(os.path.dirname(args.page))
            page_file = os.path.basename(args.page)

        session.set(
            config=config,
            output=Output(),
            costs=CostTracker(),
            shells=PythonShells(),
            filesystem=SessionFileSystem(f"file://{page_path}"),
            keys=KeyEnv(),
        )
        log_timestamp()

        st.session_state.user_email = "local"
        new_user = setup_flowco_files()
        if new_user and st.query_params.get("test", None) == "1":
            folder_id = os.environ["GOOGLE_DRIVE_TEST_FOLDER_ID"]
            copy_from_google_folder(folder_id)

        if page_file is not None:
            set_ui_page(UIPage(page_file))
        else:
            set_ui_page(UIPage("welcome.flowco"))

        if os.environ.get("OPENAI_API_KEY", None) is None:
            error(
                "No OpenAI API key found. Please set the OPENAI_API_KEY environment variable."
            )
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
    print(e)
    print(traceback.format_exc())
    st.error(e)
    st.exception(e)
    # try:
    #     zip_bug(f"""{str(e)}\n{traceback.format_exc()}""")
    # except Exception as e2:
    #     print("Error zipping bug report", e2)
    print("Restarting Session Components...")
    config, args = parse_args()
    if os.path.isdir(args.page):
        page_path = os.path.abspath(args.page)
    else:
        page_path = os.path.abspath(os.path.dirname(args.page))
    session.set(
        config=config,
        output=Output(),
        costs=CostTracker(),
        shells=PythonShells(),
        filesystem=SessionFileSystem(f"file://{page_path}"),
        keys=KeyEnv(),
    )
    st.rerun()  # Restart the Streamlit app to reset the session state

import argparse
from flowco.session.session import StreamlitSession, session

import os
import sys
import streamlit as st
from flowco.pythonshell.shells import PythonShells
from flowco.session.session_file_system import SessionFileSystem
from flowco.ui.ui_init import st_init
from flowco.ui.authenticate import authenticate
from flowco.ui.ui_page import UIPage, set_ui_page
from flowco.ui.ui_st_pages import st_pages

from flowco.util.config import Config
from flowco.util.costs import CostTracker
from flowco.util.files import get_flowco_files, setup_flowco_files
from flowco.util.output import Output, log, log_timestamp


@st.cache_resource
def python_shells():
    return PythonShells()


if "user_email" not in st.session_state:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--user_email",
        default=None,
        type=str,
    )
    args, _ = parser.parse_known_args(sys.argv[1:])
    st.session_state.user_email = args.user_email


def init_service():
    st_init(page_config=False)
    if "service_initialized" not in st.session_state:
        global session
        session = StreamlitSession()

        key = st.context.cookies["_streamlit_xsrf"].split("|")[-1]

        session.set(
            config=Config(),
            output=Output(prefix=f"{key}_{st.session_state.user_email}"),
            costs=CostTracker(),
            shells=python_shells(),
            filesystem=SessionFileSystem(
                f"s3://go-flowco/{st.session_state.user_email}"
            ),
        )
        log_timestamp()
        log(f"Initialized session for {st.session_state.user_email}")
        log(f"  key is {key}")
        setup_flowco_files()
        set_ui_page(UIPage(get_flowco_files()[0]))

        if os.environ.get("OPENAI_API_KEY", None) is None:
            st.write(
                "No OpenAI API key found. Please set the OPENAI_API_KEY environment variable."
            )
            st.stop()

        st.session_state.service_initialized = True


if st.session_state.user_email is None:
    if "credentials" not in st.session_state:
        st.session_state.credentials = None

    if st.session_state.credentials is None:
        authenticate()

init_service()
pg = st_pages()
pg.run()

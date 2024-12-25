from flowco.session.session import StreamlitSession, session

import os
import sys
import streamlit as st
from flowco.pythonshell.shells import PythonShells
from flowco.session.session_file_system import SessionFileSystem
from flowco.ui.ui_init import st_init
from flowco.ui.authenticate import authenticate
from flowco.ui.ui_page import UIPage
from flowco.ui.ui_st_pages import st_pages

from flowco.util.config import config
from flowco.util.costs import CostTracker
from flowco.util.files import get_flowco_files, setup_flowco_files
from flowco.util.output import Output, log


@st.cache_data
def parse_args():
    parser = config.parser()
    parser.add_argument(
        "--user_email",
        default=None,
        type=str,
    )
    args = parser.parse_args(sys.argv[1:])
    return args


if "user_email" not in st.session_state:
    st.session_state.args = parse_args()
    st.session_state.user_email = st.session_state.args.user_email


def init_service():
    if "service_initialized" not in st.session_state:
        global session
        session = StreamlitSession()

        key = st.context.cookies["_streamlit_xsrf"].split("|")[-1]

        session.set(
            output=Output(prefix=key),
            costs=CostTracker(),
            filesystem=SessionFileSystem(
                f"s3://go-flowco/{st.session_state.user_email}"
            ),
        )
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
st_init()
pg = st_pages()
pg.run()

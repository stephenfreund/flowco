import argparse
import textwrap
from flowco.assistant.flowco_keys import UserKeys
from flowco.session.session import StreamlitSession, session

import os
import sys
import streamlit as st
from flowco.pythonshell.shells import PythonShells
from flowco.session.session_file_system import SessionFileSystem, fs_exists
from flowco.ui.ui_init import st_init
from flowco.ui.authenticate import authenticate
from flowco.ui.ui_page import UIPage, set_ui_page
from flowco.ui.ui_st_pages import st_pages

from flowco.util.config import Config
from flowco.util.costs import CostTracker
from flowco.util.files import (
    copy_from_google_folder,
    get_flowco_files,
    setup_flowco_files,
)
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


@st.dialog("Welcome to Flowco!", width="large")
def splash_screen():
    st.write(
        textwrap.dedent(
            """\
        ### Getting started
        * You'll begin by editing a simple diagram named "welcome.flowco".  Follow the instructions on the right-hand side of the screen to get started.
        * Switch to other projects by selecting the **Projects** view in the top-left corner.  We recommend following the numbered tutorials to learn more about Flowco.
        * If you have any questions, click the **Help** view in the top-left corner of the screen. 

        ### OpenAI API Key
        * For the next hour, you can use Flowco without providing an OpenAI API key.  After that, you'll need to provide an API key to continue using Flowco.

        ### Bugs
        * While many users have worked in Flowco, there are undoubtedly still bugs.  
        * Please report any bugs you find by clicking the "Report Bug" button at the bottom of the screen.
        """
        )
    )
    st.image("static/flowco.png", width=200)


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
            keys=UserKeys(),
        )
        log_timestamp()
        log(f"Initialized session for {st.session_state.user_email}")
        log(f"  key is {key}")

        if not fs_exists("welcome.flowco"):
            st.toast("**Setting Up Account...**")

        new_user = setup_flowco_files()
        if new_user and st.query_params.get("test", None) == "1":
            folder_id = os.environ["GOOGLE_DRIVE_TEST_FOLDER_ID"]
            copy_from_google_folder(folder_id)

        if fs_exists("welcome.flowco"):
            file = "welcome.flowco"
        else:
            file = get_flowco_files()[0]
        set_ui_page(UIPage(file))

        if new_user:
            splash_screen()

        st.session_state.service_initialized = True


if st.session_state.user_email is None:
    if "credentials" not in st.session_state:
        st.session_state.credentials = None

    if st.session_state.credentials is None:
        authenticate()

init_service()
pg = st_pages()
pg.run()

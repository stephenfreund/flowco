import os
from flowco.assistant.flowco_assistant import test_anthropic_key, test_openai_key
from flowco.assistant.flowco_keys import get_api_key_status, set_api_key
from llm import models
from typing import Callable
import pandas as pd
import streamlit as st
from flowco.dataflow.dfg import Node
from flowco.session.session import session
from flowco.ui.authenticate import cache
from flowco.ui.ui_page import UIPage
from flowco.util.config import config
from flowco.util.config import AbstractionLevel
from flowco.util.files import make_default_files
from flowco.util.output import Output


@st.dialog("Settings", width="large")
def settings(ui_page: UIPage):

    supported_models = models.supported_models()
    current_model = (
        supported_models.index(config().model)
        if config().model in supported_models
        else 0
    )
    model = st.selectbox("LLM model", supported_models, current_model)

    if model != None:
        config().model = model

    st.divider()
    # with st.expander("API Keys"):
    openai_key = st.text_input(
        f"OpenAI API Key",
        value="",
        placeholder="Enter new OpenAI API key",
        # label_visibility="collapsed",
    )

    # anthropic_key = st.text_input(
    #     "Anthropic API Key",
    #     value="",
    #     placeholder="Enter new Anthropic API key",
    #     label_visibility="collapsed",
    # )

    if openai_key:
        set_api_key("OPENAI_API_KEY", openai_key)
        # if anthropic_key:
        #     set_api_key("ANTHROPIC_API_KEY", anthropic_key)

    if st.button("Check OpenAI API Key Status"):
        if test_openai_key():
            st.success("OpenAI API key is valid")
        else:
            st.error("OpenAI API key is invalid")

            # if test_anthropic_key():
            #     st.success("Anthropic key is valid")
            # else:
            #     st.error("Anthropic key is invalid")

            # st.write(get_api_key_status("OPENAI_API_KEY"))
            # st.write(get_api_key_status("ANTHROPIC_API_KEY"))

    st.divider()

    config().zero_temp = st.toggle("Zero temperature", value=config().zero_temp)

    config().debug = st.toggle("Show llm messages [debugging]", value=config().debug)

    config().retries = int(st.number_input("Repair retries", value=config().retries))

    if st.button("Reset Demo Files"):
        with st.spinner("Resetting demo files"):
            make_default_files()

    with st.expander("Experimental Features"):
        config().x_no_right_panel = st.toggle(
            "Hide right panel", value=config().x_no_right_panel
        )

        config().x_no_image_cache = st.toggle(
            "Don't cache images", value=config().x_no_image_cache
        )

        config().x_lock_stops_updates = st.toggle(
            "Use the LLM to check for precondition changes",
            value=config().x_lock_stops_updates,
        )
        config().x_no_dfg_image_in_prompt = st.toggle(
            "Dont' send dataflow image in prompt",
            value=config().x_no_dfg_image_in_prompt,
        )
        config().x_trust_ama = st.toggle(
            "Trust AMA to provide correct completions", value=config().x_trust_ama
        )
        config().x_algorithm_phase = st.toggle(
            "Include algorithm phase", value=config().x_algorithm_phase
        )

        builders = config().get_build_passes_keys()
        config().builder = st.selectbox(
            "Builder", builders, index=builders.index(config().builder)
        )

    if st.button("Done"):
        st.session_state.selected_node = None

        st.rerun()

    st.divider()

    # Read the commit SHA and build date from environment variables
    release = os.getenv("RELEASE_VERSION", "unknown")
    commit_sha = os.getenv("COMMIT_SHA", "unknown")[:7]
    build_date = os.getenv("BUILD_DATE", "unknown")

    st.write(f"Flowco Release {release}, {build_date}")
    st.write("[Release Notes](https://github.com/stephenfreund/flowco/releases)")

    try:
        session_info = f"**Session Info:** {st.session_state.user_email}   {session.get('output', Output).prefix}"
    except:
        session_info = "No session info available"
    st.caption(session_info)


@st.dialog("Data file", width="large")
def show_file(self, file_name: str):
    st.write(f"### {file_name}")
    df = pd.read_csv(file_name)
    st.dataframe(df)


# @st.dialog("Report", width="large")
# def run_report():
#     report = Report()
#     main = st.empty()
#     with main.container(height=600):
#         st.write_stream(report.make(ui_page.page()))

#     with main.container(height=600):
#         st.markdown(report.with_embedded_images, unsafe_allow_html=True)

#     @st.fragment
#     def download():
#         st.download_button(
#             "Download", report.with_embedded_images, f"{page.file_name}.md"
#         )

#     download()


@st.dialog("Confirm", width="small")
def confirm(message: str, on_confirm: Callable[[], None]):
    st.write(message)
    if st.button("OK"):
        on_confirm()
        st.rerun()

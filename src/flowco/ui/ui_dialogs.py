from typing import Callable
import pandas as pd
import streamlit as st
from flowco.assistant.openai import OpenAIAssistant
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

    models = OpenAIAssistant.supported_models()
    current_model = models.index(config.model) if config.model in models else 0
    model = st.selectbox("LLM model", models, current_model)
    if model != None:
        config.model = model

    config.zero_temp = st.toggle("Zero temperature", value=config.zero_temp)

    config.debug = st.toggle("Show llm messages [debugging]", value=config.debug)

    config.retries = int(st.number_input("Repair retries", value=config.retries))

    if st.button("Reset Demo Files"):
        make_default_files()

    with st.expander("Experimental Features"):
        config.x_no_image_cache = st.toggle(
            "Don't cache images", value=config.x_no_image_cache
        )

        config.x_no_descriptions = st.toggle(
            "Don't generate descriptions for each node", value=config.x_no_descriptions
        )
        config.x_lock_stops_updates = st.toggle(
            "Use the LLM to check for precondition changes",
            value=config.x_lock_stops_updates,
        )
        config.x_no_dfg_image_in_prompt = st.toggle(
            "Dont' send dataflow image in prompt", value=config.x_no_dfg_image_in_prompt
        )
        config.x_trust_ama = st.toggle(
            "Trust AMA to provide correct completions", value=config.x_trust_ama
        )
        config.x_algorithm_phase = st.toggle(
            "Include algorithm phase", value=config.x_algorithm_phase
        )

        builders = config.get_build_passes_keys()
        config.builder = st.selectbox(
            "Builder", builders, index=builders.index(config.builder)
        )

    if st.button("Done"):
        st.session_state.selected_node = None
        st.rerun()

    st.divider()

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


@st.dialog("Report", width="large")
def run_report():
    report = Report()
    main = st.empty()
    with main.container(height=600):
        st.write_stream(report.make(ui_page.page()))

    with main.container(height=600):
        st.markdown(report.with_embedded_images, unsafe_allow_html=True)

    @st.fragment
    def download():
        st.download_button(
            "Download", report.with_embedded_images, f"{page.file_name}.md"
        )

    download()


@st.dialog("Confirm", width="medium")
def confirm(message: str, on_confirm: Callable[[], None]):
    st.write(message)
    if st.button("OK"):
        on_confirm()
        st.rerun()

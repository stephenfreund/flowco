import os

import numpy as np
from flowco.assistant.flowco_assistant import test_anthropic_key, test_openai_key
from flowco.assistant.flowco_keys import get_api_key_status, set_api_key
from flowco.llm import models
from typing import Callable
import pandas as pd
import streamlit as st
from flowco.dataflow.dfg import Node
from flowco.page.output import OutputType
from flowco.session.session import session
from flowco.ui.authenticate import cache
from flowco.ui.ui_page import UIPage
from flowco.util.config import config
from flowco.util.config import AbstractionLevel
from flowco.util.files import make_default_files
from flowco.util.output import Output


@st.dialog("Settings", width="large")
def settings(ui_page: UIPage):

    # with st.expander("API Keys"):
    openai_key = st.text_input(
        f"OpenAI API Key",
        value="",
        placeholder="Enter new OpenAI API key",
        # label_visibility="collapsed",
    )
    st.caption(
        "You will not need a key for the first hour.  You can obtain a key [Get a key here](https://platform.openai.com/account/api-keys).  Your key will be stored on our server and will only be used by you."
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

    supported_models = models.supported_models()
    current_model = (
        supported_models.index(config().model)
        if config().model in supported_models
        else 0
    )
    model = st.selectbox("LLM model", supported_models, current_model)

    if model != None:
        config().model = model

    config().zero_temp = st.toggle(
        "Use zero temperature for LLM", value=config().zero_temp
    )

    # config().debug = st.toggle("Show llm messages [debugging]", value=config().debug)

    # config().retries = int(st.number_input("Repair retries", value=config().retries))

    # if st.button("Reset Demo Files"):
    #     with st.spinner("Resetting demo files"):
    #         make_default_files()

    # with st.expander("Experimental Features"):
    #     config().x_no_right_panel = st.toggle(
    #         "Hide right panel", value=config().x_no_right_panel
    #     )

    #     config().x_no_image_cache = st.toggle(
    #         "Don't cache images", value=config().x_no_image_cache
    #     )

    #     config().x_lock_stops_updates = st.toggle(
    #         "Use the LLM to check for precondition changes",
    #         value=config().x_lock_stops_updates,
    #     )
    #     config().x_no_dfg_image_in_prompt = st.toggle(
    #         "Dont' send dataflow image in prompt",
    #         value=config().x_no_dfg_image_in_prompt,
    #     )
    #     config().x_trust_ama = st.toggle(
    #         "Trust AMA to provide correct completions", value=config().x_trust_ama
    #     )
    #     config().x_algorithm_phase = st.toggle(
    #         "Include algorithm phase", value=config().x_algorithm_phase
    #     )

    #     builders = config().get_build_passes_keys()
    #     config().builder = st.selectbox(
    #         "Builder", builders, index=builders.index(config().builder)
    #     )

    st.divider()

    st.toggle(
        "Enable New Diagram Editor",
        value=st.session_state.ui_version == 2,
        key="ui_version_toggle",
        on_change=lambda: st.session_state.update(
            {"ui_version": 2 if st.session_state.ui_version_toggle else 1}
        ),
    )

    st.divider()

    if st.button("Done"):
        st.session_state.selected_node = None
        st.rerun()

    # st.divider()

    # # Read the commit SHA and build date from environment variables
    release = os.getenv("RELEASE_VERSION", "unknown")
    commit_sha = os.getenv("COMMIT_SHA", "unknown")[:7]
    build_date = os.getenv("BUILD_DATE", "unknown")

    st.caption(f"Flowco Release {release}, {commit_sha}, {build_date}")

    # try:
    #     session_info = f"**Session Info:** {st.session_state.user_email}   {session.get('output', Output).prefix}"
    # except:
    #     session_info = "No session info available"
    # st.caption(session_info)


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


def inspect_node(node: Node):
    @st.dialog(node.pill, width="large")
    def show_node(node: Node):
        if node is not None and node.result is not None:
            if (
                node.result.result is not None
                and node.function_return_type is not None
                and not node.function_return_type.is_None_type()
            ):
                value = node.result.result.to_value()
                if type(value) in [np.ndarray, list, pd.Series]:
                    value = pd.DataFrame(value)
                if type(value) == pd.DataFrame:
                    st.dataframe(value, hide_index=True, height=400)
                elif type(value) == dict:
                    for k, v in list(value.items())[0:10]:
                        st.write(f"**{k}**:")
                        if type(v) in [np.ndarray, list, pd.Series]:
                            v = pd.DataFrame(v)
                        if type(v) == pd.DataFrame:
                            st.dataframe(v, hide_index=True, height=200)
                        elif type(v) == dict:
                            st.json(v)
                        elif type(v) == str:
                            if v.startswith("{" or v.startswith("[")):
                                st.json(v)
                            else:
                                st.code(v)
                        else:
                            st.code(v)
                    if len(value) > 10:
                        st.write(f"And {len(value)-10} more...")
                elif type(value) == str:
                    if value.startswith("{" or value.startswith("[")):
                        st.json(value)
                    else:
                        st.code(value)
                else:
                    st.code(value)
            elif node.result.output is not None:
                output = node.result.output
                if output is not None:
                    if output.output_type == OutputType.text:
                        st.text(f"```{output.data}\n```")
                    elif output.output_type == OutputType.image:
                        base64encoded = output.data.split(",", maxsplit=1)
                        image_data = base64encoded[0] + ";base64," + base64encoded[1]
                        st.image(image_data)

    show_node(node)

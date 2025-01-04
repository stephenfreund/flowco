import os
import uuid
import streamlit as st

from flowco.util.config import config


@st.cache_resource
def custom_css():
    #
    # stSkeleton height shouldb e 0
    #
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    filename = "special.css"
    file_path = os.path.join(script_dir, filename)
    with open(file_path, "r") as file:
        return file.read()


def st_init(page_config=True):
    if page_config:
        st.set_page_config(layout="wide", page_icon=":material/account_tree:")

    st.markdown(f"<style>{custom_css()}</style>", unsafe_allow_html=True)

    if "init" not in st.session_state:

        st.session_state.init = True
        st.session_state.last_sequence_number = -1

        st.session_state.nonce = uuid.uuid4().hex

        st.session_state.selected_node = None

        st.session_state.ama_responding = False

        st.session_state.trigger_build_toggle = None
        st.session_state.builder = None
        st.session_state.builder_progress = 0.0

        st.session_state.force_update = False
        st.session_state.clear_graph = False

        st.session_state.abstraction_level = config.abstraction_level
        st.session_state.show_requirements = True
        st.session_state.show_algorithm = True
        st.session_state.show_code = True

        st.session_state.show_description = False
        st.session_state.show_output = True
        st.session_state.show_ama = True

        st.session_state.current_page = None

        st.session_state.edited_node = None
        st.session_state.edit_assistant = None
        st.session_state.edit_assistant_history = None

        st.session_state.generate = False
        st.session_state.copy_generated = False
        st.session_state.chat_command = None

        st.session_state.ama = None

        st.session_state.code_editor = None

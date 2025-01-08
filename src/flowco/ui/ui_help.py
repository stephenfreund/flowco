import os
import streamlit as st


@st.dialog("Help", width="large")
def help_dialog():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    help_file = os.path.join(this_dir, "help.md")
    with open(help_file, "r") as help_file:
        help_text = help_file.read()
        with st.container():
            st.markdown(help_text)

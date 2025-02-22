from dataclasses import dataclass
from typing import Any, Dict, List

import seaborn as sns

from code_editor import code_editor
import streamlit as st

from flowco.assistant.flowco_assistant import fast_transcription
from flowco.builder.cache import BuildCache
from flowco.builder.synthesize import algorithm, requirements, compile
from flowco.dataflow.dfg import DataFlowGraph, Node
from flowco.dataflow.extended_type import schema_to_text
from flowco.dataflow.phase import Phase
from flowco.page.ama_node import AskMeAnythingNode
from flowco.page.page import Page
from flowco.page.tables import file_path_to_table_name
from flowco.session.session_file_system import fs_glob
from flowco.ui.ui_page import UIPage
from flowco.ui.ui_util import (
    phase_for_last_shown_part,
    show_code,
    visible_phases,
)
from flowco.util.config import config
from flowco.util.output import log, logger


@st.dialog("New Node", width="large")
def new_node_dialog(node):
    page: Page = st.session_state.ui_page.page()
    st.segmented_control(
        "Type",
        default="Compute",
        options=["Table", "Compute", "Plot"],
        key="new_node_type",
    )
    if st.session_state.new_node_type != "Table":
        label = st.text_input(
            "Label",
            "" if node.label == "..." else node.label,
            key="new_node_label",
        )

        if st.button("Save"):
            ui_page: UIPage = st.session_state.ui_page
            new_node = node.update(label=label)
            new_node = ui_page.dfg().update_node_pill(new_node)
            ui_page.update_dfg(ui_page.dfg().with_node(new_node))
            st.session_state.force_update = True
            st.session_state
            st.rerun()

    else:

        files = [file for file in fs_glob("", "*.csv")] + sns.get_dataset_names()
        label = st.pills("File", files, key="new_node_file")

        if st.button("Save", disabled=(label == None)):
            assert label in files
            ui_page: UIPage = st.session_state.ui_page
            new_node = node.update(
                pill=file_path_to_table_name(label), label=f"Load the `{label}` table"
            )
            ui_page.update_dfg(ui_page.dfg().with_node(new_node))
            st.session_state.force_update = True
            st.session_state
            st.rerun()

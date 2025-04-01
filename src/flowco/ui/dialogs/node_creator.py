import seaborn as sns

import streamlit as st

from flowco.dataflow.dfg import NodeKind
from flowco.page.page import Page
from flowco.page.tables import file_path_to_table_name, table_df
from flowco.session.session_file_system import fs_glob, fs_write
from flowco.ui.ui_page import UIPage


@st.dialog("New Node", width="large")
def new_node_dialog(node):
    page: Page = st.session_state.ui_page.page()

    kinds = {
        "Compute a value": NodeKind.compute,
        "Load a dataset": NodeKind.table,
        "Make a plot": NodeKind.plot,
    }

    kind_str = st.selectbox(
        "This node will:",
        options=kinds.keys(),
        index=0,
        key="new_node_type",
        # label_visibility="collapsed",
    )
    assert kind_str in kinds
    kind = kinds[kind_str]

    if kind != NodeKind.table:
        title = (
            "Describe the computation"
            if kind == NodeKind.compute
            else "Describe the plot"
        )
        label = st.text_input(
            title,
            "" if node.label == "..." else node.label,
            key="new_node_label",
        )

        if st.button("Save"):
            ui_page: UIPage = st.session_state.ui_page
            new_node = node.update(label=label, kind=kind)
            new_node = ui_page.dfg().update_node_pill(new_node)
            ui_page.update_dfg(ui_page.dfg().with_node(new_node))
            st.session_state.force_update = True
            st.session_state
            st.rerun()

    else:

        placeholder = st.empty()

        uploaded_file = st.file_uploader(
            "Upload new dataset", type=["csv"], accept_multiple_files=False
        )
        if uploaded_file is not None:
            fs_write(uploaded_file.name, uploaded_file.getvalue().decode("utf-8"))
            label = uploaded_file.name
        else:
            files = [file for file in fs_glob("", "*.csv")] + sns.get_dataset_names()
            label = st.pills("Select existing dataset", files, key="new_node_file")

        if label is not None:

            df = table_df(label)
            st.write("**Preview**")
            st.dataframe(df)

            if placeholder.button(f"Use {label}"):
                ui_page: UIPage = st.session_state.ui_page
                new_node = node.update(
                    pill=file_path_to_table_name(label),
                    label=f"Load the `{label}` table",
                    kind=kind,
                )
                ui_page.update_dfg(ui_page.dfg().with_node(new_node))
                st.session_state.force_update = True
                st.session_state
                st.rerun()

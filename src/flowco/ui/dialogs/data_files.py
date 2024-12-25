from io import StringIO
import pandas as pd
import streamlit as st
from flowco.page.tables import file_path_to_table_name
from flowco.session.session_file_system import fs_glob, fs_read, fs_write
from flowco.ui.ui_page import UIPage
from flowco.ui.ui_util import flex_columns


@st.dialog("Manage Data Files", width="large")
def data_files_dialog():

    uploaded_files = st.file_uploader(
        "Upload CSV File", type=["csv"], accept_multiple_files=True
    )
    for uploaded_file in uploaded_files:
        print(uploaded_file.name)
        fs_write(uploaded_file.name, uploaded_file.getvalue().decode("utf-8"))

    ui_page: UIPage = st.session_state.ui_page
    page = ui_page.page()
    tables = page.tables
    files = [file for file in fs_glob("", "*.csv")]
    for file in files:
        with flex_columns():
            cols = st.columns(2)
            with cols[0]:
                include = st.checkbox(
                    file_path_to_table_name(file), value=(tables.contains(file))
                )
            with cols[1]:
                with st.popover("Show"):
                    st.dataframe(
                        pd.read_csv(StringIO(fs_read(file))),
                        selection_mode="single",
                        hide_index=True,
                        use_container_width=True,
                    )
        if include and not tables.contains(file):
            tables = tables.add(file)
        elif not include and tables.contains(file):
            tables = tables.remove(file)

    if st.button("Ok"):
        page.update_tables(tables)
        st.rerun()

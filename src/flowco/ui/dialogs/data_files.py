from io import StringIO
import pandas as pd
import streamlit as st
from flowco.page.tables import file_path_to_table_name
from flowco.session.session_file_system import fs_glob, fs_read, fs_write
from flowco.ui.ui_page import UIPage
from flowco.ui.ui_util import flex_columns

import seaborn as sns


@st.dialog("Select Data Files for Your Project", width="large")
def data_files_dialog():
    ok = st.empty()

    uploaded_files = st.file_uploader(
        "Upload New Dataset", type=["csv"], accept_multiple_files=True
    )
    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            print(uploaded_file.name)
            fs_write(uploaded_file.name, uploaded_file.getvalue().decode("utf-8"))

    ui_page: UIPage = st.session_state.ui_page
    page = ui_page.page()
    tables = page.tables
    files = [file for file in fs_glob("", "*.csv")]
    with st.expander("Your Datasets", expanded=True):
        for file in files:
            with flex_columns():
                cols = st.columns([1, 3])
                name = file_path_to_table_name(file)
                with cols[0]:
                    include = st.checkbox(name, value=(tables.contains(file)))
                with cols[1]:
                    with st.popover("Show"):
                        df = pd.read_csv(StringIO(fs_read(file)))
                        st.write(f"First 10 rows of {name} (out of {len(df)})")
                        st.dataframe(
                            df.head(10),
                            selection_mode="single",
                            hide_index=True,
                            use_container_width=True,
                        )
            if include and not tables.contains(file):
                tables = tables.add(file)
            elif not include and tables.contains(file):
                tables = tables.remove(file)

    with st.expander(
        "Example Datasets",
        expanded=(len(files) == 0)
        or any([tables.contains(x) for x in sns.get_dataset_names()]),
    ):
        for name in sns.get_dataset_names():
            with flex_columns():
                cols = st.columns([1, 3])
                with cols[0]:
                    include = st.checkbox(name, value=(tables.contains(name)))
                with cols[1]:
                    with st.popover("Show"):
                        df = sns.load_dataset(name)
                        st.write(f"First 10 rows of {name} (out of {len(df)})")
                        st.dataframe(
                            df,
                            selection_mode="single",
                            hide_index=True,
                            use_container_width=True,
                        )
            if include and not tables.contains(name):
                tables = tables.add(name)
            elif not include and tables.contains(name):
                tables = tables.remove(name)

    if ok.button("Save Selection"):
        page.update_tables(tables)
        st.rerun()

import streamlit as st

if st.session_state.ui_version == 1:
    from flowco.ui.page_files.v1.build_page import BuildPage as BaseBuildPage
    from flowco.ui.page_files.v1.build_page import BuildButton as BaseBuildButton

    BuildPage = BaseBuildPage
    BuildButton = BaseBuildButton
else:
    from flowco.ui.page_files.v2.build_page import BuildPage as BaseBuildPage
    from flowco.ui.page_files.v2.build_page import BuildButton as BaseBuildButton

    BuildPage = BaseBuildPage
    BuildButton = BaseBuildButton

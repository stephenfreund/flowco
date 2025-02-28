from ast import Global
from io import StringIO
from flowco.dataflow.dfg import Node
from flowco.page.page import Page
from flowco.page.tables import GlobalTables
from flowco.session.session_file_system import (
    fs_copy,
    fs_exists,
    fs_glob,
    fs_rm,
    fs_write,
)

import streamlit as st

from flowco import __main__
from flowco.ui.page_files.base_page import FlowcoPage
from flowco.ui.ui_dialogs import confirm
from flowco.ui.ui_page import set_ui_page
from flowco.ui.ui_page import UIPage
from flowco.util.files import create_zip_in_memory


class ProjectsPage(FlowcoPage):

    def button_bar(self):
        pass

    def graph_is_editable(self) -> bool:
        return False

    @st.dialog("New project", width="large")
    def new_project(self):
        name = st.text_input("Name", placeholder="Project name")
        if name in self.get_project_names():
            st.error("Project already exists.")
        if name.endswith(".flowco"):
            name = name[: -len(".flowco")]
        if name and name not in self.get_project_names():
            st.session_state.just_created_project = True
            self.add_project(name)
            st.rerun()

    @st.dialog("New project", width="large")
    def dup_project(self):
        name = st.text_input("Name", placeholder="Project name")
        if name.endswith(".flowco"):
            name = name[: -len(".flowco")]
        if name in self.get_project_names():
            st.error("Project already exists.")
        if st.button("OK") and name and name not in self.get_project_names():
            fs_copy(st.session_state.ui_page.page().file_name, f"{name}.flowco")
            set_ui_page(UIPage(f"{name}.flowco"))
            st.session_state.selected_node = "<<<<<"
            st.session_state.clear_graph = True
            st.session_state.force_update = True
            st.rerun()

    def select(self):
        if st.session_state.project_name == ":material/add:":
            self.new_project()
        elif st.session_state.project_name == ":material/upload:":
            self.upload_file()
        elif st.session_state.project_name is not None:
            project_name = st.session_state.project_name + ".flowco"
            set_ui_page(UIPage(project_name))
            st.session_state.selected_node = "<<<<<"
            st.session_state.clear_graph = True
            st.session_state.force_update = True
        else:
            st.session_state.project_name = self.get_current_project_name()
        # st.rerun()

    def delete_project(self):
        file = st.session_state.ui_page.page().file_name
        fs_rm(file)
        set_ui_page(None)
        st.session_state.selected_node = None
        st.session_state.force_update = True
        st.session_state.clear_graph = True
        st.rerun()

    def reset_project(self):
        st.session_state.ui_page.page().reset()
        st.session_state.force_update = True
        st.rerun()

    def add_project(self, name: str):
        if not name.endswith(".flowco"):
            name = f"{name}.flowco"
        Page.create(name)
        set_ui_page(UIPage(name))
        st.session_state.selected_node = "<<<<<"
        st.session_state.force_update = True
        st.session_state.clear_graph = True
        # st.rerun()

    def sidebar(self, node: Node | None = None):
        names = self.get_project_names()
        current = self.get_current_project_name()

        st.write("# Projects")
        selected = st.pills(
            "Select a project",
            names + [":material/add:", ":material/upload:"],
            key="project_name",
            default=current,
            on_change=self.select,
            selection_mode="single",
            label_visibility="collapsed",
        )

        if current is not None:
            st.write(f"# {current}.flowco")

            st.info("Select **Edit** above to modify or run the diagram.")

            with st.container(key="page_controls"):
                cols = st.columns(4)
                with cols[0]:
                    st.button(
                        label="",
                        icon=":material/clear_all:",
                        help="Reset all generated content from each node",
                        on_click=lambda: confirm(
                            f"Are you sure you want to reset {current}?  This will clear all generated content from each node.",
                            self.reset_project,
                        ),
                    )
                with cols[1]:
                    st.button(
                        label="",
                        icon=":material/file_copy:",
                        help=f"Duplicate project",
                        on_click=self.dup_project,
                    )
                with cols[2]:
                    st.button(
                        label="",
                        icon=":material/download:",
                        help=f"Download project",
                        on_click=self.download_files,
                    )
                with cols[3]:
                    st.button(
                        label="",
                        icon=":material/delete:",
                        help="Delete project",
                        on_click=lambda: confirm(
                            f"Are you sure you want to delete {current}?",
                            self.delete_project,
                        ),
                        disabled=len(self.get_project_names()) <= 1,
                    )

            ui_page = st.session_state.ui_page
            # st.write(ui_page.dfg().description)

            if st.session_state.just_created_project:
                st.session_state.just_created_project = False

    def get_project_names(self):
        flowco_files = fs_glob("", "*.flowco")
        names = [file[: -len(".flowco")] for file in flowco_files]
        return names

    def get_current_project_name(self):
        return (
            st.session_state.ui_page.page().file_name[: -len(".flowco")]
            if st.session_state.ui_page
            else None
        )

    @st.dialog("Download Files", width="large")
    def download_files(self):
        ui_page = st.session_state.ui_page
        flowco_name = ui_page.page().file_name
        tables = GlobalTables.from_dfg(ui_page.dfg())
        data_files = [
            x for x in tables.all_files() if x.endswith(".csv")
        ]

        with st.spinner("Creating ZIP file..."):
            zip_data = create_zip_in_memory([flowco_name] + data_files)

        st.write("ZIP ready for download!")

        if st.download_button(
            label=":material/download:",
            data=zip_data,
            file_name=f"flowco_files.zip",
            help="Download the project",
        ):
            st.rerun()

    @st.dialog("Upload Project", width="large")
    def upload_file(self):
        uploaded_file = st.file_uploader(
            "Choose a file", type="flowco", accept_multiple_files=False
        )
        if uploaded_file is not None:
            name = uploaded_file.name
            contents = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            if fs_exists(name):
                st.error("Project already exists.")
            else:
                fs_write(name, contents)
                set_ui_page(UIPage(name))
                st.session_state.selected_node = "<<<<<"
                st.session_state.force_update = True
                st.session_state.clear_graph = True
                st.rerun()  # hack -- this is in a callback, so technically a no op.

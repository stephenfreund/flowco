from typing import List, Tuple
import pandas as pd
import streamlit as st
from flowco.dataflow.dfg import Geometry, Node
from flowco.dataflow.phase import Phase
from flowco.ui.dialogs.data_files import data_files_dialog
from flowco.ui.page_files.build_page import BuildPage
from flowco.ui.ui_page import st_abstraction_level
from flowco.ui.ui_page import UIPage
from flowco.ui.ui_util import (
    phase_for_last_shown_part,
    set_session_state,
    show_requirements,
)
from flowco.util.config import config
from flowco.util.output import debug
from flowco.ui.page_files.base_page import FlowcoPage

import queue
import time


from flowco.dataflow import dfg_update
from flowco.ui.ui_builder import Builder
import streamlit as st


from flowco import __main__
from flowco.ui.ui_page import UIPage

from code_editor import code_editor
from flowthon.flowthon import FlowthonProgram

if config.x_algorithm_phase:
    from flowco.ui.dialogs.edit_node import edit_node
else:
    from flowco.ui.dialogs.edit_node_no_alg import edit_node


class CheckPage(BuildPage):

    # Override for other pages

    def update_button_label(self) -> str:
        return ":material/refresh: Update"

    def run_button_label(self) -> str:
        return ":material/play_circle: Check"

    def build_target_phase(self) -> Phase:
        return Phase.assertions_checked

    def pills(self) -> List[Tuple[str, bool]]:
        return super().pills() + [("Checks", True)]

    # override and call super in subclasses
    def node_fields_to_show(self) -> List[str]:
        fields = super().node_fields_to_show()
        if (
            "show_pills" in st.session_state
            and "Checks" in st.session_state.show_pills
            or "Checks" in [pill for pill, show in self.pills() if show]
        ):
            fields.append("assertions")

        return fields

    @st.dialog("Edit Checks", width="large")
    def edit_checks(self, node: Node):
        ui_page: UIPage = st.session_state.ui_page
        editable_df = st.data_editor(
            pd.DataFrame({"checks": (node.assertions or [])}, dtype=str),
            key="edit_checks",
            num_rows="dynamic",
            use_container_width=True,
        )

        if st.button("Ok"):
            text = list(editable_df["checks"])
            ui_page.page().user_edit_node_assertions(node.id, text)
            st.rerun()

    def show_node_details(self, node: Node):
        super().show_node_details(node)

        with st.container(key="node_checks"):
            if "Checks" in st.session_state.show_pills and node is not None:
                st.write("#### Checks")
                st.write("\n".join(["* " + x for x in (node.assertions or [])]))
            if st.button(
                ":material/edit_note:",
                disabled=not self.graph_is_editable(),
                help="Edit node checks",
            ):
                self.edit_checks(node)

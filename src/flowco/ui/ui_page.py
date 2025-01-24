import streamlit as st
import os


from flowco import __main__

from flowco.page.page import Page, PageListener
from flowco.ui import mx_diagram
from flowco.ui.mx_diagram import MxDiagram, UIImageCache
from flowco.dataflow.dfg import DataFlowGraph, Node
from flowco.util.config import config
from flowco.util.output import logger


def load_ui_page(file_name: str):
    if not os.path.exists(file_name):
        # create a new page, and add all csv files in the current directory.
        dir = os.path.dirname(file_name)
        if dir == "":
            dir = "."
        page: Page = Page.create(file_name)

    ui_page = UIPage(file_name)
    return ui_page


def st_abstraction_level():
    if (
        "abstraction_level" not in st.session_state
        or st.session_state.abstraction_level is None
    ):
        st.session_state.abstraction_level = config.abstraction_level
    return st.session_state.abstraction_level


class UIPage(PageListener):

    def __init__(self, file_name: str):
        self._page = Page.from_file(file_name)
        self._page.add_listener(self)

    def page(self):
        return self._page

    def dfg(self):
        return self._page.dfg

    def dfg_as_mx_diagram(self, image_cache: UIImageCache) -> MxDiagram:
        return mx_diagram.from_dfg(self.dfg(), image_cache)

    def node(self, node_id: str) -> Node | None:
        return self.dfg().get_node(node_id)

    def update_dfg(self, dfg: DataFlowGraph):
        self._page.update_dfg(dfg)

    def page_saved(self, page: Page):
        pass

    def page_json(self):
        return self._page.model_dump()

    def can_undo(self):
        return self._page.can_undo()

    def can_redo(self):
        return self._page.can_redo()

    def undo(self):
        self._page.undo()
        st.session_state.clear_graph = True
        st.session_state.force_update = True
        # st.rerun()

    def redo(self):
        self._page.redo()
        st.session_state.clear_graph = True
        st.session_state.force_update = True


def set_ui_page(ui_page: UIPage):
    if "ui_page" in st.session_state and st.session_state.ui_page is not None:
        with logger("closing current page"):
            page = st.session_state.ui_page.page()
    st.session_state.ui_page = ui_page
    st.session_state.image_cache.clear()
    # st.rerun()

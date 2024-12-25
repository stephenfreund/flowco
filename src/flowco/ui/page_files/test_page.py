from flowco.ui.page_files.base_page import FlowcoPage


class TestPage(FlowcoPage):

    def button_bar(self):
        pass

    def graph_is_editable(self) -> bool:
        return True  # False

from time import sleep
import re
import textwrap

import code
from code_editor import code_editor
from flowco.dataflow.extended_type import schema_to_text
from flowco.page.ama import AskMeAnything, VisibleMessage
from flowco.ui.authenticate import sign_out
import numpy as np
import pandas as pd


from flowco.dataflow import dfg_update
from flowco.dataflow.dfg import Node
from flowco.page.output import OutputType
from flowco.ui.page_files.build_page import BuildPage
from flowco.ui.ui_dialogs import settings
from flowco.ui.ui_util import (
    report_bug,
    toggle,
)
import streamlit as st

from streamlit_extras.stylable_container import stylable_container

from flowco import __main__
from flowco.ui.ui_page import UIPage
from flowco.util.config import config
from flowco.util.costs import inflight, total_cost
from flowco.util.output import error
from flowco.llm.assistant import AssistantError
from flowco.assistant.flowco_assistant import fast_transcription


class NBPage(BuildPage):

    def sidebar(self, node: Node | None = None):
        with st.container(key="masthead"):
            self.masthead()
            self.button_bar()

        try:
            tabs = st.tabs(["AMA", "Graph"])
            with tabs[0]:
                self.show_ama()
            with tabs[1]:
                self.right_panel()
        except AssistantError as e:
            error(e)
            st.error(e.message)

    def scroll_to(self, element_id):
        import streamlit.components.v1

        streamlit.components.v1.html(
            f"""
            <script>
                var element = window.parent.document.getElementById("{element_id}");
                element.scrollIntoView({{behavior: 'smooth'}});
            </script>
        """,
            height=0,
            width=0,
        )

    def masthead(self, node=None):
        super().masthead(node)

    # def button_bar(self):
    #     dfg = st.session_state.ui_page.dfg()
    #     pills = [dfg[x].pill for x in dfg.topological_sort()]
    #     s = st.pills("Pills", pills, key="pills", default=None, selection_mode="single")
    #     if s is not None:
    #         self.scroll_to(s.lower())
    #         st.session_state.pills = None

    # def write_ama_message(self, message: VisibleMessage):
    #     with st.chat_message(message.role):
    #         if not message.is_error:
    #             st.markdown(message.content, unsafe_allow_html=True)
    #         else:
    #             st.error(message.content)

    # def show_ama(self):
    #     page = st.session_state.ui_page.page()
    #     if st.session_state.ama is None or st.session_state.ama.page != page:
    #         st.session_state.ama = AskMeAnything(page)

    #     with st.container():
    #         height = 400 if not config().x_no_right_panel else 200
    #         container = st.container(height=height, border=True, key="chat_container")
    #         with container:
    #             for message in st.session_state.ama.messages():
    #                 self.write_ama_message(message)

    #         if st.audio_input(
    #             "Record a voice message",
    #             label_visibility="collapsed",
    #             key="voice_input",
    #             on_change=lambda: self.ama_voice_input(container),
    #             disabled=not self.graph_is_editable(),
    #         ):
    #             st.rerun()

    #         with st.container(key="ama_columns"):
    #             if prompt := st.chat_input(
    #                 "Ask Me Anything!",
    #                 key="ama_input",
    #                 on_submit=lambda: toggle("ama_responding"),
    #                 disabled=not self.graph_is_editable(),
    #             ):
    #                 self.ama_completion(container, prompt)

    #         if st.session_state.pending_ama:
    #             prompt = st.session_state.pending_ama
    #             st.session_state.pending_ama = None
    #             self.ama_completion(container, prompt)

    # def ama_voice_input(self, container):
    #     toggle("ama_responding")
    #     voice = st.session_state.voice_input
    #     transcription = fast_transcription(voice)
    #     self.ama_completion(container, transcription)

    # def ama_completion(self, container, prompt):
    #     page = st.session_state.ui_page.page()
    #     dfg = page.dfg
    #     ama: AskMeAnything = st.session_state.ama
    #     with container:
    #         with st.chat_message("user"):
    #             st.markdown(prompt)

    #         empty = st.empty()
    #         try:
    #             with empty.chat_message("assistant"):
    #                 response = st.write_stream(
    #                     ama.complete(prompt, st.session_state.selected_node)
    #                 )
    #         except Exception as e:
    #             error(e)
    #         finally:
    #             # self.write_ama_message(ama.last_message())
    #             st.session_state.ama_responding = False
    #             if dfg != page.dfg:
    #                 st.session_state.force_update = True
    #                 self.auto_update()
    #             st.rerun()  # TODO: This could be in a callback!  But should be okay...

    # def show_output(self, node: Node):
    #     if node is not None and node.result is not None:
    #         if (
    #             node.result.result is not None
    #             and node.function_return_type is not None
    #             and not node.function_return_type.is_None_type()
    #         ):
    #             value = node.result.result.to_value()
    #             if type(value) in [np.ndarray, list, pd.Series]:
    #                 value = pd.DataFrame(value)
    #             if type(value) == pd.DataFrame:
    #                 st.dataframe(
    #                     value, hide_index=True, height=200, use_container_width=False
    #                 )
    #             elif type(value) == dict:
    #                 for k, v in list(value.items())[0:10]:
    #                     st.write(f"**{k}**:")
    #                     if type(v) in [np.ndarray, list, pd.Series]:
    #                         v = pd.DataFrame(v)
    #                     if type(v) == pd.DataFrame:
    #                         st.dataframe(
    #                             v, hide_index=True, height=200, use_container_width=None
    #                         )
    #                     elif type(v) == dict:
    #                         st.json(v)
    #                     elif type(v) == str:
    #                         if v.startswith("{" or v.startswith("[")):
    #                             st.json(v)
    #                         else:
    #                             st.code(v)
    #                     else:
    #                         st.code(v)
    #                 if len(value) > 10:
    #                     st.write(f"And {len(value)-10} more...")
    #             elif type(value) == str:
    #                 if value.startswith("{" or value.startswith("[")):
    #                     st.json(value)
    #                 else:
    #                     st.code(value)
    #             else:
    #                 st.code(value)
    #         elif node.result.output is not None:
    #             output = node.result.output
    #             if output is not None:
    #                 if output.output_type == OutputType.text:
    #                     st.text(f"```{output.data}\n```")
    #                 elif output.output_type == OutputType.image:
    #                     base64encoded = output.data.split(",", maxsplit=1)
    #                     image_data = base64encoded[0] + ";base64," + base64encoded[1]
    #                     st.image(image_data)

    # def graph_is_editable(self) -> bool:
    #     return True

    # def bottom_bar(self):
    #     ui_page: UIPage = st.session_state.ui_page
    #     with st.container(key="bottom_bar"):
    #         cols = st.columns(3)
    #         with cols[0]:
    #             if st.button(
    #                 label="",
    #                 icon=":material/settings:",
    #                 help="Change settings",
    #             ):
    #                 settings(ui_page)

    #         with cols[1]:
    #             if st.button(
    #                 label="Report Bug", icon=":material/bug_report:", key="report_bug"
    #             ):
    #                 report_bug()

    #         with cols[2]:
    #             if st.button(
    #                 label="Logout",
    #                 icon=":material/logout:",
    #                 help=f"Sign out {st.session_state.user_email if 'user_email' in st.session_state else ''}",
    #             ):
    #                 sign_out()
    #                 st.rerun()

    # def update_ui_page(self, update: dfg_update.mxDiagramUpdate):
    #     ui_page: UIPage = st.session_state.ui_page

    #     new_dfg = dfg_update.update_dataflow_graph(ui_page.dfg(), update)

    #     if new_dfg != ui_page.dfg():
    #         ui_page.update_dfg(new_dfg)

    def init(self):
        if "selected_node" not in st.session_state:
            st.session_state.selected_node = None

    def fini(self):
        pass

    def extract_body(self, code: str) -> str:
        lines = code.splitlines()

        # 1. Extract imports (everything before the first 'def ')
        def_idx = next(i for i, L in enumerate(lines) if L.strip().startswith("def "))
        imports = "\n".join(lines[:def_idx])

        # 2. Skip a potentially multi-line signature
        paren_depth = 0
        sig_count = 0
        for L in lines[def_idx:]:
            sig_count += 1
            paren_depth += L.count("(") - L.count(")")
            if paren_depth == 0 and L.strip().endswith(":"):
                break

        body_lines = lines[def_idx + sig_count :]

        # 3. Remove any triple-quoted docstring
        filtered = []
        in_doc = False
        doc_delim = None
        for L in body_lines:
            s = L.strip()
            if not in_doc and (s.startswith('"""') or s.startswith("'''")):
                in_doc = True
                doc_delim = s[:3]
                if s.endswith(doc_delim) and len(s) > 3:
                    in_doc = False
                continue
            if in_doc:
                if s.endswith(doc_delim):
                    in_doc = False
                continue
            filtered.append(L)

        # 4. Replace 'return' in the last indented line with '<func_name> ='
        func_name = re.match(r"def\s+(\w+)", lines[def_idx]).group(1)
        if filtered and filtered[-1].strip().startswith("return "):
            filtered[-1] = filtered[-1].replace("return", f"{func_name} =", 1)

        # 5. Dedent results
        dedented_imports = textwrap.dedent(imports).strip()
        dedented_body = textwrap.dedent("\n".join(filtered)).rstrip()

        return (dedented_imports + "\n" + dedented_body).strip()

    def nb(self):
        ui_page = st.session_state.ui_page
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")

        @st.fragment()
        def cell(node: Node):
            id = node.id
            with st.container(key=f"cell_{id}", border=True):
                l, r = st.columns(2, vertical_alignment="bottom")
                with l:
                    st.markdown(f"##### {node.pill}")
                req = "\n".join(["* " + x for x in node.requirements or []])
                code = "\n".join(node.code or [])
                full = f"""\
{node.label}
---------------------------------------------------
{req}
---------------------------------------------------
{code}
"""
                x = code_editor(
                    full,
                    lang="python",
                    response_mode=["blur", "debounce", "select"],
                    key=f"full_{id}",
                )
                if x is not None and x["type"] == "selection":
                    old_node = st.session_state.selected_node
                    st.session_state.selected_node = node.id
                if x is not None and x["type"] in ["change"]:
                    try:
                        label, requirements, code = x["text"].split("---")
                        ui_page.update_dfg(
                            ui_page.dfg().update_node(
                                id,
                                label=label.strip(),
                                requirements=[
                                    x.lstrip("* ").strip()
                                    for x in (requirements or "").splitlines()
                                ],
                                code=[x.strip() for x in (code or "").splitlines()],
                            )
                        )
                    except Exception as e:
                        st.error(e)

                cols = st.columns(2)
                with cols[1]:
                    if node.function_return_type is not None:
                        function_return_type = node.function_return_type
                        if function_return_type is not None:
                            if not function_return_type.is_None_type():
                                st.caption(f"{function_return_type.description}")
                            st.code(schema_to_text(function_return_type.type_schema()))
                with cols[0]:
                    self.show_output(node)

                if st.session_state.selected_node == node.id:
                    st.html(
                        f"""
                            <style>
                            div:has(>div>div.st-key-cell_{id}) {{
                                background-color: #ffAAAA;
                            }}  
                            </style>
                            """
                    )
                else:
                    st.html(
                        f"""
                            <style>
                            div:has(>div>div.st-key-cell_{id}) {{
                                background-color: #FFFFFF;
                            }}  
                            </style>
                            """
                    )

        dfg = st.session_state.ui_page.dfg()
        for node_id in dfg.topological_sort():
            node = dfg.get_node(node_id)
            if node is not None:
                # st.divider()
                cell(node)

    def main(self):

        self.init()

        cols = st.columns([4, 1], gap="small")
        with st.container(key="nb_page"):
            self.nb()

        with st.sidebar:
            self.sidebar()
            st.divider()
            self.bottom_bar()

        self.fini()

    def right_panel(self):
        import streamlit as st
        from streamlit_flow import streamlit_flow
        from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
        from streamlit_flow.state import StreamlitFlowState

        dfg = st.session_state.ui_page.dfg()
        nodes = [
            StreamlitFlowNode(
                pos=(0, 0),
                data={
                    "content": node.pill,
                },
                id=node.id,
            )
            for node in [
                dfg[node_id]
                for node_id in st.session_state.ui_page.dfg().topological_sort()
            ]
        ]

        edges = [
            StreamlitFlowEdge(
                id = edge.id,
                source=edge.src,
                target=edge.dst,
                marker_end="arrow",
            )
            for edge in dfg.edges
        ]

        static_flow_state = StreamlitFlowState(
            nodes,
            edges,
        )
        if (
            "static_flow_state" not in st.session_state
        ):
            st.session_state.static_flow_state = static_flow_state

        from streamlit_flow.layouts import TreeLayout

        flow_state = streamlit_flow(
            "static_flow",
            st.session_state.static_flow_state,
            # fit_view=True,
            show_minimap=False,
            show_controls=False,
            pan_on_drag=False,
            allow_zoom=False,
            get_node_on_click=True,
            layout=TreeLayout(direction="down"),
        )
        st.session_state.static_flow_state = flow_state
        if flow_state.selected_id != st.session_state.selected_node:
            st.session_state.selected_node = flow_state.selected_id
            pill = dfg.get_node(flow_state.selected_id).pill
            self.scroll_to(pill.lower())

        # current = (
        #     st.session_state[f"pill_{id}"]
        #     if f"pill_{id}" in st.session_state
        #     else ["Requirements"]
        # )
        # s = r.segmented_control(
        #     "Mode",
        #     ["Requirements", "Code", "Checks", "Tests"],
        #     key=f"mode_{id}",
        #     default=current,
        #     selection_mode="multi",
        #     on_change=lambda: st.session_state.update(
        #         {"pill_" + id: st.session_state[f"mode_{id}"]}
        #     ),
        #     label_visibility="collapsed",
        # )

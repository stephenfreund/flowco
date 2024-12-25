# import streamlit as st
# from flowco.dataflow.dfg import Node
# from flowco.page.ama import AskMeAnything
# from flowco.ui.page_files.base_page import FlowcoPage


# class AMAPage(FlowcoPage):

#     def button_bar(self):
#         pass

#     def graph_is_editable(self) -> bool:
#         return True

#     def node_sidebar(self, node: Node):
#         self.show_messages(node)
#         self.global_sidebar()
#         self.show_node_details(node)


#     def global_sidebar(self):
#         page = st.session_state.ui_page.page()
#         if st.session_state.ama is None or st.session_state.ama.page != page:
#             st.session_state.ama = AskMeAnything(page)

#         container = st.container(height=400, border=True, key="chat_container")

#         dfg = page.dfg

#         with container:
#             for message in st.session_state.ama.messages():
#                 with st.chat_message(message.role):
#                     st.markdown(message.content, unsafe_allow_html=True)

#         print(st.session_state.selected_node)

#         if prompt := st.chat_input("Ask Me Anything"):
#             print(st.session_state.selected_node)
#             with container:
#                 with st.chat_message("user"):
#                     st.markdown(prompt)

#                 empty = st.empty()
#                 with empty.chat_message("assistant"):
#                     response = st.write_stream(st.session_state.ama.complete(prompt, st.session_state.selected_node))

#                 with empty.chat_message("assistant"):
#                     st.markdown(st.session_state.ama.last_message().content, unsafe_allow_html=True)

#                 if dfg != page.dfg:
#                     st.session_state.force_update = True
#                     st.rerun()

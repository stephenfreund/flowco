# import traceback
# import streamlit as st

# from typing import Callable, List
# import uuid

# from flowco.dataflow.extended_type import ExtendedType
# from flowco.page.api import InputVariable
# from flowco.page.page import Page
# from flowco.ui.old.task_runner import TaskRunner
# from streamlit_extras.stylable_container import stylable_container

# import pandas as pd
# import numpy as np


# @st.dialog("API", width="large")
# def api_config(page: Page):

#     st.write("#### Imported Pages")

#     st.text_area(
#         "Imported Pages",
#         key="imported_pages",
#         value="\n".join(page.api.imported_pages),
#         height=78,
#         label_visibility="collapsed",
#         on_change=lambda: page.update_api(
#             page.api.update(
#                 imported_pages=st.session_state["imported_pages"].split("\n")
#             )
#         ),
#     )

#     st.write("#### Inputs")

#     def on_change(i):
#         if (
#             i == len(page.api.inputs)
#             or st.session_state[f"input_description_{i}"]
#             != page.api.inputs[i].description
#         ):
#             description = st.session_state[f"input_description_{i}"]
#             if description == "":
#                 api = page.api.update(
#                     inputs=page.api.inputs[:i] + page.api.inputs[i + 1 :]
#                 )
#                 page.update_api(api)
#             else:
#                 input = InputVariable.create(description)
#                 api = page.api.update(
#                     inputs=page.api.inputs[:i] + [input] + page.api.inputs[i + 1 :]
#                 )
#                 page.update_api(api)

#     for i, input in enumerate(
#         page.api.inputs
#         + [
#             InputVariable(
#                 description="",
#                 name="placeholder",
#                 extended_type=ExtendedType(type="int"),
#                 default_value_expression="0",
#                 default_value="0",
#             )
#         ]
#     ):
#         with st.container(border=True):
#             # st.text_input("Description", placeholder="Add new input", label_visibility='collapsed', value=input.description, key=f"input_description_{i}", on_change=on_change, args=(i,))
#             if input.description != "":
#                 type_description = input.extended_type.type_description()
#                 st.write(f":orange[`{input.name}` is {type_description}]")
#                 st.write("")

#             st.text_area(
#                 f"Description",
#                 value=input.description,
#                 key=f"input_description_{i}",
#                 height=56,
#                 on_change=on_change,
#                 args=(i,),
#                 label_visibility="collapsed",
#                 placeholder="Enter new input description here...",
#             )

#             if input.description != "":
#                 type_description = input.extended_type.type_description()
#                 st.write(f":orange[The default value is:]")
#                 st.write(eval(input.default_value))
#                 st.write(f":orange[Constructed by: `{input.default_value_expression}`]")
#                 st.write("")

#     st.write("#### Output")
#     if page.dfg:
#         dfg = page.dfg
#         nodes = dfg.nodes
#         output_node_id = page.api.output_node_id
#         if output_node_id not in nodes:
#             output_node_id = None

#         pills = [node.pill for node in nodes]

#         output_node_pill = st.selectbox(
#             "Output Node",
#             pills,
#             label_visibility="collapsed",
#             index=(
#                 pills.index(dfg[output_node_id].pill)
#                 if output_node_id is not None
#                 else None
#             ),
#             placeholder="Select an output node",
#         )
#         if output_node_pill is not None:
#             output_node_id = dfg.node_for_pill(output_node_pill).id
#             node = dfg[output_node_id]
#             if node.function_return_type is not None:
#                 actual_type = node.function_return_type.as_actual_type()
#                 if actual_type is not None:
#                     type_description = node.function_return_type.type_description()
#                     st.markdown(
#                         f":orange[`{node.function_result_var}` is {type_description} {node.function_computed_value}]"
#                     )
#                 st.write("")

#             if output_node_id != page.api.output_node_id:
#                 api = page.api.update(output_node_id=output_node_id)
#                 page.update_api(api)


# # def editable_api(
# #     title: str,
# #     items: List[str],
# #     on_refresh: Callable[[], None],
# #     on_change: Callable[[List[str]], None],
# #     disabled: bool = False,
# # ):
# #     def internal_on_change():
# #         new_list = []
# #         for i in range(len(items) + 1):
# #             # be careful -- some of the text_areas may be gone...
# #             if f"{uid}_{i}" in st.session_state.keys():
# #                 item = st.session_state[f"{uid}_{i}"]
# #                 # print(item)
# #                 if item != "":
# #                     new_list.append(item)
# #         # print(new_list)
# #         # traceback.print_stack()
# #         on_change(new_list)

# #     def internal_refresh():
# #         task_runner = TaskRunner("Generating...", on_refresh)
# #         st.session_state.task = task_runner

# #     uid = str(uuid.uuid4())

# #     st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
# #     st.button(
# #         ":material/refresh:", key=f"{uid}", on_click=internal_refresh, disabled=disabled
# #     )

# #     st.markdown(f"<span style='font-size:14px;'>{title}</span>", unsafe_allow_html=True)
# #     for i, s in enumerate(items + [""]):
# #         st.text_area(
# #             f"Item {i+1}",
# #             value=s,
# #             key=f"{uid}_{i}",
# #             label_visibility="collapsed",
# #             height=56,
# #             on_change=internal_on_change,
# #             disabled=disabled,
# #             placeholder="Enter requirement here...",
# #         )

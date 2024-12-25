# from __future__ import annotations

# import textwrap
# from typing import Iterable, Tuple
# from flowco.assistant.stream import StreamingAssistantWithFunctionCalls
# from flowco.builder import type_ops
# from flowco.builder.passes import GraphView, add_graph_to_assistant
# from flowco.dataflow import extended_type
# from flowco.page.page import Page
# from flowco.util.output import log, logger
# from pydantic import BaseModel, Field
# from IPython.core.interactiveshell import InteractiveShell


# class Hypothesize(BaseModel):
#     markdown: str = Field(default="", title="Markdown content of the hypothesis report")
#     with_embedded_images: str = Field(
#         default="", title="Markdown content of the report with embedded images"
#     )

#     def make(self, page: Page, hypothesis: str) -> Iterable[str]:

#         # Create or get an existing InteractiveShell instance
#         shell = InteractiveShell.instance()

#         # # Update the shell's user namespace with provided globals
#         # if globals_dict:
#         #     shell.user_ns.update(globals_dict)

#         result = shell.run_cell(
#             textwrap.dedent(
#                 """
#                                 import pandas as pd
#                                 import numpy as np
#                                 import matplotlib.pyplot as plt
#                                 import seaborn as sns
#                                 import sklearn
#                                 import scipy
#                                 """
#             )
#         )
#         if result.error_in_exec:
#             raise ValueError("Could not import necessary libraries")

#         locals = "The following variables are defined in the local Python scope:\n"

#         for node in page.dfg.nodes:
#             result = node.result
#             if (
#                 result is not None
#                 and result.result is not None
#                 and node.function_return_type is not None
#                 and node.function_return_type.is_not_NoneType()
#             ):
#                 code = f"{node.function_result_var} = {result.result.to_repr()}"
#                 result = shell.run_cell(code)
#                 if result.error_in_exec:
#                     raise ValueError(f"Could not evaluate {code}")

#                 type_description = node.function_return_type.type_description()
#                 locals += f"* `{node.function_result_var}` is {type_description}\n\n"

#         def python_eval(code: str) -> Tuple[str, str]:
#             """
#             {
#                 "name": "python_eval",
#                 "description": "Exec python code.  Returns the values of all locals written to in the code.  You may assume numpy, scipy, and pandas are available.",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "code": {
#                             "type": "string",
#                             "description": "The Python code to evaluate"
#                         }
#                     },
#                     "required": ["code"]
#                 }
#             }
#             """
#             result = shell.run_cell(code)
#             if result.error_in_exec:
#                 return code, f"Error: Could not evaluate {code}"
#             else:
#                 return code, str(result.result)

#         with logger("Hypothesizing page"):
#             assistant = StreamingAssistantWithFunctionCalls(
#                 [python_eval], ["system-prompt", "hypothesize"], imports=""
#             )
#             assert False, "Add the global table defs here"
#             assistant.add_file_content(list(page.spec.keys()))
#             assistant.add_message("user", page.dfg.to_image_prompt_messages())
#             assistant.add_message("user", page.dfg.outputs_to_prompt_messages())

#             graph_view = GraphView(
#                 graph_fields=["edges", "description"],
#                 node_fields=[
#                     "id",
#                     "pill",
#                     "label",
#                     "predecessors",
#                     "description",
#                     "requirements",
#                     "function_result_var",
#                     "function_return_type",
#                     "function_computed_value",
#                 ],
#             )
#             add_graph_to_assistant(assistant, page.dfg, graph_view)

#             prompt = f"""
#             {locals}

#             Use those variables in any function calls you make.
#             Answer this question:
#             ```
#             {hypothesis}
#             ```
#             """

#             assistant.add_message("user", prompt)

#             for x in assistant:
#                 self.markdown += x
#                 yield x

#             self.with_embedded_images = (
#                 page.dfg.replace_placeholders_with_base64_images(self.markdown)
#             )

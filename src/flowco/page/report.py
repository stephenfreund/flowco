# from __future__ import annotations

# from typing import Iterable
# from flowco.builder.passes import GraphView, add_graph_to_assistant
# from flowco.dataflow.dfg import Node
# from flowco.util.output import log, logger
# from pydantic import BaseModel, Field

# from flowco.assistant.stream import StreamingTextAssistant


# class Report(BaseModel):
#     markdown: str = Field("", title="Markdown content of the report")
#     with_embedded_images: str = Field(
#         "", title="Markdown content of the report with embedded images"
#     )

#     def make(self, page: Page) -> Iterable[str]:
#         from flowco.page.page import Page

#         with logger("Reporting page"):
#             assistant = StreamingTextAssistant("report")
#             assert False, "Add the global table defs here"
#             assistant.add_file_content(list(page.spec.keys()))
#             assistant.add_message("user", page.dfg.to_image_prompt_messages())

#             graph_view = GraphView(
#                 graph_fields=["edges", "description"],
#                 node_fields=[
#                     "id",
#                     "pill",
#                     "label",
#                     "predecessors",
#                     "description",
#                     "requirements",
#                 ],
#             )
#             add_graph_to_assistant(assistant, page.dfg, graph_view)

#             assistant.add_message("user", page.dfg.outputs_to_prompt_messages())

#             for x in assistant:
#                 self.markdown += x
#                 yield x

#             self.with_embedded_images = (
#                 page.dfg.replace_placeholders_with_base64_images(self.markdown)
#             )

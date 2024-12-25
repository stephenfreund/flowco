# from typing import Callable, List, Literal, Tuple, Union
# from pydantic import BaseModel, Field
# from flowco.assistant.assistant import Assistant
# from flowco.page.diagram import (
#     Diagram,
#     Geometry,
# )
# from flowco.page.page import Page
# from flowco.session.session import warn, log


# class DiagramAddNode(BaseModel):
#     key: Literal["add_node"]
#     id: str = Field(description="The id of the node to add")
#     label: str = Field(description="A one sentence summary of this computation step.")
#     x: int = Field(description="The x-coordinate of the node.")
#     y: int = Field(description="The y-coordinate of the node.")


# class DiagramRemoveNode(BaseModel):
#     key: Literal["remove_node"]
#     id: str = Field(description="The id of the node to remove.")


# class DiagramAddEdge(BaseModel):
#     key: Literal["add_edge"]
#     id: str = Field(description="The id of the edge to add.")
#     src: str = Field(description="The id of the source node.")
#     dst: str = Field(description="The id of the target node.")


# class DiagramRemoveEdge(BaseModel):
#     key: Literal["remove_edge"]
#     id: str = Field(description="The id of the edge to remove.")


# class PageSuggestion(BaseModel):
#     description: str = Field(
#         description="An answer to the users question or an outline of the steps to make the requested change."
#     )
#     diagram_ops: List[
#         Union[DiagramAddEdge, DiagramAddNode, DiagramRemoveEdge, DiagramRemoveNode]
#     ] = Field(description="Text responses and graph editing commands.")


# class PageEditor:

#     def __init__(self, page: Page):
#         self.page = page
#         self.assistant = Assistant("page-suggestions")
#         self.assistant.add_file_content(page.spec.keys())

#     def formatter(self, suggestion: PageSuggestion) -> str:
#         return str(suggestion)

#     def update_diagram(
#         self,
#         diagram_ops: List[
#             Union[DiagramAddEdge, DiagramAddNode, DiagramRemoveEdge, DiagramRemoveNode]
#         ],
#     ) -> Diagram:
#         diagram = self.page.diagram
#         for op in diagram_ops:
#             log(op)
#             if op.key == "add_node":
#                 diagram = diagram.with_node(
#                     id=op.id,
#                     label=op.label,
#                     geometry=Geometry(x=op.x, y=op.y, width=200, height=100),
#                 )
#             elif op.key == "remove_node":
#                 diagram = diagram.without_node(op.id)
#             elif op.key == "add_edge":
#                 diagram = diagram.with_edge(id=op.id, src=op.src, dst=op.dst)
#             elif op.key == "remove_edge":
#                 diagram = diagram.without_edge(op.id)
#             else:
#                 warn(f"Unknown diagram operation: {op}")
#         return diagram

#     def to_string(
#         self,
#         diagram_ops: List[
#             Union[DiagramAddEdge, DiagramAddNode, DiagramRemoveEdge, DiagramRemoveNode]
#         ],
#     ) -> str:
#         text = ""
#         for op in diagram_ops:
#             if op.key == "text":
#                 text += op.text + "\n"
#         return text

#     def page_suggestions(self, question, stream_updater: Callable[[str], None] = None):
#         assistant = self.assistant
#         page = self.page
#         assistant.add_message("user", page.dfg.to_image_prompt_messages())
#         assistant.add_json_object(
#             "Here is the dataflow graph matching the diagram",
#             page.diagram.model_dump(
#                 include={
#                     "description": True,
#                     "nodes": {
#                         "__all__": {"id", "pill", "label", "description", "geometry"},
#                     },
#                     "edges": True,
#                 }
#             ),
#         )
#         assistant.add_message("user", question)

#         suggestion = assistant.model_completion(
#             PageSuggestion, lambda x: stream_updater(str(x))
#         )

#         if suggestion:
#             return self.update_diagram(suggestion.diagram_ops)
#         else:
#             return None

#     def add_question(self, question) -> Tuple[str, (Diagram | None)]:
#         page = self.page
#         self.assistant.add_message("user", page.dfg.to_image_prompt_messages())
#         self.assistant.add_json_object(
#             "Here is the current dataflow graph matching the diagram",
#             page.diagram.model_dump(
#                 include={
#                     "description": True,
#                     "nodes": {
#                         "__all__": {"id", "pill", "label", "description", "geometry"},
#                     },
#                     "edges": True,
#                 }
#             ),
#         )
#         self.assistant.add_message("user", question)
#         suggestion = self.assistant.model_completion(PageSuggestion)
#         return suggestion.description, self.update_diagram(suggestion.diagram_ops)

import random
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from flowco.dataflow.dfg import DataFlowGraph, Node, Geometry
from flowco.dataflow.extended_type import ext_type_to_summary
from flowco.util.config import config, AbstractionLevel
from flowco.util.output import log
from flowco.util.text import md_to_html


class DiagramOutput(BaseModel):
    output_type: str
    data: str


class DiagramNode(BaseModel):
    id: str
    pill: str
    label: str
    geometry: Geometry
    output_geometry: Geometry

    phase: int  # Using int to represent the Phase enum
    has_messages: bool
    is_locked: bool
    force_show_output: bool
    build_status: Optional[str] = None
    output: Optional[DiagramOutput] = None

    # html: str = ""


class DiagramEdge(BaseModel):
    id: str
    pill: str
    src: str
    dst: str


class MxDiagram(BaseModel):
    version: int = Field(description="Version of the dfg.")
    nodes: Dict[str, DiagramNode] = Field(
        description="Dictionary of DiagramNodes keyed by node id.",
    )
    edges: Dict[str, DiagramEdge] = Field(
        description="Dictionary of DiagramEdges keyed by edge id.",
    )


class UIImageCache:
    def __init__(self):
        self._cache = {}

    def get(self, key: str):
        if key in self._cache:
            return self._cache[key]
        return None

    def put(self, key: str, value):
        self._cache[key] = value

    def clear(self):
        self._cache = {}


@staticmethod
def get_output(
    node: Node, image_cache: UIImageCache | None = None
) -> Optional[DiagramOutput]:
    if node.result is not None:
        if (
            node.function_return_type is not None
            and not node.function_return_type.is_None_type()
        ):
            text = node.result.pp_result_text(clip=15)
            if text is not None:
                clipped = f"<b>{ext_type_to_summary(node.function_return_type)}</b><pre>{text}</pre>"
                return DiagramOutput(output_type="text", data=clipped)

        text = node.result.pp_output_text(clip=15)
        if text is not None:

            clipped = f"<pre>{text}</pre>"
            return DiagramOutput(output_type="text", data=clipped)

        image_url = node.result.output_image()
        if image_url is not None:
            if image_cache is None:
                return DiagramOutput(output_type="image", data=image_url)
            else:
                if image_cache.get(node.id) == image_url:
                    # log(f"Using cached image for node {node.id}")
                    return DiagramOutput(output_type="image", data="cached")
                else:
                    # log(f"Adding image to cache for node {node.id}")
                    image_cache.put(node.id, image_url)
                    return DiagramOutput(output_type="image", data=image_url)

    return None


@staticmethod
def from_dfg(dfg: DataFlowGraph, image_cache: UIImageCache | None) -> MxDiagram:

    # Create a mapping from node id to Node for easy lookup
    node_dict = {node.id: node for node in dfg.nodes}

    # Convert Nodes
    mx_nodes: Dict[str, DiagramNode] = {}

    for node in dfg.nodes:
        md = node.to_markdown()
        html = md_to_html(md)
        diagram_node = DiagramNode(
            id=node.id,
            pill=node.pill,
            label=node.label,
            geometry=node.geometry,
            phase=node.phase.value,  # Convert Phase enum to int
            has_messages=len(node.messages) > 0,
            output_geometry=(
                node.output_geometry if node.output_geometry else node.geometry
            ),
            is_locked=node.is_locked,
            force_show_output=node.force_show_output,
            output=get_output(node, image_cache),
            build_status=node.build_status,
            # html=html,
        )
        mx_nodes[node.id] = diagram_node

    # Convert Edges
    mx_edges: Dict[str, DiagramEdge] = {}
    for edge in dfg.edges:
        src_node = node_dict.get(edge.src)
        if not src_node:
            raise ValueError(
                f"Source node with id '{edge.src}' not found for edge '{edge.id}'."
            )

        # Assuming the pill for the edge is derived from the source node's pill
        edge_pill = src_node.pill

        diagram_edge = DiagramEdge(
            id=edge.id,
            pill=edge_pill,
            src=edge.src,
            dst=edge.dst,
        )
        mx_edges[edge.id] = diagram_edge

    # Assemble MXDiagram
    mx_diagram = MxDiagram(nodes=mx_nodes, edges=mx_edges, version=dfg.version)
    return mx_diagram

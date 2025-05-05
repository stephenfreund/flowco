from __future__ import annotations
import streamlit as st
from streamlit import runtime
from streamlit.runtime import caching

import base64
from typing import List, Tuple

from flowco.dataflow.dfg import DataFlowGraph, Edge, Node, Geometry, NodeKind
from flowco.dataflow.extended_type import ext_type_to_summary

from flowco.dataflow.phase import Phase
from flowco.util.text import md_to_html
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState


_styles = {
    NodeKind.compute: {
        "display": "flex",
        "justifyContent": "center",
        "alignItems": "center",
        "verticalAlign": "middle",
        "textAlign": "center",
    },
    NodeKind.table: {
        "borderRadius": "50%",
        "display": "flex",
        "justifyContent": "center",
        "alignItems": "center",
        "verticalAlign": "middle",
        "textAlign": "center",
    },
    NodeKind.plot: {
        "borderRadius": "5px",
        "boxShadow": "inset 0 0 0 1px white, inset 0 0 0 2px black,inset 0 0 0 3px white, inset 0 0 0 4px black",
        "display": "flex",
        "justifyContent": "center",
        "alignItems": "center",
        "verticalAlign": "middle",
        "textAlign": "center",
    },
}

_types = {
    NodeKind.compute: "default",
    NodeKind.table: "input",
    NodeKind.plot: "default",
}

_phase_colors = [
    "#dfedf7",  # clean
    "#dfedf7",  # requirements
    "#dfedf7",  # algorithm
    "#dfedf7",  # code
    "#dfedf7",  # runnable
    "#fef2d0",  # run_checked
    "#fef2d0",  # checks compiled
    "#ddf1da",  # checked
    "#fef2d0",  # tests compiled
    "#dedbec",  # tests checked
    "#beb8d9",
]


def update_state(
    state: StreamlitFlowState,
    dfg: DataFlowGraph,
    node_parts: List[str],
    selected_id: str | None = None,
):
    nodes = [flow_node(node, node_parts, node.id == selected_id) for node in dfg.nodes]
    edges = [flow_edge(edge) for edge in dfg.edges]

    # Add output nodes and edges
    for node in dfg.nodes:
        output = flow_output(node)
        if output is not None:
            output_node, output_edge = output
            edges.append(output_edge)
            nodes.append(output_node)

    for node in nodes:
        if "command" in node.data:
            del node.data["command"]

    changed = (nodes != state.nodes) or (edges != state.edges)

    state.nodes = nodes
    state.edges = edges

    return changed


def update_dfg(state: StreamlitFlowState, dfg: DataFlowGraph):
    # Update the DFG with the current state
    for node in state.nodes:
        geometry = Geometry(
            x=node.position["x"],
            y=node.position["y"],
            width=node.style["width"],
            height=node.style["height"],
        )
        if node.id.startswith("output-"):
            original_id = node.id[7:]
            dfg = dfg.update_node(original_id, output_geometry=geometry)
        else:
            dfg = dfg.update_node(
                node.id,
                geometry=geometry,
            )
            dfg_node = dfg.get_node(node.id)
            assert dfg_node is not None, f"Node {node.id} not found in DFG"
            if (
                node.data["pill"] != dfg_node.pill
                or node.data["content"] != dfg_node.label
            ):
                dfg = dfg.update_node(
                    node.id,
                    pill=node.data["pill"],
                    label=node.data["content"],
                    phase=Phase.clean,
                )
    return dfg


def flow_node(node: Node, node_parts: List[str], selected: bool) -> StreamlitFlowNode:

    color = _phase_colors[node.phase.value]
    md = node.to_markdown(node_parts)
    html = md_to_html(md)

    return StreamlitFlowNode(
        id=node.id,
        node_type=_types.get(node.kind, "default"),
        pos=(node.geometry.x, node.geometry.y),
        selectable=True,
        selected=selected,
        connectable=True,
        deletable=True,
        data={
            "pill": node.pill,
            "content": node.label,
            "locked": node.is_locked,
            "show_output": node.force_show_output,
            "blinkText": node.build_status,
            "html": html,
        },
        style=_styles.get(node.kind, {})
        | {
            "backgroundColor": color,
            "width": node.geometry.width,
            "height": node.geometry.height,
        },
    )


# Copied from Streamlit's implementation of st.image
def serve_image(image_url, image_id):

    base64encoded = image_url.split(",")[1]
    image = base64.b64decode(base64encoded)
    mimetype = "image/png"
    url = runtime.get_instance().media_file_mgr.add(
        image, mimetype, "1", image_id, is_for_static_download=True
    )
    print(f"{image_id} -> {url}")
    caching.save_media_data(image, mimetype, image_id)
    host = st.context.url.split("/")[2]
    return f"http://{host}{url}"


# def serve_images():
#     for node in self.dfg.nodes:
#         if node.result is not None:
#             image_url = node.result.output_image()
#             if image_url is not None:
#                 image_id = st.session_state.user_email + "-" + node.id + ".png"
#                 url = self.serve_image(image_url, image_id)
#                 print(f"SERVER {url}")


def get_output(node: Node):
    if node.result is not None:
        if (
            node.function_return_type is not None
            and not node.function_return_type.is_None_type()
        ):
            text = node.result.pp_result_text(clip=15)
            if text is not None:
                return f"<b>{ext_type_to_summary(node.function_return_type)}</b>\n```\n{text}\n```"

        text = node.result.pp_output_text(clip=15)
        if text is not None:
            return f"```\n{text}```"

        image_url = node.result.output_image()
        if image_url is not None:
            image_id = st.session_state.user_email + "-" + node.id + ".png"
            url = serve_image(image_url, image_id)
            return f"""<img src="{url}" style="object-fit:contain; width:100%; max-height:100%; margin:1px;"/>"""

    return None


def flow_output(node: Node) -> Tuple[StreamlitFlowNode, StreamlitFlowEdge] | None:
    if node.force_show_output and node.result is not None:
        output_geometry = node.output_geometry
        if output_geometry is None:
            output_geometry = Geometry(
                x=0,
                y=0,
                width=150,
                height=150,
            )
        output = get_output(node)
        if output is not None:
            flow_node = StreamlitFlowNode(
                id="output-" + node.id,
                type="output",
                pos=(output_geometry.x, output_geometry.y),
                selectable=True,
                connectable=False,
                target_position="left",
                data={
                    "content": output,
                },
                style={
                    "borderRadius": 0,
                    "backgroundColor": "#F0F0F0",
                    "boxShadow": "1px 1px 1px #777777",
                    "color": "#660000",
                    "width": output_geometry.width,
                    "height": output_geometry.height,
                },
            )
            flow_edge = StreamlitFlowEdge(
                id="output-edge-" + node.id,
                source=node.id,
                target=flow_node.id,
                marker_end={
                    "type": "arrowclosed",
                    "color": "#660000",
                    "width": 15,
                    "height": 15,
                },
                style={
                    "stroke": "#660000",
                    "strokeWidth": 1.5,
                    "strokeDasharray": "5,5",
                },
            )
            return flow_node, flow_edge

    return None


def flow_edge(edge: Edge) -> StreamlitFlowEdge:
    return StreamlitFlowEdge(
        id=edge.id,
        source=edge.src,
        target=edge.dst,
        marker_end={
            "type": "arrow",
            "width": 15,
            "height": 15,
        },
    )


def diff_state(state1: StreamlitFlowState, state2: StreamlitFlowState) -> bool:
    if state1.timestamp != state2.timestamp:
        print("DIFF TIMESTAMP", state1.timestamp, state2.timestamp)
        return True
    if len(state1.nodes) != len(state2.nodes):
        print("DIFF NODES", len(state1.nodes), len(state2.nodes))
        return True
    if len(state1.edges) != len(state2.edges):
        print("DIFF EDGES", len(state1.edges), len(state2.edges))
        return True
    for node1, node2 in zip(state1.nodes, state2.nodes):
        if node1.id != node2.id:
            print("DIFF ID", node1.id, node2.id)
            return True
        if node1.position != node2.position:
            print("DIFF POSITION", node1.position, node2.position)
            return True
        if node1.style != node2.style:
            print("DIFF STYLE", node1.style, node2.style)
            return True
    for edge1, edge2 in zip(state1.edges, state2.edges):
        if edge1.id != edge2.id:
            print("DIFF EDGE ID", edge1.id, edge2.id)
            return True
        if edge1.source != edge2.source:
            print("DIFF EDGE SOURCE", edge1.source, edge2.source)
            return True
        if edge1.target != edge2.target:
            print("DIFF EDGE TARGET", edge1.target, edge2.target)
            return True
    return False

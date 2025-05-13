from __future__ import annotations
import streamlit as st
from streamlit import runtime
from streamlit.runtime import caching

import base64
from typing import List, Tuple

from flowco.dataflow.dfg import DataFlowGraph, Edge, Node, Geometry, NodeKind
from flowco.dataflow.extended_type import ext_type_to_summary

from flowco.dataflow.phase import Phase
from flowco.ui.ui_util import phase_for_last_shown_part
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


def update_state_node(node: Node, state_node: StreamlitFlowNode, node_parts: List[str]):
    color = _phase_colors[node.phase.value]
    md = node.to_markdown(node_parts)
    html = md_to_html(md)

    messages = [x for x in node.messages if x.phase <= Phase.run_checked]
    if "assertions" in node_parts:
        messages += [
            x
            for x in node.messages
            if x.phase in [Phase.assertions_code, Phase.assertions_checked]
        ]
    if "unit_tests" in node_parts:
        messages += [
            x
            for x in node.messages
            if x.phase in [Phase.unit_tests_code, Phase.unit_tests_checked]
        ]

    # Update the state node with the current node properties
    state_node.node_type = _types.get(node.kind, "default")
    state_node.pos = (node.geometry.x, node.geometry.y)
    state_node.width = node.geometry.width
    state_node.height = node.geometry.height

    state_node.data = {
        "pill": node.pill,
        "content": node.label,
        "locked": node.is_locked,
        "show_output": node.force_show_output,
        "blinkText": node.build_status,
        "html": html,
        "editable": phase_for_last_shown_part() <= node.phase,
        "messages": messages,
    }

    state_node.style = _styles.get(node.kind, {}) | {
        "backgroundColor": color,
        "width": node.geometry.width,
        "height": node.geometry.height,
    }


def new_state_node(node: Node, node_parts) -> StreamlitFlowNode:
    color = _phase_colors[node.phase.value]
    md = node.to_markdown(node_parts)
    html = md_to_html(md)

    return StreamlitFlowNode(
        id=node.id,
        node_type=_types.get(node.kind, "default"),
        pos=(node.geometry.x, node.geometry.y),
        width=node.geometry.width,
        height=node.geometry.height,
        selectable=True,
        selected=False,
        connectable=True,
        deletable=True,
        data={
            "pill": node.pill,
            "content": node.label,
            "locked": node.is_locked,
            "show_output": node.force_show_output,
            "blinkText": node.build_status,
            "html": html,
            "editable": phase_for_last_shown_part() <= node.phase,
            "messages": [],
        },
        style=(
            _styles.get(node.kind, {})
            | {
                "backgroundColor": color,
                "width": node.geometry.width,
                "height": node.geometry.height,
            }
        ),
    )


def update_state(
    state: StreamlitFlowState,
    dfg: DataFlowGraph,
    node_parts: List[str],
    selected_id: str | None = None,
):

    new_state_nodes = []
    for node in dfg.nodes:
        state_node = next((n for n in state.nodes if n.id == node.id), None)
        if state_node is not None:
            update_state_node(node, state_node, node_parts)
            new_state_nodes.append(state_node)
        else:
            new_state_nodes.append(new_state_node(node, node_parts))

    new_state_edges = []
    for edge in dfg.edges:
        state_edge = next((e for e in state.edges if e.id == edge.id), None)
        if state_edge is not None:
            state_edge.source = edge.src
            state_edge.target = edge.dst
            new_state_edges.append(state_edge)
        else:
            new_state_edges.append(
                StreamlitFlowEdge(
                    id=edge.id,
                    source=edge.src,
                    target=edge.dst,
                    deletable=True,
                    marker_end={
                        "type": "arrow",
                        "width": 15,
                        "height": 15,
                    },
                    labelStyle={"fill": "black"},
                )
            )

    # Add output nodes and edges
    new_state_output_nodes = []
    new_state_output_edges = []
    for node in dfg.nodes:
        if node.result is not None:
            state_output_node = next(
                (n for n in state.nodes if n.id == "output-" + node.id), None
            )
            if state_output_node is not None:
                update_state_output_node(node, state_output_node)
                new_state_output_nodes.append(state_output_node)

                state_output_edge = next(
                    (e for e in state.edges if e.id == "output-edge-" + node.id), None
                )
                new_state_output_edges.append(state_output_edge)
            else:
                new_state_output_nodes.append(new_state_output_node(node))
                new_state_output_edges.append(
                    StreamlitFlowEdge(
                        id="output-edge-" + node.id,
                        source=node.id,
                        source_handle="output",
                        target="output-" + node.id,
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
                        type="smoothstep",
                    )
                )

    old_state_nodes = state.nodes
    old_state_edges = state.edges

    state.nodes = new_state_nodes + new_state_output_nodes
    state.edges = new_state_edges + new_state_output_edges

    changed = old_state_nodes != new_state_nodes or old_state_edges != new_state_edges

    return changed


def update_dfg(state: StreamlitFlowState, dfg: DataFlowGraph):
    # Update the DFG with the current state
    for state_node in state.nodes:
        geometry = Geometry(
            x=state_node.position["x"],
            y=state_node.position["y"],
            width=state_node.width,
            height=state_node.height,
        )
        if state_node.id.startswith("output-"):
            original_id = state_node.id[7:]
            if original_id in dfg.node_ids():
                dfg = dfg.update_node(original_id, output_geometry=geometry)
        else:
            dfg = dfg.update_node(
                state_node.id,
                geometry=geometry,
            )
            dfg_node = dfg.get_node(state_node.id)
            assert dfg_node is not None, f"Node {state_node.id} not found in DFG"
            if (
                state_node.data["pill"] != dfg_node.pill
                or state_node.data["content"] != dfg_node.label
            ):
                dfg = dfg.update_node(
                    state_node.id,
                    pill=state_node.data["pill"],
                    label=state_node.data["content"],
                    phase=Phase.clean,
                )

    new_edges = set()
    for state_edge in state.edges:
        if not state_edge.id.startswith("output-edge-"):
            dfg_edge = next(
                (
                    e
                    for e in dfg.edges
                    if e.src == state_edge.source and e.dst == state_edge.target
                ),
                None,
            )

            if dfg_edge is not None:
                new_edges.add(dfg_edge)
            else:
                src_id = state_edge.source
                dst_id = state_edge.target
                edge = Edge(id=f"{src_id}->{dst_id}", src=src_id, dst=dst_id)
                new_edges.add(edge)
                dfg = dfg.lower_phase_with_successors(dst_id, Phase.clean)
                dfg = dfg.lower_phase_with_successors(src_id, Phase.clean)

    new_nodes = []
    for dfg_node in dfg.nodes:
        state_node = next((n for n in state.nodes if n.id == dfg_node.id), None)
        if state_node is not None:
            new_nodes.append(dfg_node)

    for dfg_edge in dfg.edges:
        state_edge = next(
            (
                e
                for e in state.edges
                if e.source == dfg_edge.src and e.target == dfg_edge.dst
            ),
            None,
        )
        if state_edge is not None:
            new_edges.add(dfg_edge)
        else:
            dfg = dfg.lower_phase_with_successors(dfg_edge.src, Phase.clean)

    print(new_edges)
    dfg = dfg.update(nodes=new_nodes, edges=list(new_edges))

    return dfg


def flow_node(node: Node, node_parts: List[str], selected: bool) -> StreamlitFlowNode:

    color = _phase_colors[node.phase.value]
    md = node.to_markdown(node_parts)
    html = md_to_html(md)

    return StreamlitFlowNode(
        id=node.id,
        node_type=_types.get(node.kind, "default"),
        pos=(node.geometry.x, node.geometry.y),
        width=node.geometry.width,
        height=node.geometry.height,
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
            "editable": phase_for_last_shown_part() <= node.phase,
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


def update_state_output_node(node, state_output_node):
    output_geometry = node.output_geometry
    if output_geometry is None:
        output_geometry = Geometry(
            x=node.geometry.x + node.geometry.width + 50,
            y=node.geometry.y,
            width=150,
            height=150,
        )
    state_output_node.pos = (output_geometry.x, output_geometry.y)
    state_output_node.width = output_geometry.width
    state_output_node.height = output_geometry.height

    output = get_output(node)
    if output is not None:
        state_output_node.data["content"] = output
        state_output_node.style["width"] = output_geometry.width
        state_output_node.style["height"] = output_geometry.height
    else:
        state_output_node.data["content"] = ""


def new_state_output_node(node: Node) -> StreamlitFlowNode:
    output_geometry = node.output_geometry
    if output_geometry is None:
        output_geometry = Geometry(
            x=node.geometry.x + node.geometry.width + 50,
            y=node.geometry.y,
            width=150,
            height=150,
        )
    return StreamlitFlowNode(
        id="output-" + node.id,
        node_type="output",
        pos=(output_geometry.x, output_geometry.y),
        width=output_geometry.width,
        height=output_geometry.height,
        selectable=True,
        connectable=False,
        target_position="left",
        data={
            "content": get_output(node),
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
                width=output_geometry.width,
                height=output_geometry.height,
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
                source_handle="output",
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
                type="smoothstep",
            )
            return flow_node, flow_edge

    return None


def flow_edge(edge: Edge) -> StreamlitFlowEdge:
    return StreamlitFlowEdge(
        id=edge.id,
        source=edge.src,
        target=edge.dst,
        deletable=True,
        marker_end={
            "type": "arrow",
            "width": 15,
            "height": 15,
        },
        labelStyle={"fill": "black"},
    )


def diff_state(state1: StreamlitFlowState, state2: StreamlitFlowState) -> bool:
    # if state1.timestamp != state2.timestamp:
    #     print("DIFF TIMESTAMP", state1.timestamp, state2.timestamp)
    #     return True
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

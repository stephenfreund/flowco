from sys import version
from typing import Dict, List, Optional
from pydantic import BaseModel, ValidationError
from flowco.dataflow.dfg import DataFlowGraph, Edge, Geometry, Group, Node
from flowco.dataflow.phase import Phase
from flowco.ui.mx_diagram import DiagramGroup
from flowco.util.output import log, warn, error
from flowco.util.text import (
    pill_to_function_name,
    pill_to_python_name,
    pill_to_result_var_name,
)


class DiagramNodeUpdate(BaseModel):
    id: str
    pill: str
    label: str
    geometry: Geometry
    output_geometry: Optional[Geometry] = None
    is_locked: bool
    force_show_output: bool


class DiagramEdgeUpdate(BaseModel):
    id: str
    src: str
    dst: str


class mxDiagramUpdate(BaseModel):
    version: int
    nodes: Dict[str, DiagramNodeUpdate]
    edges: Dict[str, DiagramEdgeUpdate]
    groups: List[DiagramGroup]


def update_dataflow_graph(
    current_graph: DataFlowGraph, diagram_update: mxDiagramUpdate
) -> DataFlowGraph:
    """
    Updates the DataFlowGraph based on the mxDiagramUpdate.
    Returns a new DataFlowGraph instance with the updates applied.
    """

    print(diagram_update)

    # Create deep copies of current nodes and edges to avoid mutation
    new_nodes_dict: Dict[str, Node] = {
        node.id: node.model_copy(deep=True) for node in current_graph.nodes
    }
    new_edges_dict: Dict[str, Edge] = {
        edge.id: edge.model_copy(deep=True) for edge in current_graph.edges
    }

    # Handle Nodes
    updated_node_ids = set(diagram_update.nodes.keys())
    current_node_ids = set(new_nodes_dict.keys())

    # **Removal**: Nodes not present in the update should be removed
    nodes_to_remove = current_node_ids - updated_node_ids
    for node_id in nodes_to_remove:
        log(f"Removing node: {node_id}")
        del new_nodes_dict[node_id]
        # Also, remove edges connected to this node
        edges_to_remove = [
            edge_id
            for edge_id, edge in new_edges_dict.items()
            if edge.src == node_id or edge.dst == node_id
        ]
        for edge_id in edges_to_remove:
            log(f"Removing edge due to node removal: {edge_id}")
            del new_edges_dict[edge_id]

    # **Insertion and Modification**: Add or update nodes
    for node_id, node_update in diagram_update.nodes.items():
        if node_id in new_nodes_dict:
            # **Modify Existing Node**
            node = new_nodes_dict[node_id]

            if node.pill != node_update.pill or node.label != node_update.label:
                node.cache = node.cache.invalidate(Phase.requirements)
                warn("Dataflow graph changed, so must re-evaluate requirements.")
                node.phase = Phase.clean

            # Update label and geometry
            node.label = node_update.label
            node.geometry = node_update.geometry
            node.label = node_update.label
            node.pill = node_update.pill
            node.function_name = pill_to_function_name(node_update.pill)
            node.function_result_var = pill_to_result_var_name(node_update.pill)

            output_geometry = node_update.output_geometry
            if output_geometry is None:
                output_geometry = node.output_geometry
            if output_geometry is None:
                output_geometry = node.geometry.translate(
                    node_update.geometry.width + 60, 0
                ).resize(120, 80)

            node.output_geometry = output_geometry

            node.is_locked = node_update.is_locked
            node.force_show_output = node_update.force_show_output

        else:
            pill = node_update.pill
            label = node_update.label

            # Create a new Node with default or placeholder values for missing fields
            new_node = Node(
                id=node_update.id,
                pill=pill,
                label=label,
                geometry=node_update.geometry,
                predecessors=[],  # To be updated based on edges later
                function_name=pill_to_function_name(pill),
                function_result_var=pill_to_result_var_name(pill),
                output_geometry=node_update.output_geometry
                or node_update.geometry.translate(
                    node_update.geometry.width + 100, 0
                ).resize(120, 80),
                is_locked=node_update.is_locked,
                force_show_output=node_update.force_show_output,
            )
            new_nodes_dict[node_update.id] = new_node
            log(f"Inserted new node: {node_id}")

    # Handle Edges
    updated_edge_ids = set(diagram_update.edges.keys())
    current_edge_ids = set(new_edges_dict.keys())

    # **Removal**: Edges not present in the update should be removed
    edges_to_remove = current_edge_ids - updated_edge_ids
    for edge_id in edges_to_remove:
        log(f"Removing edge: {edge_id}")
        to_delete = current_graph.get_edge(edge_id)
        new_nodes_dict[to_delete.dst].phase = Phase.clean
        del new_edges_dict[edge_id]

    # **Insertion and Modification**: Add or update edges
    for edge_id, edge_update in diagram_update.edges.items():
        if edge_id in new_edges_dict:
            edge = new_edges_dict[edge_id]
            assert (
                edge.src == edge_update.src and edge.dst == edge_update.dst
            ), f"Edge '{edge_id}' cannot be modified."
        else:
            # **Insert New Edge**
            # Validate that src and dst nodes exist
            assert (
                edge_update.src in new_nodes_dict
            ), f"Source node '{edge_update.src}' does not exist."
            assert (
                edge_update.dst in new_nodes_dict
            ), f"Destination node '{edge_update.dst}' does not exist."

            # Must regeneration src, because if a new edge is added, the src node may need
            # to be given a real return type...
            new_nodes_dict[edge_update.src].phase = Phase.clean

            new_edge = Edge(id=edge_update.id, src=edge_update.src, dst=edge_update.dst)
            new_edges_dict[edge_id] = new_edge
            log(
                f"Inserted new edge: {edge_id} from '{edge_update.src}' to '{edge_update.dst}'"
            )

    # Update Predecessors based on edges
    # First, clear all predecessors
    for node in new_nodes_dict.values():
        node.predecessors = []

    # Convert dicts back to lists
    updated_nodes = list(new_nodes_dict.values())
    updated_edges = list(new_edges_dict.values())

    # Create the new DataFlowGraph
    try:
        groups = [
            Group(
                id=group.id,
                label=group.label,
                is_collapsed=group.is_collapsed,
                collapsed_geometry=group.collapsed_geometry,
                parent_group=group.parent_group,
                nodes=group.nodes,
            )
            for group in diagram_update.groups
        ]
        print(groups)

        new_graph = DataFlowGraph(
            description=current_graph.description,
            nodes=updated_nodes,
            edges=updated_edges,
            version=diagram_update.version,
            groups=groups,
        )

        def predecessors(node: Node) -> List[str]:
            direct_preds = {edge.src for edge in new_graph.edges if edge.dst == node.id}
            indirect_preds = {
                p for pred in direct_preds for p in predecessors(new_graph[pred])
            }
            preds = list(direct_preds | indirect_preds)
            preds = sorted(preds, key=lambda x: new_graph[x].pill)
            return preds

        for node_id in new_graph.topological_sort():
            new_graph[node_id].predecessors = predecessors(new_graph[node_id])

        return new_graph

    except ValidationError as e:
        error(f"Validation error", e)
        return current_graph

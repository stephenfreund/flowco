from typing import Any, Dict, List, Optional, TypeVar, get_origin

from pydantic import BaseModel, Field, create_model
from typing import get_args, Union

from flowco.assistant.assistant import Assistant
from flowco.dataflow.dfg import (
    DataFlowGraph,
    GraphLike,
    NodeLike,
    Node,
)
from flowco.dataflow.phase import Phase
from flowco.util.output import log


def extract_type(field_annotation) -> type[Any]:
    if get_origin(field_annotation) is Union:
        args = get_args(field_annotation)
        # Filter out NoneType and return the other type
        for arg in args:
            if arg is not type(None):
                return arg
    return field_annotation


def node_completion_model(
    *to_complete: str, include_explanation=False
) -> type[NodeLike]:
    field_names = set(to_complete)

    kwargs = {
        name: (extract_type(field.annotation), Field(description=field.description))
        for name, field in Node.model_fields.items()
        if name in field_names and field.annotation is not None
    }
    if include_explanation:
        kwargs["explanation"] = (
            str,
            Field(description="Explanation of the completion."),
        )

    model = create_model("NodeCompletion", **kwargs)  # type: ignore
    model.model_rebuild()
    return model


def node_like_model(to_include: List[str]) -> type[NodeLike]:

    field_names = set(to_include) | {"id", "pill", "label", "predecessors"}
    assert field_names.issubset(
        Node.model_fields.keys()
    ), f"Fields {field_names.difference(Node.model_fields.keys())} is not in Node model fields."

    kwargs = {
        name: (
            field.annotation,
            Field(
                description=field.description,
            ),
        )
        for name, field in Node.model_fields.items()
        if name in field_names and field.annotation is not None
    }

    return create_model("NodeLike", **kwargs)  # type: ignore


def make_node_like(node: Node, model: type[NodeLike]) -> NodeLike:
    return model.model_validate(node.model_dump(include=set(model.model_fields.keys())))


U = TypeVar("U", bound=NodeLike)


def node_completion(assistant: Assistant, node_like_response_model: type[U]) -> U:
    new_node = assistant.model_completion(node_like_response_model)
    return new_node


def graph_node_like_model(
    node_like_model: type[NodeLike], to_include: List[str]
) -> type[GraphLike]:
    field_names = set(to_include)
    assert field_names.issubset(DataFlowGraph.model_fields.keys() - {"nodes"})

    kwargs = {
        name: (
            field.annotation,
            Field(
                description=field.description,
            ),
        )
        for name, field in DataFlowGraph.model_fields.items()
        if name in field_names and field.annotation is not None
    } | {
        "nodes": (
            List[node_like_model],
            Field(
                description="List of nodes in the data flow graph.",
            ),
        )
    }
    return create_model("GraphLike", **kwargs)  # type: ignore


def make_graph_node_like(dfg: DataFlowGraph, model: type[GraphLike]) -> GraphLike:
    return model.model_validate(dfg.model_dump())


def extend_model_to_include_error_option(model: type[BaseModel]):
    return create_model(
        "OptionalError",
        __doc__="Fill in only one field.  Fill in the error field if you cannot provide a responsible response with high confidence due to ambiguity in the information provided.  Explain exactly why you cannot proceed and why you cannot answer with confidence.",
        error=(
            Optional[str],
            Field(default=None, description="Error message if there is a problem."),
        ),
        result=(
            Optional[model],
            Field(default=None, description="The result of the completion."),
        ),
    )


def messages_for_graph(
    graph: DataFlowGraph,
    graph_fields: List[str] = [],
    node_fields: List[str] = [],
) -> List[str | Dict[str, Any]]:
    initial_graph_model = graph_node_like_model(
        node_like_model(node_fields), graph_fields
    )

    initial_graph = make_graph_node_like(graph, initial_graph_model)

    return [
        {"type": "text", "text": "Here is the dataflow graph"},
        {
            "type": "text",
            "text": initial_graph.model_dump_json(exclude_none=True, indent=2),
        },
    ]


def messages_for_node(
    node: Node, node_fields: List[str] = []
) -> List[str | Dict[str, Any]]:
    initial_node_model = node_like_model(node_fields)

    initial_node = make_node_like(node, initial_node_model)

    return [
        {"type": "text", "text": f"Here is the node named `{node.pill}`"},
        {
            "type": "text",
            "text": initial_node.model_dump_json(exclude_none=True, indent=2),
        },
    ]


def cache_prompt(phase: Phase, node: Node, description: str) -> str:
    """
    Use when it isn't a total match...
    """
    if node.cache.matches_in(phase, node):
        log(f"Rebuild due to user edits to output.")
        # return f"Modify the existing {description} to improve clarity and precision."
        return f"Make as few changes to the existing {description} as possible."
    else:
        log(f"Rebuild due to input changes.")
        diff_map = node.cache.diff(phase, node)
        return f"Here are the changes to reflect in the new {description}:\n```\n{diff_map}\n```\nPreserve any existing {description} as much as possible."


def update_node_with_completion(node: Node, completion: BaseModel) -> Node:
    """
    Updates the `node` instance with fields from the `completion` instance.

    Only the fields that are explicitly set in `completion` will be used to update `node`.

    Args:
        node (Node): The original Node instance to be updated.
        completion (BaseModel): The instance containing update values.

    Returns:
        Node: A new Node instance with updated fields.
    """
    # Dump the node's data
    node_data = node.model_dump()

    # Dump only the fields that are set in completion
    completion_data = completion.model_dump(exclude_unset=True)

    # Merge the dictionaries, with completion_data overriding node_data
    merged_data = node_data | completion_data

    # Validate and create a new Node instance
    updated_node = Node.model_validate(merged_data)

    return updated_node

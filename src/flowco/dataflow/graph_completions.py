from typing import Any, Callable, List, Optional, TypeVar, get_origin

from pydantic import BaseModel, Field, create_model
from typing import get_args, Union

from flowco.assistant.assistant import Assistant
from flowco.dataflow.phase import Phase
from flowco.dataflow.dfg import (
    DataFlowGraph,
    GraphLike,
    NodeLike,
    Node,
    SanityCheck,
)
from flowco.util.config import config


def extract_type(field_annotation) -> type[Any]:
    if get_origin(field_annotation) is Union:
        args = get_args(field_annotation)
        # Filter out NoneType and return the other type
        for arg in args:
            if arg is not type(None):
                return arg
    return field_annotation


def restrict_model_fields(model: type[BaseModel], *to_include: str) -> type[BaseModel]:
    assert set(to_include).issubset(model.model_fields.keys())
    kwargs = {
        name: (
            field.annotation,
            Field(
                description=field.description,
            ),
        )
        for name, field in model.model_fields.items()
    }
    return create_model(model.__name__, **kwargs)  # type: ignore


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


def node_like_model(*to_include: str) -> type[NodeLike]:
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


def node_completion(
    assistant: Assistant, node_like_response_model: type[U], can_fail=False
) -> U:

    if can_fail:
        node_like_response_model = extend_model_to_include_error_option(node_like_response_model)  # type: ignore

        new_node = assistant.model_completion(node_like_response_model)
        if new_node.result:
            return new_node.result
        else:
            assert False, f"Error: {new_node.error}"

    else:
        new_node = assistant.model_completion(node_like_response_model)
        return new_node


T = TypeVar("T", bound=GraphLike)


def graph_completion(
    assistant: "Assistant",
    graph_like_response_model: type[T],
    stream_update: Optional[Callable[[T], None]] = None,
    can_fail=False,
) -> T:
    if can_fail:
        graph_like_response_model = extend_model_to_include_error_option(graph_like_response_model)  # type: ignore

        new_graph = assistant.model_completion(
            graph_like_response_model,
            stream_update if config.stream else None,
        )
        if new_graph.result:
            return new_graph.result
        else:
            assert False, f"Error: {new_graph.error}"
    else:
        new_graph = assistant.model_completion(
            graph_like_response_model,
            stream_update if config.stream else None,
        )
        return new_graph


def graph_node_completion_model(
    node_completion_model: Optional[type[NodeLike]],
    *to_include: str,
    include_explanation=False,
) -> type[GraphLike]:
    field_names = set(to_include)
    assert field_names.issubset(DataFlowGraph.model_fields.keys() - {"nodes"})

    kwargs = {}
    if node_completion_model:
        kwargs["nodes"] = (
            List[node_completion_model],
            Field(
                description="List of nodes in the data flow graph.",
            ),
        )
    kwargs = kwargs | {
        name: (
            extract_type(field.annotation),
            Field(
                description=field.description,
            ),
        )
        for name, field in DataFlowGraph.model_fields.items()
        if name in to_include and field.annotation is not None
    }

    if include_explanation:
        kwargs["explanation"] = (
            str,
            Field(description="Explanation of the completion."),
        )

    return create_model("GraphCompletion", **kwargs)  # type: ignore


def graph_node_like_model(
    node_like_model: type[NodeLike], *to_include: str
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


if __name__ == "__main__":
    node = Node(
        id="0",
        pill="pill",
        label="label",
        body="body",
        predecessors=["1"],
        phase=Phase.TEST,
        unit_test=SanityCheck(
            requirement="requirement",
            definition="definition",
            call="call",
        ),
        function_name="function_name",
        function_result_var="function_result_var",
    )

    node_like = make_node_like(node, node_like_model("id", "pill", "label"))
    print(node_like)
    print(node_like.model_dump_json(indent=2))

    _node_completion = node_completion_model("body")
    print(_node_completion.model_fields)

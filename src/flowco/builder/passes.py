import ast
import textwrap
from typing import Dict, List, Optional, TypeVar, Any

from pydantic import BaseModel

from nbclient.exceptions import CellExecutionError

from flowco.assistant.assistant import Assistant, AssistantBase
from flowco.builder.build import PassConfig, node_pass
from flowco.builder.type_ops import types_equal

# from flowco.dataflow.check_extended_types import check_type
from flowco.dataflow.extended_type import ExtendedType
from flowco.dataflow.phase import Phase
from flowco.dataflow.dfg import DataFlowGraph, Node, SanityCheck
from flowco.dataflow.graph_completions import (
    graph_completion,
    graph_node_completion_model,
    graph_node_like_model,
    make_graph_node_like,
    make_node_like,
    node_completion,
    node_completion_model,
    node_like_model,
)
from flowco.page.error_messages import error_message
from flowco.util.config import config
from flowco.util.errors import FlowcoError
from flowco.util.output import logger, message, warn, log
from flowco.util.stoppable import Stoppable
from flowco.util.text import strip_ansi


class GraphView(BaseModel):
    graph_fields: List[str]
    node_fields: List[str]


class NodeView(BaseModel):
    node_fields: List[str]


def add_graph_to_assistant(
    assistant: AssistantBase, graph: DataFlowGraph, view: GraphView
) -> None:
    initial_graph_model = graph_node_like_model(
        node_like_model(*view.node_fields), *view.graph_fields
    )

    initial_graph = make_graph_node_like(graph, initial_graph_model)

    assistant.add_json_object(
        "Here is the dataflow graph matching the diagram",
        initial_graph.model_dump(exclude_none=True),
    )


def add_node_to_assistant(assistant: AssistantBase, node: Node, view: NodeView) -> None:
    initial_node_model = node_like_model(*view.node_fields)

    initial_node = make_node_like(node, initial_node_model)

    assistant.add_json_object(
        "Here is the node we are working on",
        initial_node.model_dump(exclude_none=True),
    )


def complete_graph_pass_query(
    prompt_key: str,
    prompt_substitutions: Dict[str, str],
    pass_config: PassConfig,
    graph: DataFlowGraph,
    initial_view: GraphView,
    completion_view: GraphView,
) -> DataFlowGraph:
    with logger("Creating assistant"):
        assistant = Assistant(
            ["system-prompt", prompt_key],
            **prompt_substitutions,
        )
        assistant.add_message("user", pass_config.diagram.to_prompt_messages())

        add_graph_to_assistant(assistant, graph, initial_view)

        if completion_view.node_fields:
            ncm = node_completion_model(*completion_view.node_fields, "id")
            gcm = graph_node_completion_model(ncm, *completion_view.graph_fields)
        else:
            gcm = graph_node_completion_model(None, *completion_view.graph_fields)

    with logger("Running assistant"):
        return graph_completion(
            assistant,
            gcm,  # type: ignore
            stream_update=pass_config.partial_updater,
        )


def complete_node_pass_query(
    prompt_key: str,
    prompt_substitutions: Dict[str, Any],
    pass_config: PassConfig,
    node: Node,
    initial_view: NodeView,
    completion_view: NodeView,
    can_fail=False,
) -> Node:
    with logger("Creating assistant"):
        assistant = Assistant(
            ["system-prompt", prompt_key],
            **prompt_substitutions,
        )

        add_node_to_assistant(assistant, node, initial_view)

        ncm = node_completion_model(*completion_view.node_fields)

    with logger("Running assistant"):
        return node_completion(assistant, ncm, can_fail)  # type: ignore


def complete_node_pass_query_with_graphview(
    prompt_key: str,
    prompt_substitutions: Dict[str, Any],
    pass_config: PassConfig,
    node: Node,
    initial_view: NodeView,
    completion_view: NodeView,
    graph: DataFlowGraph,
    graph_view: GraphView,
) -> Node:
    with logger("Creating assistant"):
        assistant = Assistant(
            ["system-prompt", prompt_key],
            **prompt_substitutions,
        )

        add_graph_to_assistant(assistant, graph, graph_view)
        add_node_to_assistant(assistant, node, initial_view)

        ncm = node_completion_model(*completion_view.node_fields)

    with logger("Running assistant"):
        return node_completion(assistant, ncm)  # type: ignore


T = TypeVar("T", bound=BaseModel)


def complete_node_pass_custom_query(
    prompt_key: str,
    prompt_substitutions: Dict[str, Any],
    pass_config: PassConfig,
    node: Node,
    initial_view: NodeView,
    completion: type[T],
    graph: Optional[DataFlowGraph] = None,
    graph_view: Optional[GraphView] = None,
) -> T:
    with logger("Creating assistant"):
        assistant = Assistant(
            ["system-prompt", prompt_key],
            **prompt_substitutions,
        )

        if graph and graph_view:
            add_graph_to_assistant(assistant, graph, graph_view)

        add_node_to_assistant(assistant, node, initial_view)

    with logger("Running assistant"):
        return node_completion(assistant, completion)  # type: ignore


# @graph_pass(required_phase=Phase.clean, target_phase=Phase.unit_tests)
# def compile(pass_config: PassConfig, graph: DataFlowGraph) -> DataFlowGraph:
#     with logger("Compile"):
#         new_graph = complete_graph_pass_query(
#             "monolithic-compile",
#             {},
#             pass_config,
#             graph,
#             GraphView(graph_fields=["edges"], node_fields=[]),
#             GraphView(
#                 graph_fields=["description"],
#                 node_fields=[
#                     "description",
#                     "requirements",
#                     "function_name",
#                     "function_parameters",
#                     "function_return_type",
#                     "code",
#                     "sanity_checks",
#                 ],
#             ),
#         )
#         new_graph = graph.merge(new_graph).with_phase(
#             node_ids=None, phase=Phase.unit_tests
#         )
#         return new_graph


# ###


@node_pass(required_phase=Phase.code, target_phase=Phase.runnable)
def check_node_syntax(
    pass_config: PassConfig, graph: DataFlowGraph, node: Node
) -> Node:
    new_node = _repair_node_syntax(node, max_retries=config.retries)
    if new_node:
        new_node = new_node.update(phase=Phase.runnable)

        if config.diff:
            diff = node.diff(new_node)
            if diff:
                message(diff)

        return new_node
    else:
        return node


def _check_node_syntax(node: Node) -> None:

    if node.code is None:
        raise FlowcoError(f"It has no code.")
    if node.function_parameters is None:
        raise FlowcoError(f"It has no function parameters.")
    if node.function_return_type is None:
        raise FlowcoError(f"It has no function return type.")

    tree = ast.parse(node.code_str(), "<string>", "exec")

    # Search the tree to find the function definition with the name node.function_name.
    # That should be the only function definition in the code.
    function_def: Optional[ast.FunctionDef] = None
    for tree_node in ast.walk(tree):
        if isinstance(tree_node, ast.FunctionDef):
            if tree_node.name == node.function_name:
                function_def = tree_node
                break
            else:
                raise FlowcoError(
                    f"Function name does not match: should be {node.function_name} but is {tree_node.name}."
                )

    if function_def is None:
        raise FlowcoError(f"Code does not define a function.")

    return_type = node.function_return_type.to_python_type()

    def signature_message():
        return f"\nRewrite the function to have exactly this signature:\n```\ndef {node.function_name}({', '.join([f'{p.name}: {p.type.to_python_type()}' for p in node.function_parameters])}) -> {return_type}:\n```"

    # check the function_def matches node.function_name, node.function_return_type and node.function_parameters
    if function_def.name != node.function_name:
        raise FlowcoError(
            f"Function name does not match: {function_def.name} != {node.function_name}.{signature_message()}"
        )
    if function_def.returns is None:
        raise FlowcoError(
            f"Function does not have a return type. {signature_message()}"
        )

    log(f"Function return type: {ast.unparse(function_def.returns)}")
    log(f"Node return type: {return_type}")
    try:
        if not types_equal(ast.unparse(function_def.returns), return_type):
            raise FlowcoError(
                f"Function return type does not match: {ast.unparse(function_def.returns)} != {return_type}. {signature_message()}"
            )
    except SyntaxError as e:
        raise FlowcoError(
            f"Function return type is not a valid Python type.  The type must be `{return_type}`."
        )
    if len(function_def.args.args) != len(node.function_parameters):
        raise FlowcoError(
            f"Function parameter lengths do not match: {len(function_def.args.args)} != {len(node.function_parameters)}. {signature_message()}"
        )
    for i, arg in enumerate(function_def.args.args):
        arg_type = node.function_parameters[i].type.to_python_type()
        if arg.arg != node.function_parameters[i].name:
            raise FlowcoError(
                f"Function parameter {i} names do not match: {arg.arg} != {node.function_parameters[i].name}. {signature_message()}"
            )
        if arg.annotation is None:
            raise FlowcoError(
                f"Function parameter {i} does not have a type.  Should be: {arg_type}. {signature_message()}"
            )
        if not types_equal(ast.unparse(arg.annotation), arg_type):
            raise FlowcoError(
                f"Function parameter {i} types do not match: {ast.unparse(arg.annotation)} != {arg_type}. {signature_message()}"
            )


def _repair_node_syntax(node: Node, max_retries: int) -> Optional[Node]:
    assistant = Assistant("repair-system")

    retries = 0
    while True:
        try:
            _check_node_syntax(node)
            return node
        except (SyntaxError, FlowcoError) as e:
            retries += 1
            message(
                error_message(
                    f"Syntax error in {node.pill}.",
                    e,
                )
            )
            if retries > max_retries:
                if max_retries > 0:
                    warn(
                        error_message(
                            f"Repair failed",
                            FlowcoError("Too many repair attempts."),
                        )
                    )

                assistant.add_message(
                    "user",
                    "The attempted repairs are not working.  Explain to the user what the problem is, and ask the user a question to help you understand and fix the problem.  Do not refer to the code directly, but only to what property of the code you are trying to check with the failing test.",
                )
                m = assistant.str_completion()
                message(f"A Question: {m}")

                return None

            message(f"Repair attempt {retries} of {config.retries}")

            initial_node = make_node_like(
                node,
                node_like_model(
                    "description",
                    "function_name",
                    "function_parameters",
                    "function_return_type",
                    "requirements",
                    "code",
                ),
            )

            assistant.add_prompt_by_key(
                "repair-syntax",
                node=initial_node.model_dump_json(indent=2),
                error=strip_ansi(str(e)),
            )

            new_node = node_completion(
                assistant, node_completion_model("code", include_explanation=True)
            )

            message(
                "\n".join(
                    textwrap.wrap(
                        f"Explanation of repair: {new_node.explanation}",  # type: ignore
                        subsequent_indent=" " * 4,
                    )
                )
            )
            node = node.merge(new_node)


@node_pass(required_phase=Phase.unit_tests, target_phase=Phase.tests_runnable)
def check_tests_syntax(
    pass_config: PassConfig, graph: DataFlowGraph, node: Node
) -> Node:
    success = True
    new_node = node
    assert node.sanity_checks is not None, "Node has no sanity checks."
    for i in range(len(node.sanity_checks)):
        assert new_node.sanity_checks is not None, "Node has no sanity checks."
        new_test = _repair_test_syntax(
            new_node.sanity_checks[i], max_retries=pass_config.max_retries
        )
        if new_test:
            new_node = new_node.update(
                sanity_checks=new_node.sanity_checks[:i]
                + [new_test]
                + new_node.sanity_checks[i + 1 :]
            )
        else:
            success = False

    if success:
        new_node = new_node.update(phase=Phase.tests_runnable)

    if config.diff:
        diff = node.diff(new_node)
        if diff:
            message(diff)

    return new_node


def _check_test_syntax(test: SanityCheck) -> None:

    for block in test.get_code():
        _ = ast.parse(block, "<string>", "exec")


def _repair_test_syntax(test: SanityCheck, max_retries: int) -> Optional[SanityCheck]:
    assistant = Assistant("repair-system")

    retries = 0
    while True:
        if Stoppable.should_stop():
            return None
        try:
            _check_test_syntax(test)
            return test
        except (SyntaxError, FlowcoError) as e:
            retries += 1
            message(
                error_message(
                    f"Syntax error in test {test.get_call()}.",
                    e,
                )
            )
            if retries > max_retries:
                if max_retries > 0:
                    warn(
                        error_message(
                            f"Repair failed",
                            FlowcoError("Too many repair attempts."),
                        )
                    )

                assistant.add_message(
                    "user",
                    "The attempted repairs are not working.  Explain to the user what the problem is, and ask the user a question to help you understand and fix the problem.  Do not refer to the code directly, but only to what property of the code you are trying to check with the failing test.",
                )
                m = assistant.str_completion()
                message(f"A Question: {m}")

                return None

            message(f"Repair attempt {retries} of {config.retries}")

            assistant.add_prompt_by_key(
                "repair-test-syntax",
                test=test.model_dump_json(indent=2),
                error=strip_ansi(str(e)),
            )

            class TestCompletionModel(BaseModel):
                unit_test: SanityCheck
                explanation: str

            completion = assistant.model_completion(TestCompletionModel)

            message(
                "\n".join(
                    textwrap.wrap(
                        f"Explanation of repair: {completion.explanation}",  # type: ignore
                        subsequent_indent=" " * 4,
                    )
                )
            )
            message(str(completion.unit_test))
            test = completion.unit_test


@node_pass(
    required_phase=Phase.runnable,
    target_phase=Phase.run_checked,
    pred_required_phase=Phase.run_checked,
)
def check_run(pass_config: PassConfig, graph: DataFlowGraph, node: Node) -> Node:

    try:
        new_node = _repair_run(
            pass_config, graph, node, max_retries=pass_config.max_retries
        )
        return new_node.update(phase=Phase.run_checked)
    except CellExecutionError as e:
        warn(str(e))
        node.error(phase=Phase.run_checked, message=strip_ansi(str(e).split("\n")[-2]))
        return node


def _repair_run(
    pass_config: PassConfig, graph: DataFlowGraph, node: Node, max_retries: int
) -> Optional[Node]:
    assistant = Assistant("repair-system")
    retries = 0
    stashed_error = None

    while True:
        try:
            shell = pass_config.get_shell_for_node(node.id)
            result = shell.eval_node(graph, node)
            return node.update(result=result)
        except Exception as e:
            if retries == 0:
                stashed_error = e
            retries += 1

            message(
                error_message(
                    f"Runtime error in {node.pill}.",
                    e,
                )
            )
            if retries > max_retries:
                raise stashed_error

            message(f"Repair attempt {retries} of {config.retries}")

            initial_node = make_node_like(
                node,
                node_like_model(
                    "description",
                    "function_name",
                    "function_parameters",
                    "function_return_type",
                    "requirements",
                    "algorithm",
                    "code",
                ),
            )

            assistant.add_prompt_by_key(
                "repair-node-run",
                error=strip_ansi(str(e)),
            )

            assistant.add_json_object(
                "Here is the offending node", initial_node.model_dump()
            )

            new_node = node_completion(
                assistant, node_completion_model("code", include_explanation=True)
            )

            message(
                "\n".join(
                    textwrap.wrap(
                        f"Explanation of repair: {new_node.explanation}",  # type: ignore
                        subsequent_indent=" " * 4,
                    )
                )
            )
            node = node.merge(new_node)

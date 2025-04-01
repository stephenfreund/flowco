import ast
import textwrap
from typing import Optional, Tuple


from flowco.assistant.flowco_assistant import flowco_assistant
from flowco.builder.build import PassConfig, node_pass
from flowco.builder.type_ops import types_equal

from flowco.dataflow.phase import Phase
from flowco.dataflow.dfg import DataFlowGraph, Node
from flowco.builder.graph_completions import (
    make_node_like,
    node_completion,
    node_completion_model,
    node_like_model,
)
from flowco.page.error_messages import error_message
from flowco.util.config import config
from flowco.util.errors import FlowcoError
from flowco.util.output import message, warn
from flowco.util.text import strip_ansi


@node_pass(required_phase=Phase.code, target_phase=Phase.runnable)
def check_syntax(pass_config: PassConfig, graph: DataFlowGraph, node: Node) -> Node:

    max_retries = pass_config.max_retries

    if node.is_locked:
        max_retries = 0

    new_node, success = _repair_node_syntax(node, max_retries=max_retries)
    if success:
        new_node = new_node.update(phase=Phase.runnable)

    if config().diff:
        diff = node.diff(new_node)
        if diff:
            message(diff)

    return new_node


def _check_node_syntax(node: Node) -> None:

    if node.code is None:
        raise FlowcoError(f"The node has no code.")
    if node.function_parameters is None:
        raise FlowcoError(f"The node has no function parameters.")
    if node.function_return_type is None:
        raise FlowcoError(f"The node has no function return type.")

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
                    f"Function name should be `{node.function_name}` but is `{tree_node.name}`."
                )

    if function_def is None:
        raise FlowcoError(f"The node code does not define a function.")

    return_type = node.function_return_type.to_python_type()

    def signature_message():

        assert (
            node.function_parameters is not None
        ), "Function parameters should not be None"

        return f"\nThe function must have exactly this signature:\n```\ndef {node.function_name}({', '.join([f'{p.name}: {p.type.to_python_type()}' for p in node.function_parameters])}) -> {return_type}:\n```"

    # check the function_def matches node.function_name, node.function_return_type and node.function_parameters
    if function_def.name != node.function_name:
        raise FlowcoError(
            f"The function name should be `{node.function_name}` but is `{function_def.name}`. {signature_message()}"
        )
    if function_def.returns is None:
        raise FlowcoError(
            f"The function does not have a return type. {signature_message()}"
        )

    try:
        if not types_equal(ast.unparse(function_def.returns), return_type):
            raise FlowcoError(
                f"The function return type should be `{return_type}` but is `{ast.unparse(function_def.returns)}. {signature_message()}"
            )
    except SyntaxError as e:
        raise FlowcoError(
            f"The function return type is not a valid Python type.  The type must be `{return_type}`, not `{ast.unparse(function_def.returns)}. {signature_message()}"
        )
    if len(function_def.args.args) != len(node.function_parameters):
        raise FlowcoError(
            f"The function should have {len(node.function_parameters)} but has {len(function_def.args.args)}. {signature_message()}"
        )
    for i, arg in enumerate(function_def.args.args):
        arg_type = node.function_parameters[i].type.to_python_type()
        if arg.arg != node.function_parameters[i].name:
            raise FlowcoError(
                f"Function parameter {i+1} should be named `{node.function_parameters[i].name}`, not `{arg.arg}`. {signature_message()}"
            )
        if arg.annotation is None:
            raise FlowcoError(
                f"Function parameter {i+1} does not have a type.  It should be {arg_type}. {signature_message()}"
            )
        if not types_equal(ast.unparse(arg.annotation), arg_type):
            raise FlowcoError(
                f"Function parameter {i+1} should have type `{arg_type}` but has type `{ast.unparse(arg.annotation)}`. {signature_message()}"
            )


def _repair_node_syntax(node: Node, max_retries: int) -> Tuple[Node, bool]:
    assistant = flowco_assistant()

    original = node.model_copy()

    retries = 0
    stashed_error = None

    while True:
        try:
            _check_node_syntax(node)
            return node, True
        except (SyntaxError, FlowcoError) as e:
            if stashed_error is None:
                stashed_error = e

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

                    m = f"**Code** creation failed.  Automatic repair failed.  Please refine requirements or code, or try again!\n\n{e}"
                else:
                    if node.is_locked:
                        m = f"**Code** creation failed.  Unlock and run again to attempt automatic repair."
                    else:
                        m = f"**Code** creation failed, and automatic repair did not fix the problem.  Please fix the error manually or try running again."

                return (
                    original.error(
                        Phase.runnable,
                        message=f"{m}\n\nDetails: *{e}*",
                    ),
                    False,
                )

            message(f"Repair attempt {retries} of {config().retries}")

            initial_node = make_node_like(
                node,
                node_like_model(
                    [
                        "description",
                        "function_name",
                        "function_parameters",
                        "function_return_type",
                        "requirements",
                        "code",
                    ]
                ),
            )

            prompt = config().get_prompt(
                "repair-syntax",
                node=initial_node.model_dump_json(indent=2),
                signature=node.signature_str(),
                error=strip_ansi(str(e)),
            )
            assistant.add_text("user", prompt)

            new_node = node_completion(
                assistant,
                node_completion_model("code", include_explanation=True),
            )

            message("\n".join(["**Old Code**"] + (node.code or [])))
            message("\n".join(["**New Code**"] + (new_node.code or [])))  # type: ignore
            message(
                "\n".join(
                    textwrap.wrap(
                        f"Explanation of repair: {new_node.explanation}",  # type: ignore
                        subsequent_indent=" " * 4,
                    )
                )
            )
            node = node.merge(new_node)

import ast
import textwrap
from typing import Optional


from flowco.assistant.assistant import Assistant
from flowco.builder.build import PassConfig, node_pass
from flowco.builder.type_ops import types_equal

# from flowco.dataflow.check_extended_types import check_type
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
from flowco.util.output import message, warn, log
from flowco.util.text import strip_ansi


@node_pass(required_phase=Phase.code, target_phase=Phase.runnable)
def check_syntax(pass_config: PassConfig, graph: DataFlowGraph, node: Node) -> Node:
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

    original = node.model_copy()

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

                # assistant.add_message(
                #     "user",
                #     "The attempted repairs are not working.  Explain to the user what the problem is, and ask the user a question to help you understand and fix the problem.  Do not refer to the code directly, but only to what property of the code you are trying to check with the failing test.",
                # )
                # m = assistant.str_completion()
                # message(f"A Question: {m}")

                return original.error(
                    Phase.code,
                    message=f"Too many failed attempts to write code.  Please refine requirements or try again!\n\n{e}",
                )

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


# @node_pass(required_phase=Phase.unit_tests, target_phase=Phase.tests_runnable)
# def check_tests_syntax(
#     pass_config: PassConfig, graph: DataFlowGraph, node: Node
# ) -> Node:
#     success = True
#     new_node = node
#     assert node.sanity_checks is not None, "Node has no sanity checks."
#     for i in range(len(node.sanity_checks)):
#         assert new_node.sanity_checks is not None, "Node has no sanity checks."
#         new_test = _repair_test_syntax(
#             new_node.sanity_checks[i], max_retries=pass_config.max_retries
#         )
#         if new_test:
#             new_node = new_node.update(
#                 sanity_checks=new_node.sanity_checks[:i]
#                 + [new_test]
#                 + new_node.sanity_checks[i + 1 :]
#             )
#         else:
#             success = False

#     if success:
#         new_node = new_node.update(phase=Phase.tests_runnable)

#     if config.diff:
#         diff = node.diff(new_node)
#         if diff:
#             message(diff)

#     return new_node


# def _check_test_syntax(test: SanityCheck) -> None:

#     for block in test.get_code():
#         _ = ast.parse(block, "<string>", "exec")


# def _repair_test_syntax(test: SanityCheck, max_retries: int) -> Optional[SanityCheck]:
#     assistant = Assistant("repair-system")

#     retries = 0
#     while True:
#         if session.get("stopper", Stopper).should_stop():
#             return None
#         try:
#             _check_test_syntax(test)
#             return test
#         except (SyntaxError, FlowcoError) as e:
#             retries += 1
#             message(
#                 error_message(
#                     f"Syntax error in test {test.get_call()}.",
#                     e,
#                 )
#             )
#             if retries > max_retries:
#                 if max_retries > 0:
#                     warn(
#                         error_message(
#                             f"Repair failed",
#                             FlowcoError("Too many repair attempts."),
#                         )
#                     )

#                 assistant.add_message(
#                     "user",
#                     "The attempted repairs are not working.  Explain to the user what the problem is, and ask the user a question to help you understand and fix the problem.  Do not refer to the code directly, but only to what property of the code you are trying to check with the failing test.",
#                 )
#                 m = assistant.str_completion()
#                 message(f"A Question: {m}")

#                 return None

#             message(f"Repair attempt {retries} of {config.retries}")

#             assistant.add_prompt_by_key(
#                 "repair-test-syntax",
#                 test=test.model_dump_json(indent=2),
#                 error=strip_ansi(str(e)),
#             )

#             class TestCompletionModel(BaseModel):
#                 unit_test: SanityCheck
#                 explanation: str

#             completion = assistant.model_completion(TestCompletionModel)

#             message(
#                 "\n".join(
#                     textwrap.wrap(
#                         f"Explanation of repair: {completion.explanation}",  # type: ignore
#                         subsequent_indent=" " * 4,
#                     )
#                 )
#             )
#             message(str(completion.unit_test))
#             test = completion.unit_test

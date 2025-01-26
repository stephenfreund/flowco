import textwrap
from typing import Optional
from flowco.assistant.assistant import Assistant
from flowco.builder.build import PassConfig, node_pass
from flowco.builder.graph_completions import (
    make_node_like,
    node_completion,
    node_completion_model,
    node_like_model,
)
from flowco.dataflow.dfg import DataFlowGraph, Node
from flowco.dataflow.phase import Phase
from flowco.page.error_messages import error_message
from flowco.pythonshell.shells import PythonShells
from flowco.session.session import session
from flowco.util.config import config
from flowco.util.errors import FlowcoError
from flowco.util.output import log, logger, message, warn
from flowco.util.text import strip_ansi


from nbclient.exceptions import CellExecutionError


@node_pass(
    required_phase=Phase.runnable,
    target_phase=Phase.run_checked,
    pred_required_phase=Phase.run_checked,
)
def check_run(pass_config: PassConfig, graph: DataFlowGraph, node: Node) -> Node:

    max_retries = pass_config.max_retries
    if node.is_locked:
        max_retries = 0

    try:
        new_node = _repair_run(pass_config, graph, node, max_retries=max_retries)
        return new_node.update(phase=Phase.run_checked)
    except Exception as e:
        if node.is_locked:
            message = (
                f"**Run** failed.  Unlock and run again to attempt automatic repair."
            )
        else:
            message = f"**Run** failed, and automatic repair did not fix the problem.  Please fix the error manually or try running again."

        log(f"Run didn't work for {node.pill}", e)
        error_line = strip_ansi("\n".join(str(e).split("\n")[-2:]))
        node = node.error(
            phase=Phase.run_checked, message=f"{message}\n\nDetails: *{error_line}*"
        )

        return node


def _repair_run(
    pass_config: PassConfig, graph: DataFlowGraph, node: Node, max_retries: int
) -> Node:
    assistant = Assistant("repair-system")
    retries = 0
    stashed_error = None

    while True:
        try:
            result = session.get("shells", PythonShells).run_node(
                pass_config.tables, graph, node
            )

            with logger("Typechecking result"):
                if result.result is not None:
                    return_value = result.result.to_value()
                    return_type = node.function_return_type
                    if not return_type.matches_value(return_value):
                        raise FlowcoError(
                            f"Return value {return_value} does not match expected type {return_type}."
                        )

            return node.update(result=result)
        except Exception as e:
            if stashed_error is None:
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

            node_fields = [
                "description",
                "function_name",
                "function_parameters",
                "function_return_type",
                "requirements",
                "code",
            ]

            if config.x_algorithm_phase:
                node_fields.append("algorithm")

            initial_node = make_node_like(node, node_like_model(node_fields))

            assistant.add_prompt_by_key(
                "repair-node-run",
                error=strip_ansi(str(e)),
            )

            assistant.add_json_object(
                "Here is the offending node", initial_node.model_dump()
            )

            new_node = node_completion(
                assistant,
                node_completion_model("code", include_explanation=True),
            )

            message("\n".join(["New Code"] + new_node.code))
            message(
                "\n".join(
                    textwrap.wrap(
                        f"Explanation of repair: {new_node.explanation}",  # type: ignore
                        subsequent_indent=" " * 4,
                    )
                )
            )
            node = node.merge(new_node)

import textwrap
from flowco.assistant.flowco_assistant import flowco_assistant
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
from flowco.util.output import log, logger, message
from flowco.util.text import strip_ansi


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
        error_line = strip_ansi(str(e))
        node = node.error(
            phase=Phase.run_checked, message=f"{message}\n\nDetails:\n```\n{error_line}\n```"
        )

        return node


def _repair_run(
    pass_config: PassConfig, graph: DataFlowGraph, node: Node, max_retries: int
) -> Node:
    assistant = flowco_assistant(
        f"{node.pill}-repair-run",
        "repair-system")
    retries = 0
    stashed_error = None

    while True:
        try:
            result = session.get("shells", PythonShells).run_node(
                pass_config.tables, graph, node
            )

            node = node.update(result=result)

            with logger("Typechecking result"):
                if result.result is not None:
                    return_value = result.result.to_value()
                    return_type = node.function_return_type
                    if return_type is not None:
                        return_type.check_value(return_value)
                    else:
                        ValueError("No return type for function")

            return node
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

            message(f"Repair attempt {retries} of {config().retries}")

            node_fields = [
                "description",
                "function_name",
                "function_parameters",
                "function_return_type",
                "requirements",
                "code",
            ]

            if config().x_algorithm_phase:
                node_fields.append("algorithm")

            initial_node = make_node_like(node, node_like_model(node_fields))

            repair_prompt = config().get_prompt(
                "repair-node-run",
                error=strip_ansi(str(e)),
                signature=node.signature_str(),
            )
            assistant.add_text("user", repair_prompt)
            assistant.add_text("user", "Here is the offending node")
            assistant.add_json("user", initial_node.model_dump())

            if node.result is not None:
                for p in node.result.to_content_parts():
                    assistant.add_content_parts("user", p)

            new_node = assistant.model_completion(
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

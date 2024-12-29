import textwrap
from typing import Optional
from flowco.assistant.assistant import Assistant
from flowco.builder.build import PassConfig, node_pass
from flowco.builder.graph_completions import make_node_like, node_completion, node_completion_model, node_like_model
from flowco.dataflow.dfg import DataFlowGraph, Node
from flowco.dataflow.phase import Phase
from flowco.page.error_messages import error_message
from flowco.pythonshell.shells import PythonShells
from flowco.session.session import session
from flowco.util.config import config
from flowco.util.errors import FlowcoError
from flowco.util.output import logger, message, warn
from flowco.util.text import strip_ansi


from nbclient.exceptions import CellExecutionError


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
            result = session.get("shells", PythonShells).run_node(
                pass_config.tables, graph, node
            )

            with logger("Typechecking result"):
                if result.result is not Node:
                    return_value = result.result.to_value()
                    return_type = node.function_return_type
                    if not return_type.matches_value(return_value):
                        raise FlowcoError(
                            f"Return value {return_value} does not match expected type {return_type}."
                        )

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


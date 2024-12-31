import json
import textwrap
from typing import List, Optional, OrderedDict

from nbclient.exceptions import CellExecutionError
from pydantic import Field, BaseModel

from flowco.assistant.assistant import Assistant
from flowco.assistant.openai import OpenAIAssistant
from flowco.builder.build import PassConfig, node_pass
from flowco.builder.graph_completions import (
    make_node_like,
    messages_for_node,
    node_completion,
    node_completion_model,
    node_like_model,
)
from flowco.dataflow.checks import CheckOutcomes, QuantitiveCheck, Check
from flowco.dataflow.dfg import DataFlowGraph, Node
from flowco.dataflow.phase import Phase
from flowco.page.error_messages import error_message
from flowco.pythonshell.shells import PythonShells
from flowco.session.session import session
from flowco.util.config import config
from flowco.util.output import logger, log, message, warn
from flowco.util.text import strip_ansi


@node_pass(required_phase=Phase.run_checked, target_phase=Phase.assertions_code)
def compile_assertions(
    pass_config: PassConfig, graph: DataFlowGraph, node: Node
) -> Node:
    with logger("Compile Assertions step"):

        # TODO add diff part

        # Check cache
        if node.cache.matches_in_and_out(Phase.assertions_code, node):
            log("Using cache for code.")
            return node.update(phase=Phase.assertions_code)

        if not node.assertions:
            log("No assertions to compile.")
            new_node = node.update(assertion_checks={}, phase=Phase.assertions_code)
            return new_node.update(
                cache=new_node.cache.update(Phase.assertions_code, new_node)
            )

        assistant = assertions_assistant(node)

        class AssertionCompletion(BaseModel):
            requirement: str
            check: Check = Field(description="The check to perform.")
            error: str | None = Field(
                description="An error message if the assertion is inconsistent with the requirements."
            )

        class AssertionsCompletion(BaseModel):
            assertions: List[AssertionCompletion]

        completion = assistant.completion(AssertionsCompletion)

        checks = {x.requirement: x.check for x in completion.assertions}
        new_node = node.update(assertion_checks=checks, phase=Phase.assertions_code)

        failures = [x for x in completion.assertions if x.error]
        if failures:
            for check in failures:
                new_node = new_node.error(
                    phase=Phase.assertions_code,
                    message=f"{check.requirement}: {check.error}",
                )
            return new_node

        new_node = new_node.update(
            cache=new_node.cache.update(Phase.assertions_code, new_node)
        )

        return new_node


def assertions_assistant(node: Node):
    parameter_types = {param.name: param.type for param in node.function_parameters}
    parameter_type_str = "\n".join(
        [
            f"* {name}:\n{textwrap.indent(t.to_markdown(), '    ')}\n"
            for name, t in parameter_types.items()
        ]
    )

    if node.function_return_type.is_None_type():
        prompt = "assertions-inspect"
    else:
        prompt = "assertions-code"

    assistant = OpenAIAssistant(
        config.model,
        interactive=False,
        system_prompt_key=["system-prompt", prompt],
        input_vars=parameter_type_str,
        preconditions=json.dumps(node.preconditions, indent=2),
        output_var=node.function_result_var,
        postconditions="\n".join([f"* {r}" for r in node.requirements]),
        requirements=json.dumps(node.assertions, indent=2),
    )

    return assistant


@node_pass(
    required_phase=Phase.assertions_code,
    target_phase=Phase.assertions_checked,
    pred_required_phase=Phase.assertions_code,
)
def check_assertions(pass_config: PassConfig, graph: DataFlowGraph, node: Node) -> Node:
    try:
        return _repair_assertions(
            pass_config, graph, node, max_retries=pass_config.max_retries
        )

    except CellExecutionError as e:
        warn(str(e))
        node.error(
            phase=Phase.assertions_checked, message=strip_ansi(str(e).split("\n")[-2])
        )
        return node


def _repair_assertions(
    pass_config: PassConfig, graph: DataFlowGraph, node: Node, max_retries: int
) -> Optional[Node]:
    assistant = Assistant("repair-system")
    retries = 0
    original = None

    max_retries = 0

    while True:
        node = node.update(assertion_outcomes=CheckOutcomes())
        node = session.get("shells", PythonShells).run_assertions(
            pass_config.tables, graph, node
        )
        if original is None:
            original = node

        num_failed = len([x for x in node.assertion_outcomes.outcomes.values() if x])
        if num_failed == 0:
            return node.update(phase=Phase.assertions_checked)

        # message(f"Failed {num_failed} assertions")

        retries += 1

        if retries > max_retries:
            for assertion, outcome in node.assertion_outcomes.outcomes.items():
                if outcome is not None:
                    message = f"Assertion failure: {assertion}\n\n{outcome}"
                    original = original.error(
                        phase=Phase.assertions_checked, message=message
                    )
            return original

        message(f"Repair attempt {retries} of {config.retries}")

        assistant.add_message(
            "user",
            messages_for_node(
                node=node,
                node_fields=[
                    "pill",
                    "preconditions",
                    "requirements",
                    "function_name",
                    "function_return_type",
                    "function_parameters",
                    "function_result_var",
                    "code",
                ],
            ),
        )

        assistant.add_prompt_by_key(
            "repair-node-assertions",
            errors=[
                outcome
                for outcome in node.assertion_outcomes.outcomes.values()
                if outcome is not None
            ],
            context=json.dumps(
                {x: v.to_text() for x, v in node.assertion_outcomes.context.items()},
                indent=2,
            ),
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

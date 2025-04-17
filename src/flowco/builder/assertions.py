import json
import textwrap
from typing import List

from nbclient.exceptions import CellExecutionError
from pydantic import Field, BaseModel

from flowco.assistant.flowco_assistant import flowco_assistant
from flowco.builder.build import PassConfig, node_pass
from flowco.builder.graph_completions import (
    json_for_node_view,
    node_completion,
    node_completion_model,
)
from flowco.dataflow.checks import (
    CheckOutcomes,
    QualitativeCheck,
    QuantitiveCheck,
)
from flowco.dataflow.dfg import DataFlowGraph, Node
from flowco.dataflow.phase import Phase
from flowco.pythonshell.shells import PythonShells
from flowco.session.session import session
from flowco.util.config import config
from flowco.util.output import logger, log, warn, message
from flowco.util.text import black_format, strip_ansi


def suggest_assertions(graph: DataFlowGraph, node: Node) -> List[str]:
    assistant = assertions_assistant(node, suggest=True)

    class SuggestedAssertions(BaseModel):
        suggestions: List[str]

    suggestions = assistant.model_completion(SuggestedAssertions)
    assert suggestions is not None, "No completion from assistant"
    return suggestions.suggestions


def none_return_type_completion(node):
    class AssertionCompletion(BaseModel):
        requirement: str
        qualitative_analysis: str = Field(
            description="A description of the requirement for this test."
        )
        error: str | None = Field(
            description="An error message if the requirements do not guarantee the assertion."
        )

    class AssertionsCompletion(BaseModel):
        assertions: List[AssertionCompletion]

    assistant = assertions_assistant(node)
    completion = assistant.model_completion(AssertionsCompletion)
    assert completion, "No completion from assistant"

    checks = {
        a: QualitativeCheck(
            type="qualitative", requirement=x.qualitative_analysis, warning=x.error
        )
        for a, x in zip(node.assertions, completion.assertions)
    }
    return checks


def not_none_return_type_completion(node):
    class AssertionCompletion(BaseModel):
        requirement: str
        code: List[str] = Field(
            description="Code to run to verify the this requirement is met.  The code is stored as a list of source lines."
        )
        error: str | None = Field(
            description="An error message if the requirements do not guarantee the assertion."
        )

    class AssertionsCompletion(BaseModel):
        assertions: List[AssertionCompletion]

    assistant = assertions_assistant(node)
    completion = assistant.model_completion(AssertionsCompletion)
    assert completion, "No completion from assistant"

    checks = {
        a: QuantitiveCheck(
            type="quantitative", code=black_format(x.code), warning=x.error
        )
        for a, x in zip(node.assertions, completion.assertions)
    }
    return checks


@node_pass(
    required_phase=Phase.requirements,
    target_phase=Phase.assertions_code,
)
def suggest_assertions_pass(
    pass_config: PassConfig, graph: DataFlowGraph, node: Node
) -> Node:
    with logger("Suggest Unit Test step"):

        if not node.assertions:
            suggestions = suggest_assertions(graph, node)
            if suggestions:
                phase = node.phase
                node = node.update(assertions=suggestions)
                node = compile_assertions(pass_config, graph, node)
                node = node.update(phase=phase)

        return node


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

        assert node.function_return_type is not None, "No function return type"
        if node.function_return_type.is_None_type():
            checks = none_return_type_completion(node)
        else:
            checks = not_none_return_type_completion(node)

        new_node = node.update(assertion_checks=checks, phase=Phase.assertions_code)

        new_node = new_node.update(
            messages=[x for x in new_node.messages if x.phase != Phase.assertions_code]
        )

        if any([x.warning for x in checks.values()]):
            for assertion, check in checks.items():
                if check.warning:
                    new_node = new_node.warn(
                        phase=Phase.assertions_code,
                        message=f"**Check '{assertion}'**: {check.warning}",
                    )
            return new_node

        new_node = new_node.update(
            cache=new_node.cache.update(Phase.assertions_code, new_node)
        )

        return new_node


def assertions_assistant(node: Node, suggest=False):

    assert node.function_parameters is not None, "No function parameters"
    assert node.function_return_type is not None, "No function return type"
    assert node.requirements is not None, "No requirements"
    assert node.preconditions is not None, "No preconditions"

    parameter_types = {param.name: param.type for param in node.function_parameters}
    parameter_type_str = "\n".join(
        [
            f"* {name}:\n{textwrap.indent(t.to_markdown(), '    ')}\n"
            for name, t in parameter_types.items()
        ]
    )

    if node.function_return_type.is_None_type():
        if suggest:
            prompt = "suggest-assertions-inspect"
        else:
            prompt = "assertions-inspect"
    else:
        if suggest:
            prompt = "suggest-assertions-code"
        else:
            prompt = "assertions-code"

    # TODO: Think about whether we can reuse any cached code here.  Might be easier
    # just to regenerate it every time...

    substitutions = {
        "input_vars": parameter_type_str,
        "preconditions": node.preconditions.model_dump_json(indent=2),
        "output_var": node.function_result_var,
        "postconditions": "\n".join([f"* {r}" for r in node.requirements]),
        "return_type": node.function_return_type.model_dump_json(indent=2),
        "requirements": json.dumps(node.assertions or [], indent=2),
    }

    assistant = flowco_assistant(f"assertions-{node.id}", prompt, **substitutions)

    return assistant


@node_pass(
    required_phase=Phase.assertions_code,
    target_phase=Phase.assertions_checked,
    pred_required_phase=Phase.assertions_code,
)
def check_assertions(pass_config: PassConfig, graph: DataFlowGraph, node: Node) -> Node:
    try:
        return _repair_assertions(pass_config, graph, node, max_retries=0)

    except CellExecutionError as e:
        warn(e)
        node.error(
            phase=Phase.assertions_checked, message=strip_ansi(str(e).split("\n")[-2])
        )
        return node


@node_pass(
    required_phase=Phase.assertions_code,
    target_phase=Phase.assertions_checked,
    pred_required_phase=Phase.assertions_code,
)
def repair_assertions(
    pass_config: PassConfig, graph: DataFlowGraph, node: Node
) -> Node:

    # NOTE: This will modify the node, even if locked, by design!

    node = node.update(
        messages=[x for x in node.messages if x.phase != Phase.assertions_checked]
    )

    try:
        return _repair_assertions(
            pass_config, graph, node, max_retries=pass_config.max_retries_for_node(node)
        )

    except CellExecutionError as e:
        warn(e)
        node.error(
            phase=Phase.assertions_checked, message=strip_ansi(str(e).split("\n")[-2])
        )
        return node


def _repair_assertions(
    pass_config: PassConfig, graph: DataFlowGraph, node: Node, max_retries: int
) -> Node:
    assistant = flowco_assistant(
        f"repair-assertions-{node.id}", prompt_key="repair-assertions")
    retries = 0
    original = None

    while True:
        node = node.update(assertion_outcomes=CheckOutcomes())
        shells = session.get("shells", PythonShells)

        if retries > 0:
            # rerun if changes were made
            result = shells.run_node(pass_config.tables, graph, node)
            node = node.update(result=result)

        node = shells.run_assertions(pass_config.tables, graph, node)

        assert node.assertion_outcomes is not None, "No assertion outcomes"

        if original is None:
            original = node

        num_failed = len([x for x in node.assertion_outcomes.outcomes.values() if x])
        if num_failed == 0:
            if original.requirements != node.requirements:
                return node.update(phase=Phase.clean)
            else:
                return node.update(phase=Phase.assertions_checked)

        if retries >= max_retries:
            if retries > 0:
                original = original.error(
                    phase=Phase.assertions_checked,
                    message=f"**Checks** failed and automatic repair didn not fix the problem.  Verify the requirements ensure the assertions are met and try again.",
                )

            for assertion, outcome in node.assertion_outcomes.outcomes.items():
                if outcome is not None:
                    original = original.error(
                        phase=Phase.assertions_checked,
                        message=f"**Check Failed: {assertion}**\n\n*{outcome}*",
                    )
            return original
        else:
            retries += 1

        message(f"Repair attempt {retries} of {config().retries}")

        assistant.add_text("user", f"Here is the current state of node {node.pill}:")
        assistant.add_json(
            "user",
            json_for_node_view(
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

        repair_prompt = config().get_prompt(
            "repair-node-assertions",
            errors=[
                x for x in node.assertion_outcomes.outcomes.values() if x is not None
            ],
            context=json.dumps(
                {x: v.to_text() for x, v in node.assertion_outcomes.context.items()},  # type: ignore
                indent=2,
            ),
        )

        assistant.add_text("user", repair_prompt)

        new_node = node_completion(
            assistant,
            node_completion_model("requirements", "code", include_explanation=True),
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

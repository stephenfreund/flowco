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
    QualitativeCheckWithCode,
    QuantitiveCheck,
)
from flowco.dataflow.dfg import DataFlowGraph, Node
from flowco.dataflow.phase import Phase
from flowco.dataflow.tests import UnitTest
from flowco.pythonshell.shells import PythonShells
from flowco.session.session import session
from flowco.util.config import config
from flowco.util.output import logger, log, warn, message
from flowco.util.text import black_format, strip_ansi


def suggest_unit_tests(graph: DataFlowGraph, node: Node) -> List[UnitTest]:
    assistant = unit_tests_assistant(node, suggest=True)

    class SuggestedUnitTests(BaseModel):
        suggestions: List[UnitTest]

    suggestions = assistant.model_completion(SuggestedUnitTests)
    assert suggestions is not None, "No completion from assistant"
    return suggestions.suggestions


@node_pass(
    required_phase=Phase.requirements,
    target_phase=Phase.unit_tests_code,
)
def suggest_unit_tests_pass(
    pass_config: PassConfig, graph: DataFlowGraph, node: Node
) -> Node:
    with logger("Suggest Unit Test step"):

        if not node.unit_tests:
            suggestions = suggest_unit_tests(graph, node)
            if suggestions:
                phase = node.phase
                node = node.update(unit_tests=suggestions)
                node = compile_unit_tests(pass_config, graph, node)
                node = node.update(phase=phase)

        return node


def none_return_type_completion(node):
    class UnitTestCompletion(BaseModel):
        code: List[str] = Field(
            description="Code to run to generate the output to examine.  The code is stored as a list of source lines."
        )
        qualitative_analysis: str = Field(
            description="A description of the requirement for this test."
        )
        error: str | None = Field(
            description="An error message if the requirements do not guarantee the unit_test."
        )

    class UnitTestsCompletion(BaseModel):
        unit_tests: List[UnitTestCompletion]

    assistant = unit_tests_assistant(node)
    completion = assistant.model_completion(UnitTestsCompletion)
    assert completion, "No completion from assistant"

    checks = {
        str(a): QualitativeCheckWithCode(
            type="qualitative-code",
            code=x.code,
            requirement=x.qualitative_analysis,
            warning=x.error,
        )
        for a, x in zip(node.unit_tests, completion.unit_tests)
    }
    return checks


def not_none_return_type_completion(node):
    class UnitTestCompletion(BaseModel):
        code: List[str] = Field(
            description="Code to run to verify the this requirement is met.  The code is stored as a list of source lines."
        )
        error: str | None = Field(
            description="An error message if the requirements do not guarantee the unit_test."
        )

    class UnitTestsCompletion(BaseModel):
        unit_tests: List[UnitTestCompletion]

    assistant = unit_tests_assistant(node)
    completion = assistant.model_completion(UnitTestsCompletion)
    assert completion, "No completion from assistant"

    checks = {
        str(a): QuantitiveCheck(
            type="quantitative", code=black_format(x.code), warning=x.error
        )
        for a, x in zip(node.unit_tests, completion.unit_tests)
    }
    return checks


@node_pass(required_phase=Phase.assertions_checked, target_phase=Phase.unit_tests_code)
def compile_unit_tests(
    pass_config: PassConfig, graph: DataFlowGraph, node: Node
) -> Node:
    with logger("Compile UnitTests step"):

        # TODO add diff part

        # Check cache
        if node.cache.matches_in_and_out(Phase.unit_tests_code, node):
            log("Using cache for code.")
            return node.update(phase=Phase.unit_tests_code)

        if not node.unit_tests:
            log("No unit_tests to compile.")
            new_node = node.update(unit_test_checks={}, phase=Phase.unit_tests_code)
            return new_node.update(
                cache=new_node.cache.update(Phase.unit_tests_code, new_node)
            )

        assert node.function_return_type is not None, "No function return type"
        if node.function_return_type.is_None_type():
            checks = none_return_type_completion(node)
        else:
            checks = not_none_return_type_completion(node)

        new_node = node.update(unit_test_checks=checks, phase=Phase.unit_tests_code)

        new_node = new_node.update(
            messages=[x for x in new_node.messages if x.phase != Phase.unit_tests_code]
        )

        if any([x.warning for x in checks.values()]):
            for unit_test, check in checks.items():
                if check.warning:
                    new_node = new_node.warn(
                        phase=Phase.unit_tests_code,
                        message=f"**Test '{unit_test}'**: {check.warning}",
                    )
            return new_node

        new_node = new_node.update(
            cache=new_node.cache.update(Phase.unit_tests_code, new_node)
        )

        return new_node


def unit_tests_assistant(node: Node, suggest=False):

    assert node.function_parameters is not None, "No function parameters"
    assert node.function_return_type is not None, "No function return type"
    assert node.requirements is not None, "No requirements"

    parameter_types = {param.name: param.type for param in node.function_parameters}
    parameter_type_str = "\n".join(
        [
            f"* {name}:\n{textwrap.indent(t.to_markdown(), '    ')}\n"
            for name, t in parameter_types.items()
        ]
    )

    call_to_node = f"{node.function_name}({', '.join([parameter.name for parameter in node.function_parameters])})"
    input_lines = "\n".join(
        [f"{parameter.name} = ..." for parameter in node.function_parameters]
    )

    if node.function_return_type.is_None_type():
        if suggest:
            prompt = "suggest-unit_tests-inspect"
        else:
            prompt = "unit_tests-inspect"
    else:
        if suggest:
            prompt = "suggest-unit_tests-code"
        else:
            prompt = "unit_tests-code"

    unit_tests = [x.model_dump() for x in node.unit_tests or []]

    substitutions = {
        "function_name": node.function_name,
        "input_vars": parameter_type_str,
        "call_to_node": call_to_node,
        "input_lines": input_lines,
        "preconditions": json.dumps(node.preconditions, indent=2),
        "output_var": node.function_result_var,
        "return_type": node.function_return_type.model_dump_json(indent=2),
        "postconditions": "\n".join([f"* {r}" for r in node.requirements]),
        "unit_tests": json.dumps(unit_tests or [], indent=2),
    }

    assistant = flowco_assistant(prompt, **substitutions)

    return assistant


@node_pass(
    required_phase=Phase.unit_tests_code,
    target_phase=Phase.unit_tests_checked,
    pred_required_phase=Phase.unit_tests_code,
)
def check_unit_tests(pass_config: PassConfig, graph: DataFlowGraph, node: Node) -> Node:
    try:
        return _repair_unit_tests(pass_config, graph, node, max_retries=0)

    except CellExecutionError as e:
        warn(e)
        node.error(
            phase=Phase.unit_tests_checked, message=strip_ansi(str(e).split("\n")[-2])
        )
        return node


@node_pass(
    required_phase=Phase.unit_tests_code,
    target_phase=Phase.unit_tests_checked,
    pred_required_phase=Phase.unit_tests_code,
)
def repair_unit_tests(
    pass_config: PassConfig, graph: DataFlowGraph, node: Node
) -> Node:

    # NOTE: This will modify the node, even if locked, by design!
    node = node.update(
        messages=[x for x in node.messages if x.phase != Phase.unit_tests_checked]
    )

    try:
        return _repair_unit_tests(
            pass_config, graph, node, max_retries=pass_config.max_retries_for_node(node)
        )

    except CellExecutionError as e:
        warn(e)
        node.error(
            phase=Phase.unit_tests_checked, message=strip_ansi(str(e).split("\n")[-2])
        )
        return node


def _repair_unit_tests(
    pass_config: PassConfig, graph: DataFlowGraph, node: Node, max_retries: int
) -> Node:
    assistant = flowco_assistant("repair-system")
    retries = 0
    original = None

    while True:
        node = node.update(unit_test_outcomes=CheckOutcomes())
        shells = session.get("shells", PythonShells)

        if retries > 0:
            # rerun if changes were made
            result = shells.run_node(pass_config.tables, graph, node)
            node = node.update(result=result)

        node = shells.run_unit_tests(pass_config.tables, graph, node)

        assert node.unit_test_outcomes is not None, "No unit_test outcomes"

        if original is None:
            original = node

        num_failed = len([x for x in node.unit_test_outcomes.outcomes.values() if x])
        if num_failed == 0:
            if original.requirements != node.requirements:
                return node.update(phase=Phase.clean)
            elif original.code != node.code:
                return node.update(phase=Phase.code)
            else:
                return node.update(phase=Phase.unit_tests_checked)

        if retries >= max_retries:
            if retries > 0:
                original = original.error(
                    phase=Phase.unit_tests_checked,
                    message=f"**Unit Tests** failed and automatic repair didn not fix the problem.  Verify the requirements ensure the unit_tests are met and try again.",
                )

            for unit_test, outcome in node.unit_test_outcomes.outcomes.items():
                if outcome is not None:
                    original = original.error(
                        phase=Phase.unit_tests_checked,
                        message=f"**Unit Test Failed: {unit_test}**\n\n*{outcome}*",
                    )
            return original
        else:
            retries += 1

        message(f"Repair attempt {retries} of {config.retries}")

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

        repair_prompt = config.get_prompt(
            "repair-node-unit_tests",
            errors=[
                x for x in node.unit_test_outcomes.outcomes.values() if x is not None
            ],
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

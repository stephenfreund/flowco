from collections import deque

import json
import textwrap
from typing import Dict, List, Tuple
from pydantic import Field, BaseModel
from flowco.assistant.openai import OpenAIAssistant
from flowco.builder.graph_completions import (
    cache_prompt,
    node_completion_model,
    update_node_with_completion,
)
from flowco.builder.graph_completions import messages_for_graph
from flowco.builder.graph_completions import messages_for_node
from flowco.dataflow.phase import Phase
from flowco.util.semantic_diff import semantic_diff_strings
from flowco.util.output import logger, log
from flowco.builder.build import PassConfig, node_pass
from flowco.util.config import config

from flowco.dataflow.dfg import DataFlowGraph, Parameter, Node


@node_pass(
    required_phase=Phase.clean,
    target_phase=Phase.requirements,
    pred_required_phase=Phase.requirements,
)
def requirements(pass_config: PassConfig, graph: DataFlowGraph, node: Node) -> Node:
    with logger("Requirements step"):

        old_preconditions = node.preconditions
        # Compute inputs
        node = node.update(
            function_parameters=create_parameters(graph, node),
            preconditions=create_preconditions(pass_config, graph, node),
        )

        # Check cache
        if node.cache.matches_in_and_out(Phase.requirements, node):
            log("Using cache for requirements.")
            return node.update(phase=Phase.requirements)

        diff_instructions = cache_prompt(Phase.requirements, node, "requirements")

        if not config.x_shortcurcuit_requirements:
            node = check_precondition_consistency(node)
        else:
            node = check_precondition_shortcurcuit(node, old_preconditions)
            if node.phase == Phase.requirements:
                return node.update(cache=node.cache.update(Phase.requirements, node))

        # TODO - this is wonky.  need to regenerate if you add the first successor..

        # Create assistant
        assistant = requirements_assistant(pass_config, graph, node, diff_instructions)

        with logger("requirements assistant"):
            completion = get_requirements_completion(assistant)

        new_node = update_node_with_completion(node, completion)
        new_node = new_node.update(phase=Phase.requirements)

        if config.x_no_descriptions:
            new_node = new_node.update(description="", function_computed_value="")

        new_node = new_node.update(
            cache=new_node.cache.update(Phase.requirements, new_node)
        )
        return new_node


def get_requirements_completion(assistant) -> BaseModel | None:
    completion_model = requirements_completion_model()
    completion = assistant.completion(completion_model)
    return completion


def requirements_assistant(
    pass_config: PassConfig,
    graph: DataFlowGraph,
    node: Node,
    diff_instructions: str,
    interactive=False,
):
    if graph.successors(node.id):
        prompt = "postconditions"
    else:
        prompt = "postconditions-for-sink"

    if config.x_no_descriptions:
        prompt += "-no-descriptions"

    assistant = OpenAIAssistant(
        config.model,
        interactive=interactive,
        system_prompt_key=["system-prompt", prompt],
        preconditions=json.dumps(node.preconditions),
        function_return_var=node.function_result_var,
        diff=diff_instructions,
        label=node.label,
    )

    assistant.add_message(
        "user",
        messages_for_graph(
            graph=graph,
            graph_fields=["edges", "description"],
            node_fields=["requirements"],
        ),
    )

    image = graph.to_image_prompt_messages()
    assistant.add_message("user", image)

    assistant.add_message(
        "user",
        messages_for_node(
            node=node,
            node_fields=[
                "id",
                "pill",
                "label",
                "predecessors",
                "preconditions",
                "requirements",
            ],
        ),
    )

    return assistant


def requirements_completion_model():
    if not config.x_no_descriptions:
        fields = [
            "requirements",
            "description",
            "function_return_type",
            "function_computed_value",
        ]
    else:
        fields = ["requirements", "function_return_type"]

    return node_completion_model(*fields)


def interactive_requirements_assistant(
    pass_config: PassConfig,
    graph: DataFlowGraph,
    node: Node,
    new_requirements: List[str],
) -> Tuple[OpenAIAssistant, BaseModel]:

    # clear all messages to start fresh.
    node = node.update(messages=[])

    # Compute inputs
    assert node.function_parameters == create_parameters(
        graph, node
    ), f"Parameters not update to date: {node.function_parameters} != {create_parameters(graph, node)}"
    preconditions = create_preconditions(pass_config, graph, node)
    assert (
        node.preconditions == preconditions
    ), f"Preconditions not update to date: {node.preconditions} != {preconditions}"

    with logger("Semantic diff"):
        diff_strings = semantic_diff_strings(
            "requirements", json.dumps(node.requirements), json.dumps(new_requirements)
        )

    newline = "\n"
    diff_instructions = textwrap.dedent(
        f"""\
        The current requirements are:
        ```
        {newline.join([ f"* {x}" for x in node.requirements or []])}
        ```
        I want to change them to be:
        ```
        {newline.join([ f"* {x}" for x in new_requirements])}
        ```
        The key differences between the old and new are:
        ```
        {diff_strings}
        ```
        Improve that new list for clarity and precision.  If there are any ambiguiuties
        or contradictions, ask for clarification.
        """
    )

    # Verify preconditions are consistent
    with logger("Preconditions checks"):
        node = check_precondition_consistency(node)

    assistant = requirements_assistant(
        pass_config, graph.with_node(node), node, diff_instructions, interactive=True
    )

    completion_model = requirements_completion_model()

    return assistant, completion_model


def check_precondition_consistency(node: Node) -> Node:
    with logger("Preconditions checks"):

        class Warning(BaseModel):
            inconsistencies: List[str] = Field(
                description="Inconsistencies in the requirements.",
            )
            ok: bool = Field(
                description="Whether the requirements are consistent.",
            )

        assistant = OpenAIAssistant(
            config.model,
            interactive=False,
            system_prompt_key="inconsistent-preconditions",
            preconditions=json.dumps(node.preconditions),
        )

        completion = assistant.completion(Warning)

        if not completion.ok:
            for warning in completion.inconsistencies:
                node = node.warn(
                    phase=Phase.requirements,
                    message=warning,
                )

    return node


def check_precondition_shortcurcuit(
    node: Node, old_preconditions: Dict[str, List[str]]
) -> Node:
    with logger("Preconditions shortcurcuit checks"):

        class Warning(BaseModel):
            inconsistencies: List[str] = Field(
                description="Inconsistencies in the preconditions.",
            )
            no_inconsistencies: bool = Field(
                description="Whether the preconditions are consistent.",
            )
            semantic_implications: List[str] = Field(
                description="Semantic implications of change from the old to the new preconditions.",
            )
            no_semantic_implications: bool = Field(
                description="Whether the preconditions have semantic implications.",
            )
            impact_on_postconditions: List[str] = Field(
                description="Impact on postconditions of the change from the old to the new preconditions.",
            )
            no_impact_on_postconditions: bool = Field(
                description="Whether the preconditions have impact on postconditions.",
            )

        assistant = OpenAIAssistant(
            config.model,
            interactive=False,
            system_prompt_key="precondition-checks",
            preconditions=json.dumps(node.preconditions),
            old_preconditions=json.dumps(old_preconditions),
            postconditions=json.dumps(node.requirements),
        )

        completion = assistant.completion(Warning)

        with logger("Precondition inconsistencies"):
            if not completion.no_inconsistencies:
                for warning in completion.inconsistencies:
                    node = node.warn(
                        phase=Phase.requirements,
                        message=warning,
                    )

        with logger("Semantic implications"):
            if not completion.no_semantic_implications:
                for t in completion.semantic_implications:
                    log(t)

        with logger("Impact on postconditions"):
            if not completion.no_impact_on_postconditions:
                for t in completion.impact_on_postconditions:
                    log(t)
            else:
                log("No impact on postconditions.")
                node = node.update(phase=Phase.requirements)

    return node


def summarize_changes(node: Node, phase: Phase, diff_map: Dict[str, List[str]]):
    summary = diff_map["summary"]
    if summary:
        node.info(
            phase=phase,
            message=f"{summary}",
        )


def create_preconditions(
    pass_config: PassConfig, graph: DataFlowGraph, node: Node
) -> Dict[str, List[str]]:
    preconditions: Dict[str, List[str]] = {}
    if node.predecessors:
        for pred in node.predecessors:
            req = graph[pred].requirements
            assert req is not None, "Predecessor must have requirements."
            preconditions[graph[pred].function_result_var] = req

    return preconditions | pass_config.tables.as_preconditions()


def create_parameters(graph, node: Node) -> List[Parameter]:
    params = [
        Parameter(
            name=graph[pred].function_result_var,
            type=graph[pred].function_return_type,
        )
        for pred in node.predecessors
    ]
    return params


@node_pass(
    required_phase=Phase.requirements,
    target_phase=Phase.algorithm,
    pred_required_phase=Phase.requirements,
)
def algorithm(pass_config: PassConfig, graph: DataFlowGraph, node: Node) -> Node:
    assert node.function_parameters is not None, "Parameters must be defined."
    assert node.preconditions is not None, "Preconditions must be defined."
    assert node.requirements is not None, "Requirements must be defined."
    assert node.description is not None, "Description must be defined."

    if graph.successors(node.id):
        assert node.function_return_type is not None, "Return type must be defined."
        assert (
            node.function_computed_value is not None
        ), "Computed value must be defined."

    # Check cache
    if node.cache.matches_in_and_out(Phase.algorithm, node):
        log("Using cache for algorithm.")
        return node.update(phase=Phase.algorithm)

    with logger("Diff instructions"):
        diff_instructions = cache_prompt(Phase.algorithm, node, "algorithm")

    with logger("Algorithm Completion"):
        assistant = algorithm_assistant(node, diff_instructions)
        completion_model = algorithm_completion_model()
        completion = assistant.completion(completion_model)

    with logger("Update node with completion"):
        new_node = update_node_with_completion(node, completion)
        new_node = new_node.update(phase=Phase.algorithm)
        new_node = new_node.update(
            cache=new_node.cache.update(Phase.algorithm, new_node)
        )

    return new_node


def algorithm_assistant(node, diff_instructions, interactive=False):
    assistant = OpenAIAssistant(
        config.model,
        interactive=interactive,
        system_prompt_key=["system-prompt", "algorithm"],
        preconditions=json.dumps(node.preconditions),
        postconditions=json.dumps(node.requirements),
        diff=diff_instructions,
    )

    assistant.add_message(
        "user",
        messages_for_node(
            node=node,
            node_fields=[
                "id",
                "pill",
                "preconditions",
                "requirements",
                "description",
                "function_return_type",
                "function_computed_value",
                "function_parameters",
                "function_result_var",
                "algorithm",
            ],
        ),
    )

    return assistant


def algorithm_completion_model():
    completion_model = node_completion_model("algorithm")
    return completion_model


def interactive_algorithm_assistant(
    pass_config: PassConfig, graph: DataFlowGraph, node: Node, new_algorithm: List[str]
) -> Tuple[OpenAIAssistant, BaseModel]:
    with logger("Algorithm step"):
        assert node.function_parameters is not None, "Parameters must be defined."
        assert node.preconditions is not None, "Preconditions must be defined."
        assert node.requirements is not None, "Requirements must be defined."
        assert node.description is not None, "Description must be defined."

        if graph.successors(node.id):
            assert node.function_return_type is not None, "Return type must be defined."
            assert (
                node.function_computed_value is not None
            ), "Computed value must be defined."

        newline = "\n"
        diff_instructions = textwrap.dedent(
            f"""\
            The current algorithm is:
            ```
            {newline.join(node.algorithm)}
            ```
            I want to change it to something like:
            ```
            {newline.join(new_algorithm)}
            ```
            The key differences between the old and new algorithm are:
            ```
            {semantic_diff_strings("algorithm", json.dumps(node.algorithm), json.dumps(new_algorithm))}
            ```
            The algorithm must also conform to the requirements:
            ```
            {newline.join([ f"* {x}" for x in node.requirements])}
            ```
            Also improve that new algorithm for clarity and precision.  If there are any ambiguiuties
            or contradictions, ask for clarification.
            """
        )

        assistant = algorithm_assistant(node, diff_instructions, interactive=True)

        completion_model = algorithm_completion_model()

        return assistant, completion_model


@node_pass(required_phase=Phase.algorithm, target_phase=Phase.code)
def compile(pass_config: PassConfig, graph: DataFlowGraph, node: Node) -> Node:
    with logger("Compile step"):

        assert node.function_parameters is not None, "Parameters must be defined."
        assert node.preconditions is not None, "Preconditions must be defined."
        assert node.requirements is not None, "Requirements must be defined."
        assert node.description is not None, "Description must be defined."
        assert node.algorithm is not None, "Algorithm must be defined."

        if graph.successors(node.id):
            assert node.function_return_type is not None, "Return type must be defined."
            assert (
                node.function_computed_value is not None
            ), "Computed value must be defined."

        # Check cache
        if node.cache.matches_in_and_out(Phase.code, node):
            log("Using cache for code.")
            return node.update(phase=Phase.code)

        diff_instructions = cache_prompt(Phase.code, node, "code")

        assistant = code_assistant(node, diff_instructions)

        completion_model = code_completion_model()
        completion = assistant.completion(completion_model)

        new_node = update_node_with_completion(node, completion)
        new_node = new_node.update(phase=Phase.code)

        new_node = new_node.update(cache=new_node.cache.update(Phase.code, new_node))

        return new_node


def code_assistant(node: Node, diff_instructions, interactive=False):
    parameter_types = {param.name: param.type for param in node.function_parameters}
    parameter_type_str = "\n".join(
        [
            f"* {name}:\n{textwrap.indent(t.to_markdown(include_description=False), '    ')}\n"
            for name, t in parameter_types.items()
        ],
    )

    assistant = OpenAIAssistant(
        config.model,
        interactive=interactive,
        system_prompt_key=["system-prompt", "compile"],
        signature=node.signature_str(),
        parameter_types=parameter_type_str,
        diff=diff_instructions,
    )

    assistant.add_message(
        "user",
        messages_for_node(
            node=node,
            node_fields=[
                "id",
                "pill",
                "preconditions",
                "requirements",
                "description",
                "function_return_type",
                "function_computed_value",
                "function_parameters",
                "function_result_var",
                "algorithm",
            ],
        ),
    )

    return assistant


def code_completion_model():
    completion_model = node_completion_model("code")
    return completion_model


def interactive_code_assistant(
    pass_config: PassConfig, graph: DataFlowGraph, node: Node, new_code: List[str]
) -> Tuple[OpenAIAssistant, BaseModel]:
    with logger("Compile step"):

        assert node.function_parameters is not None, "Parameters must be defined."
        assert node.preconditions is not None, "Preconditions must be defined."
        assert node.requirements is not None, "Requirements must be defined."
        assert node.description is not None, "Description must be defined."
        assert node.algorithm is not None, "Algorithm must be defined."

        if graph.successors(node.id):
            assert node.function_return_type is not None, "Return type must be defined."
            assert (
                node.function_computed_value is not None
            ), "Computed value must be defined."

        newline = "\n"
        diff_instructions = textwrap.dedent(
            f"""\
            The current code is:
            ```
            {newline.join(node.code)}
            ```
            I want to change it to be:
            ```
            {newline.join(new_code)}
            ```
            The key differences between the old and new are:
            ```
            {semantic_diff_strings("code", json.dumps(node.code), json.dumps(new_code))}
            ```
            The code must also conform to the requirements:
            ```
            {newline.join([ f"* {x}" for x in node.requirements])}
            ```
            The code must also conform to the algorithm:
            ```
            {newline.join([ f"{x}" for x in node.algorithm])}
            ```

            Also improve that new code for clarity and precision.  If there are any ambiguiuties
            or contradictions, ask for clarification.
            """
        )

        assistant = code_assistant(node, diff_instructions, interactive=True)

        completion_model = node_completion_model("code")

        return assistant, completion_model


#######################


@node_pass(
    required_phase=Phase.clean,
    target_phase=Phase.code,
    pred_required_phase=Phase.requirements,
)
def full_pass(pass_config: PassConfig, graph: DataFlowGraph, node: Node) -> Node:
    with logger("Requirements-To-Code step"):

        # Compute inputs
        node = node.update(
            function_parameters=create_parameters(graph, node),
            preconditions=create_preconditions(pass_config, graph, node),
        )

        diff = {
            Phase.requirements: "",
            Phase.algorithm: "",
            Phase.code: "",
        }

        # Check cache
        if node.cache.matches_in_and_out(Phase.requirements, node):
            log("Using cache for requirements.")
            node = node.update(phase=Phase.requirements)
            if node.cache.matches_in_and_out(Phase.algorithm, node):
                log("Using cache for algorithm.")
                node = node.update(phase=Phase.algorithm)
                if node.cache.matches_in_and_out(Phase.code, node):
                    log("Using cache for code.")
                    return node.update(phase=Phase.code)
                else:
                    diff[Phase.code] = cache_prompt(Phase.code, node, "code")
            else:
                diff[Phase.algorithm] = cache_prompt(Phase.algorithm, node, "algorithm")
        else:
            diff[Phase.requirements] = cache_prompt(
                Phase.requirements, node, "requirements"
            )

            # Verify preconditions are consistent
            with logger("Preconditions checks"):
                node = check_precondition_consistency(node)

        assistant = full_assistant(pass_config, graph, node, diff)

        with logger("requirements assistant"):
            completion = full_completion(assistant, node.phase)

        new_node = update_node_with_completion(node, completion)
        new_node = new_node.update(phase=Phase.code)

        if config.x_no_descriptions:
            new_node = new_node.update(description="", function_computed_value="")

        new_node = new_node.update(
            cache=new_node.cache.update(Phase.requirements, new_node)
            .update(Phase.algorithm, new_node)
            .update(Phase.code, new_node)
        )
        return new_node


def full_assistant(
    pass_config: PassConfig,
    graph: DataFlowGraph,
    node: Node,
    diff: Dict[Phase, str],
    interactive=False,
):
    phase = node.phase
    params = ", ".join([str(p) for p in node.function_parameters])

    parameter_types = {param.name: param.type for param in node.function_parameters}
    parameter_type_str = "\n".join(
        [
            f"* {name}:\n{textwrap.indent(t.to_markdown(), '    ')}\n"
            for name, t in parameter_types.items()
        ]
    )

    prompts = ["system-prompt"]

    if phase < Phase.requirements:
        if graph.successors(node.id):
            prompts += ["full-compile"]
        else:
            prompts += ["full-compile-sink"]
        signature = f"{node.function_name}({params})"
    else:
        signature = node.signature_str()

    if phase < Phase.algorithm:
        prompts += ["full-compile-algorithm"]

    if phase < Phase.code:
        prompts += ["full-compile-code"]

    assistant = OpenAIAssistant(
        config.model,
        interactive=interactive,
        system_prompt_key=prompts,
        label=node.label,
        preconditions=json.dumps(node.preconditions),
        parameter_types=parameter_type_str,
        function_return_var=node.function_result_var,
        signature=signature,
        diff_req=diff[Phase.requirements],
        diff_alg=diff[Phase.algorithm],
        diff_code=diff[Phase.code],
    )

    if phase < Phase.requirements:
        messages = messages_for_graph(
            graph=graph,
            graph_fields=["edges", "description"],
            node_fields=["requirements"],
        )
        assistant.add_message("user", messages)

        image = graph.to_image_prompt_messages()
        assistant.add_message("user", image)

    assistant.add_message(
        "user",
        messages_for_node(
            node=node,
            node_fields=[
                "id",
                "pill",
                "label",
                "predecessors",
                "preconditions",
                "requirements",
                "function_parameters",
                "function_return_type",
                "function_computed_value",
                "description",
                "algorithm",
                "code",
            ],
        ),
    )

    return assistant


def full_completion(assistant: OpenAIAssistant, phase: Phase):
    fields = []
    if phase < Phase.requirements:
        if not config.x_no_descriptions:
            fields += [
                "requirements",
                "description",
                "function_return_type",
                "function_computed_value",
            ]
        else:
            fields += ["requirements", "function_return_type"]
    if phase < Phase.algorithm:
        fields += ["algorithm"]
    if phase < Phase.code:
        fields += ["code"]

    completion_model = node_completion_model(*fields)
    return assistant.completion(completion_model)


############


# @node_pass(required_phase=Phase.run_checked, target_phase=Phase.sanity_checks)
# def sanity_checks(pass_config: PassConfig, graph: DataFlowGraph, node: Node) -> Node:
#     with logger("Write sanity checks step"):

#         assert node.requirements is not None, "Requirements must be defined."
#         assert node.function_name is not None, "Function name must be defined."
#         assert (
#             node.function_parameters is not None
#         ), "Function parameters must be defined."
#         assert (
#             node.function_return_type is not None
#         ), "Function return type must be defined."
#         assert node.code is not None, "Code must be defined."

#         if node.build_cache.sanity_checks is not None:
#             if node.build_cache.sanity_checks.matches(
#                 requirements=node.requirements,
#                 function_name=node.function_name,
#                 function_parameters=node.function_parameters,
#                 function_return_type=node.function_return_type,
#                 code=node.code,
#             ):
#                 if (
#                     node.sanity_checks
#                     == node.build_cache.sanity_checks.out_sanity_checks
#                 ):
#                     log("Using cache for sanity checks.")
#                     return node.update(**node.build_cache.sanity_checks.out())
#                 elif node.sanity_checks is not None:
#                     node.warn(
#                         phase=Phase.sanity_checks,
#                         message="Requirements, algorithm, or code may have changed -- regenerate tests.",
#                     )

#         completion = complete_node_pass_query(
#             "sanity-checks",
#             {},
#             pass_config,
#             node,
#             NodeView(
#                 node_fields=[
#                     "requirements",
#                     "function_name",
#                     "function_parameters",
#                     "function_return_type",
#                     "code",
#                 ]
#             ),
#             NodeView(
#                 node_fields=[
#                     "sanity_checks",
#                 ]
#             ),
#         )

#         sanity_checks = completion.sanity_checks
#         assert sanity_checks is not None, "Sanity checks must be defined."

#         build_cache = node.build_cache.update(
#             sanity_checks=SanityChecksCache(
#                 in_requirements=node.requirements,
#                 in_function_name=node.function_name,
#                 in_function_parameters=node.function_parameters,
#                 in_function_return_type=node.function_return_type,
#                 in_code=node.code,
#                 out_sanity_checks=sanity_checks,
#             )
#         )

#         assert build_cache.sanity_checks, "Sanity checks must be defined."

#         new_node = node.update(
#             **build_cache.sanity_checks.out(), build_cache=build_cache
#         )
#         return new_node


# @node_pass(required_phase=Phase.sanity_checks, target_phase=Phase.unit_tests)
# def unit_tests(pass_config: PassConfig, graph: DataFlowGraph, node: Node) -> Node:
#     with logger("Write unit tests step"):

#         assert node.requirements is not None, "Requirements must be defined."
#         assert node.function_name is not None, "Function name must be defined."
#         assert (
#             node.function_parameters is not None
#         ), "Function parameters must be defined."
#         assert (
#             node.function_return_type is not None
#         ), "Function return type must be defined."
#         assert node.code is not None, "Code must be defined."

#         if node.build_cache.unit_tests is not None:
#             if node.build_cache.unit_tests.matches(
#                 requirements=node.requirements,
#                 function_name=node.function_name,
#                 function_parameters=node.function_parameters,
#                 function_return_type=node.function_return_type,
#             ):
#                 if node.unit_tests == node.build_cache.unit_tests.out_unit_tests:
#                     log("Using cache for unit checks.")
#                     return node.update(**node.build_cache.unit_tests.out())
#                 elif node.unit_tests is not None:
#                     node.warn(
#                         phase=Phase.unit_tests,
#                         message="Requirements, algorithm, or code may have changed -- regenerate tests.",
#                     )

#         unit_tests_data_model = unit_tests_base_model(node)

#         completion = complete_node_pass_custom_query(
#             "unit-tests",
#             {},
#             pass_config,
#             node,
#             NodeView(
#                 node_fields=[
#                     "requirements",
#                     "function_name",
#                     "function_parameters",
#                     "function_return_type",
#                     "function_result_var",
#                 ]
#             ),
#             unit_tests_data_model,
#         )

#         tests = completion.tests
#         unit_tests = [
#             UnitTest(
#                 inputs={
#                     param.name: getattr(test, param.name)
#                     for param in node.function_parameters
#                 },
#                 requirement=test.requirement,
#                 expected=test.expected if hasattr(test, "expected") else None,
#                 function_name=node.function_name,
#                 result_name=node.function_result_var,
#             )
#             for test in tests
#         ]

#         assert unit_tests is not None, "Unit tests must be defined."

#         build_cache = node.build_cache.update(
#             unit_tests=UnitTestsCache(
#                 in_requirements=node.requirements,
#                 in_function_name=node.function_name,
#                 in_function_parameters=node.function_parameters,
#                 in_function_return_type=node.function_return_type,
#                 out_unit_tests=unit_tests,
#             )
#         )

#         assert build_cache.unit_tests, "Sanity checks must be defined."

#         new_node = node.update(**build_cache.unit_tests.out(), build_cache=build_cache)
#         return new_node


# def unit_tests_base_model(node):
#     unit_test_data_model = unit_test_base_model(node)

#     unit_tests_data_model = create_model(
#         "UnitTestsWithData",
#         tests=(
#             List[unit_test_data_model],
#             Field(description="List of unit tests with input and output data."),
#         ),
#     )

#     return unit_tests_data_model


# def unit_test_base_model(node):
#     fields: Dict[str, Any] = {
#         param.name: (
#             str,
#             Field(
#                 description=f"Value for the parameter {param}. This should be a valid Python expression of type {param.type.as_python_type()}"
#             ),
#         )
#         for param in node.function_parameters
#     }
#     fields["requirement"] = (
#         str,
#         Field(description="Requirement for the test."),
#     )
#     if node.function_return_type.as_actual_type() != None:
#         fields["expected"] = (
#             str,
#             Field(description="A Python boolean expression for the test."),
#         )

#     unit_test_data_model = create_model("UnitTestWithData", **fields)
#     return unit_test_data_model

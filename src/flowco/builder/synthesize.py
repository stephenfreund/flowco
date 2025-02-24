import json
import textwrap
from typing import Dict, List
from pydantic import Field, BaseModel
from flowco.assistant.flowco_assistant import flowco_assistant
from flowco.builder.graph_completions import (
    cache_prompt,
    json_for_graph_view,
    json_for_node_view,
    node_completion_model,
    update_node_with_completion,
)
from flowco.dataflow.extended_type import ExtendedType, NoneType
from flowco.dataflow.phase import Phase
from flowco.util.output import logger, log
from flowco.builder.build import PassConfig, node_pass
from flowco.util.config import config

from flowco.dataflow.dfg import DataFlowGraph, NodeKind, Parameter, Node
from llm.assistant import Assistant


@node_pass(
    required_phase=Phase.clean,
    target_phase=Phase.requirements,
    pred_required_phase=Phase.requirements,
)
def requirements(pass_config: PassConfig, graph: DataFlowGraph, node: Node) -> Node:
    with logger("Requirements step"):

        # Compute inputs
        node = node.update(
            function_parameters=create_parameters(graph, node),
            preconditions=create_preconditions(pass_config, graph, node),
        )

        # Check cache
        if node.cache.matches_in_and_out(Phase.requirements, node):
            log("Using cache for requirements.")
            return node.update(phase=Phase.requirements)

        if node.is_locked and node.requirements is not None:
            return requirements_when_locked(node)

        # TODO: Merge consistency and diffs completions...
        node = check_precondition_consistency(node)

        diff_instructions = cache_prompt(Phase.requirements, node, "requirements")
        assistant = requirements_assistant(pass_config, graph, node, diff_instructions)

        with logger("requirements compute assistant"):
            completion = get_requirements_completion(node, assistant)

        new_node = update_node_with_completion(node, completion)
        new_node = new_node.update(
            phase=Phase.requirements, description="", function_computed_value=""
        )

        if node.kind == NodeKind.plot:
            new_node = new_node.update(
                function_return_type=ExtendedType(
                    the_type=NoneType(type="None", description="No output value."),
                    description="No output value.",
                ),
            )

        new_node = new_node.update(
            cache=new_node.cache.update(Phase.requirements, new_node)
        )
        return new_node


def get_requirements_completion(node: Node, assistant: Assistant) -> BaseModel:
    completion_model = requirements_completion_model(node)
    completion = assistant.model_completion(completion_model)
    assert completion is not None, "Completion must be defined."
    return completion


def requirements_assistant(
    pass_config: PassConfig,
    graph: DataFlowGraph,
    node: Node,
    diff_instructions: str,
):
    if node.kind != NodeKind.plot:
        prompt = "postconditions"
    else:
        prompt = "postconditions-for-plot"

    assistant = flowco_assistant("system-prompt")
    prompt_text = config.get_prompt(
        prompt_key=prompt,
        preconditions=json.dumps(node.preconditions),
        function_return_var=node.function_result_var,
        diff=diff_instructions,
        label=node.label,
    )
    assistant.add_text("system", prompt_text)

    assistant.add_text("user", "Here is the current state of the graph:")
    assistant.add_json(
        "user",
        json_for_graph_view(
            graph=graph,
            graph_fields=["edges", "description"],
            node_fields=["requirements"],
        ),
    )

    image_url = graph.to_image_url()
    if image_url:
        assistant.add_text("system", f"Here is the dataflow graph.")
        assistant.add_image("user", image_url)

    assistant.add_text("user", f"Here is the current state of node {node.pill}:")
    assistant.add_json(
        "user",
        json_for_node_view(
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


def requirements_completion_model(node: Node):
    if node.kind != NodeKind.plot:
        fields = ["requirements", "function_return_type"]
    else:
        fields = ["requirements"]

    return node_completion_model(*fields)


def check_precondition_consistency(node: Node) -> Node:
    with logger("Preconditions checks"):

        class Warning(BaseModel):
            inconsistencies: List[str] = Field(
                description="Inconsistencies in the requirements.",
            )
            ok: bool = Field(
                description="Whether the requirements are consistent.",
            )

        assistant = flowco_assistant(
            prompt_key="inconsistent-preconditions",
            preconditions=json.dumps(node.preconditions),
        )

        completion = assistant.model_completion(Warning)

        if not completion.ok:
            for warning in completion.inconsistencies:
                node = node.warn(
                    phase=Phase.requirements,
                    message=warning,
                )

    return node


def requirements_when_locked(node: Node) -> Node:
    with logger("Preconditions shortcurcuit checks"):

        old_inputs = node.cache.get_in(Phase.requirements, node)
        current_inputs = node.model_dump(include=set(old_inputs.keys()))

        if old_inputs != current_inputs:

            class Warning(BaseModel):
                changes_to_preconditions: List[str] = Field(
                    description="Inconsistencies in the preconditions.",
                )
                impacts_postconditions: bool = Field(
                    description="Whether the preconditions and parameters have semantic implications for the postconditions.",
                )
                impacts_on_postconditions: List[str] = Field(
                    description="Semantic implications of change from the old to the new parameters and preconditions.",
                )
                label_impacts_postconditions: bool = Field(
                    description="Whether the label change has semantic implications for the postconditions.",
                )
                impact_of_label_change: str = Field(
                    description="Semantic implications of the label change for the postconditions.",
                )

            assistant = flowco_assistant(
                prompt_key="locked-requirements-checks",
                parameters=json.dumps(current_inputs["function_parameters"]),
                old_parameters=json.dumps(old_inputs["function_parameters"]),
                preconditions=json.dumps(current_inputs["preconditions"]),
                old_preconditions=json.dumps(old_inputs["preconditions"]),
                postconditions=json.dumps(node.requirements),
                new_label=node.label,
                old_label=old_inputs["label"],
            )

            completion = assistant.model_completion(Warning)

            assert completion is not None, "Completion must be defined."
            log(completion)

            if completion.impacts_postconditions:
                impacts = [f"* {x}" for x in completion.impacts_on_postconditions]
                message = (
                    f"Changes to the inputs may impact **requirements**:\n"
                    + "\n".join(impacts)
                )
                node = node.warn(phase=Phase.requirements, message=message)
            if completion.label_impacts_postconditions:
                message = (
                    f"Label has changed from '{old_inputs['label']}' to '{node.label}'.  This may impact the requirements:\n"
                    + completion.impact_of_label_change
                )
                node = node.warn(phase=Phase.requirements, message=message)

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

    return preconditions


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

    if config.x_algorithm_phase:
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
            completion = assistant.model_completion(completion_model)
            new_node = update_node_with_completion(node, completion)

    else:
        new_node = node

    new_node = new_node.update(phase=Phase.algorithm)
    new_node = new_node.update(cache=new_node.cache.update(Phase.algorithm, new_node))

    return new_node


def algorithm_assistant(node, diff_instructions):

    assert config.x_algorithm_phase, "Algorithm phase must be enabled."

    assistant = flowco_assistant(
        prompt_key="algorithm",
        preconditions=json.dumps(node.preconditions),
        postconditions=json.dumps(node.requirements),
        diff=diff_instructions,
    )

    assistant.add_text("user", "Here is the current state of the node:")
    assistant.add_json(
        "user",
        json_for_node_view(
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

    assert config.x_algorithm_phase, "Algorithm phase must be enabled."

    completion_model = node_completion_model("algorithm")
    return completion_model


def generate_docstring_for_node(node: Node) -> str:
    """
    Generate a Google style docstring for a Node instance representing a computation stage.

    This docstring summarizes the function's purpose, its parameters (with types and any
    associated preconditions), and the return type (with additional return requirements).

    Args:
        Each parameter is documented with its name, type, description, and its preconditions
        (if any).

    Returns:
        {return_type}: Description of the return value.
        Return Requirements:
            - Additional requirements for the return value.
    """
    assert node.function_parameters is not None, "Parameters must be defined."
    assert node.preconditions is not None, "Preconditions must be defined."
    assert node.function_result_var is not None, "Result var must be defined."
    assert node.function_return_type is not None, "Return type must be defined."
    assert node.requirements is not None, "Requirements must be defined."

    lines = []

    # Summary: function name and a brief label description.
    lines.append(f"{node.label}")
    lines.append("")
    lines.append("This function has the following behavior:")
    for requirement in node.requirements:
        modified_requirement = requirement.replace(
            f"`{node.function_result_var}`", "the result"
        )
        lines.append(f"        - {modified_requirement}")

    # Args section: list each parameter with its type, description, and associated preconditions.
    lines.append("Args:")
    for param in node.function_parameters:
        # Document the parameter.
        lines.append(
            f"    {param.name} ({param.type.to_python_type()}): {param.type.description}"
        )
        # Append preconditions if they exist for this parameter.
        if param.name in node.preconditions:
            lines.append("        Preconditions:")
            for condition in node.preconditions[param.name]:
                condition = condition.replace("output", param.name)
                lines.append(f"            - {condition}")
    lines.append("")

    # Returns section: include return type and any additional return requirements.
    lines.append("Returns:")
    lines.append(
        f"    {node.function_return_type.to_python_type()}: {node.function_return_type.description}"
    )
    lines.append("")

    return "\n".join(lines)


@node_pass(required_phase=Phase.algorithm, target_phase=Phase.code)
def compile(pass_config: PassConfig, graph: DataFlowGraph, node: Node) -> Node:
    with logger("Compile step"):

        assert node.function_parameters is not None, "Parameters must be defined."
        assert node.preconditions is not None, "Preconditions must be defined."
        assert node.requirements is not None, "Requirements must be defined."
        assert node.description is not None, "Description must be defined."

        if config.x_algorithm_phase:
            assert node.algorithm is not None, "Algorithm must be defined."

        # if graph.successors(node.id):
        assert node.function_return_type is not None, "Return type must be defined."
        assert (
            node.function_computed_value is not None
        ), "Computed value must be defined."

        # Check cache
        if node.cache.matches_in_and_out(Phase.code, node):
            log("Using cache for code.")
            return node.update(phase=Phase.code)

        # shortcurcuit if locked and code has not changed
        # otherwise, recompute
        if node.is_locked and node.code is not None:
            return compile_when_locked(node)
        else:
            diff_instructions = cache_prompt(Phase.code, node, "code")
            assistant = code_assistant(node, diff_instructions)

        completion_model = code_completion_model()
        completion = assistant.model_completion(completion_model)

        assert completion is not None, "Completion must be defined."

        new_node = update_node_with_completion(node, completion)
        new_node = new_node.update(phase=Phase.code)

        new_node = new_node.update(cache=new_node.cache.update(Phase.code, new_node))

        return new_node


def code_assistant(node: Node, diff_instructions):

    assert node.function_parameters is not None, "Parameters must be defined."

    parameter_types = {param.name: param.type for param in node.function_parameters}

    if node.kind != NodeKind.plot:
        prompt = "compile-pydoc"
    else:
        prompt = "compile-pydoc-for-plot"

    assistant = flowco_assistant(
        prompt_key=prompt,
        signature=node.signature_str(),
        diff=diff_instructions,
        pydoc=textwrap.indent(
            generate_docstring_for_node(node), prefix="    ", predicate=lambda x: True
        ),
    )

    node_fields = [
        # "id",
        # "pill",
        # "preconditions",
        # "requirements",
        # "description",
        "function_return_type",
        # "function_computed_value",
        "function_parameters",
        # "function_result_var",
    ]

    # if config.x_algorithm_phase:
    #     node_fields.append("algorithm")

    assistant.add_text(
        "user",
        f"Here is the details of the parameter and return types, and what they represent:",
    )
    assistant.add_json(
        "user",
        json_for_node_view(node=node, node_fields=node_fields),
    )

    # assistant.add_text("user", f"Here is the current state of node {node.pill}:")
    # assistant.add_json(
    #     "user",
    #     json_for_node_view(node=node, node_fields=node_fields),
    # )

    return assistant


def code_completion_model():
    completion_model = node_completion_model("code")
    return completion_model


def compile_when_locked(node: Node) -> Node:
    with logger("Compile shortcurcuit checks"):

        old_inputs = node.cache.get_in(Phase.code, node)
        current_inputs = node.model_dump(include=set(old_inputs.keys()))

        if old_inputs != current_inputs:

            # "preconditions",
            # "requirements",
            # "function_parameters",
            # "function_return_type",

            class Warning(BaseModel):
                changes_to_preconditions_parameters: List[str] = Field(
                    description="Changes to the parameters and preconditions",
                )
                precondition_and_parameter_changes_impact_code: bool = Field(
                    description="Whether the changes to preconditions and parameters have semantic implications for the code.",
                )
                precondition_and_parameter_impacts_on_code: List[str] = Field(
                    description="Semantic implications of change from the old to the new preconditions and parameters.",
                )
                changes_to_requirements_and_return_type: List[str] = Field(
                    description="Changes to the requirements and return type",
                )
                requirements_and_return_type_changes_impact_code: bool = Field(
                    description="Whether the changes to requirements and return type have semantic implications for the code.",
                )
                requirements_and_return_type_impacts_on_code: List[str] = Field(
                    description="Semantic implications of change from the old to the new requirements and return type.",
                )

            assistant = flowco_assistant(
                prompt_key="locked-code-checks",
                old=old_inputs
                | {"code": node.cache.get_out(Phase.code, node).get("code", None)},
                new=current_inputs | {"code": node.code},
            )

            completion = assistant.model_completion(Warning)

            assert completion is not None, "Completion must be defined."
            log(completion)

            change_types = []
            full_impacts = []
            if completion.precondition_and_parameter_changes_impact_code:
                change_types += ["inputs"]
                full_impacts += [
                    f"* {x}"
                    for x in completion.precondition_and_parameter_impacts_on_code
                ]
            if completion.requirements_and_return_type_changes_impact_code:
                change_types += ["requirements"]
                full_impacts += [
                    f"* {x}"
                    for x in completion.requirements_and_return_type_impacts_on_code
                ]
            if change_types:
                message = (
                    f"Changes to the {' and '.join(change_types)} may impact the **code**:\n"
                    + "\n".join(full_impacts)
                )
                node = node.warn(phase=Phase.code, message=message)

        node = node.update(phase=Phase.code)

        return node

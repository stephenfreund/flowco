# import json
# import sys
# import traceback
# from typing import Any, Dict, List, Optional
# from pydantic import Field, create_model, BaseModel
# from flowco.assistant.assistant import Assistant
# from flowco.dataflow.build_cache import (
#     AlgorithmCache,
#     AlgorithmIn,
#     CompileCache,
#     CompileIn,
#     RequirementsCache,
#     RequirementsIn,
#     SanityChecksCache,
#     UnitTestsCache,
# )
# from flowco.dataflow.phase import Phase
# from flowco.util.config import AbstractionLevel, RegenerationPolicy, config
# from flowco.util.output import logger
# from flowco.builder.build import PassConfig, node_pass
# from flowco.util.output import log
# from flowco.dataflow.extended_type import ExtendedType

# from flowco.builder.passes import (
#     GraphView,
#     NodeView,
#     complete_node_pass_custom_query,
#     complete_node_pass_query,
#     complete_node_pass_query_with_graphview,
# )
# from flowco.dataflow.dfg import DataFlowGraph, UnitTest, Parameter, Node, UnitTest


# @node_pass(
#     required_phase=Phase.clean,
#     target_phase=Phase.requirements,
#     pred_required_phase=Phase.requirements,
# )
# def requirements(pass_config: PassConfig, graph: DataFlowGraph, node: Node) -> Node:
#     try:
#         with logger("Requirements step"):

#             # Compute inputs
#             params = create_parameters(graph, node)
#             preconditions = create_preconditions(graph, node)

#             requirements = node.requirements

#             diff = ""
#             if node.build_cache.requirements is not None:
#                 if node.build_cache.requirements.matches(params, preconditions):
#                     if requirements == node.build_cache.requirements.out_requirements:
#                         # cache hit, no user edits to requirements
#                         log("Using cache for requirements.")
#                         return node.update(**node.build_cache.requirements.out())
#                     elif requirements is not None:
#                         # cache hit, but user has changed requirements, warn about that.
#                         node.warn(
#                             phase=Phase.requirements,
#                             message="Predecessors have changed -- regenerating requirements.",
#                         )
#                 else:
#                     diff_map = node.build_cache.requirements.diff(params, preconditions)
#                     diff = f"Here are the changes to reflect in the new requirements:\n```\n{diff_map}\n```\n"
#                     summarize_changes(node, Phase.requirements, diff_map)

#             # Any checks
#             with logger("Preconditions checks"):

#                 class Warning(BaseModel):
#                     inconsistencies: List[str] = Field(
#                         default_factory=list,
#                         description="Inconsistencies in the requirements.",
#                     )
#                     ok: bool = Field(
#                         default=False,
#                         description="Whether the requirements are consistent.",
#                     )

#                 assistant = Assistant(
#                     "inconsistent-preconditions",
#                     preconditions=json.dumps(preconditions),
#                 )

#                 completion = assistant.model_completion(Warning)

#                 if not completion.ok:
#                     for warning in completion.inconsistencies:
#                         node.warn(
#                             phase=Phase.requirements,
#                             message=warning,
#                         )

#             if graph.successors(node.id):
#                 prompt = "postconditions"
#             else:
#                 prompt = "postconditions-for-sink"

#             completion = complete_node_pass_query_with_graphview(
#                 prompt_key=prompt,
#                 prompt_substitutions={
#                     "preconditions": preconditions,
#                     "function_return_var": node.function_result_var,
#                     "diff": diff,
#                 },
#                 pass_config=pass_config,
#                 node=node,
#                 initial_view=NodeView(
#                     node_fields=["id", "pill", "label", "predecessors", "requirements"]
#                 ),
#                 completion_view=NodeView(node_fields=["description", "requirements"]),
#                 graph=graph,
#                 graph_view=GraphView(
#                     graph_fields=["edges", "description"],
#                     node_fields=["requirements"],
#                 ),
#             )

#             generated_requirements = completion.requirements
#             assert generated_requirements is not None, "Requirements must be defined."

#             # if (
#             #     config.regenerate_policy == RegenerationPolicy.merge
#             #     and node.requirements is not None
#             #     and node.requirements != generated_requirements
#             # ):
#             #     response = todo.merge(
#             #         TodoMergeList(
#             #             kind="items",
#             #             node_id=node.id,
#             #             phase=Phase.requirements,
#             #             message="Updating requirements",
#             #             old=node.requirements,
#             #             new=generated_requirements,
#             #         )
#             #     )
#             #     assert response is not None, "Response must be defined."
#             #     assert isinstance(response.result, list), "Response must be a list."
#             #     generated_requirements = response.result
#             #     assert (
#             #         generated_requirements is not None
#             #     ), "Requirements must be defined."

#             new_node = node.update(
#                 requirements=generated_requirements, description=completion.description
#             )

#             assert new_node.requirements is not None, "Requirements must be defined."
#             assert new_node.description is not None, "Description must be defined."

#             # Return type -- compute every time!
#             class ReturnInfo(BaseModel):
#                 function_return_type: ExtendedType = Field(
#                     description="The return type of the function implementing this computation stage, as an ExtendedType JSON object.",
#                 )
#                 function_computed_value: Optional[str] = Field(
#                     description="A one sentence description of the value computed by this function.",
#                 )

#             assistant = Assistant(
#                 ["system-prompt", "return-type"],
#                 requirements="\n".join(new_node.requirements),
#             )

#             import flowco.dataflow.extended_type

#             log(flowco.dataflow.extended_type.ExtendedType)
#             log(id(flowco.dataflow.extended_type.ExtendedType))

#             completion = assistant.model_completion(ExtendedType)
#             new_node = new_node.update(
#                 function_return_type=completion,
#             )
#             assert (
#                 new_node.function_return_type is not None
#             ), "Return type must be defined."

#             if new_node.function_return_type.as_actual_type() is not None:
#                 assistant = Assistant(
#                     ["system-prompt", "return-description"],
#                     requirements="\n".join(new_node.requirements),
#                 )

#                 completion = assistant.str_completion()
#                 new_node = new_node.update(
#                     function_computed_value=completion,
#                 )
#                 assert (
#                     new_node.function_computed_value is not None
#                 ), "Computed value must be defined."
#             else:
#                 new_node.function_computed_value = "None"

#             assert new_node.requirements is not None, "Requirements must be defined."
#             assert new_node.description is not None, "Description must be defined."

#             # Add to cache and return updated Node.
#             build_cache = node.build_cache.update(
#                 requirements=RequirementsCache(
#                     in_=RequirementsIn(
#                         function_parameters=params, preconditions=preconditions
#                     ),
#                     out_requirements=new_node.requirements,
#                     out_function_parameters=params,
#                     out_function_return_type=new_node.function_return_type,
#                     out_function_computed_value=new_node.function_computed_value,
#                     out_description=new_node.description,
#                 )
#             )

#             new_node = node.update(
#                 **build_cache.requirements.out(), build_cache=build_cache
#             )

#             # Any checks
#             with logger("Requirements checks"):
#                 assert (
#                     new_node.requirements is not None
#                 ), "Requirements must be defined."
#                 assert (
#                     new_node.function_return_type is not None
#                 ), "Return type must be defined."
#                 assert (
#                     new_node.function_computed_value is not None
#                 ), "Computed value must be defined."
#                 assert (
#                     new_node.function_parameters is not None
#                 ), "Parameters must be defined."
#                 assert new_node.description is not None, "Description must be defined."

#                 assistant = Assistant(
#                     "inconsistent-requirements",
#                     label=new_node.label,
#                     requirements="\n".join(f"* {r}" for r in new_node.requirements),
#                 )

#                 completion = assistant.model_completion(Warning)

#                 if not completion.ok:
#                     for warning in completion.inconsistencies:
#                         node.warn(
#                             phase=Phase.requirements,
#                             message=warning,
#                         )

#             return new_node
#     except Exception as e:
#         print("Biffer", file=sys.stderr)
#         print(e, file=sys.stderr)
#         traceback.print_exc(file=sys.stderr)
#         raise e


# def summarize_changes(node: Node, phase: Phase, diff_map: Dict[str, List[str]]):
#     summary = diff_map["summary"]
#     if summary:
#         node.info(
#             phase=phase,
#             message=f"{summary}",
#         )


# def create_preconditions(graph: DataFlowGraph, node: Node) -> Dict[str, List[str]]:
#     preconditions: Dict[str, List[str]] = {}
#     for pred in node.predecessors:
#         req = graph[pred].requirements
#         assert req is not None, "Predecessor must have requirements."
#         preconditions[graph[pred].function_result_var] = req

#     return preconditions


# def create_parameters(graph, node):
#     params = [
#         Parameter(
#             name=graph[pred].function_result_var,
#             type=graph[pred].function_return_type,
#         )
#         for pred in node.predecessors
#     ]
#     return params


# @node_pass(
#     required_phase=Phase.requirements,
#     target_phase=Phase.algorithm,
#     pred_required_phase=Phase.requirements,
# )
# def algorithm(pass_config: PassConfig, graph: DataFlowGraph, node: Node) -> Node:
#     with logger("Algorithm step"):

#         assert node.requirements is not None, "Requirements must be defined."

#         preconditions = create_preconditions(graph, node)

#         diff = ""
#         if node.build_cache.algorithm is not None:
#             if node.build_cache.algorithm.matches(
#                 requirements=node.requirements,
#                 preconditions=preconditions,
#             ):
#                 if node.algorithm == node.build_cache.algorithm.out_algorithm:
#                     log("Using cache for algorithm.")
#                     return node.update(**node.build_cache.algorithm.out())
#                 elif node.algorithm is not None:
#                     node.warn(
#                         phase=Phase.algorithm,
#                         message="Requirements have changed -- regenerating algorithm.",
#                     )
#             else:
#                 diff_map = node.build_cache.algorithm.diff(
#                     node.requirements, preconditions
#                 )
#                 diff = f"Here are the changes to reflect in the new algorithm:\n```\n{diff_map}\n```\n"
#                 summarize_changes(node, Phase.algorithm, diff_map)

#         completion = complete_node_pass_query(
#             prompt_key="algorithm",
#             prompt_substitutions={
#                 "preconditions": preconditions,
#                 "postconditions": node.requirements,
#                 "diff": diff,
#             },  # type: ignore
#             pass_config=pass_config,
#             node=node,
#             initial_view=NodeView(
#                 node_fields=[
#                     "id",
#                     "pill",
#                     "label",
#                     "requirements",
#                     "description",
#                     "algorithm",
#                 ]
#             ),
#             completion_view=NodeView(node_fields=["algorithm"]),
#         )

#         generated_algorithm = completion.algorithm
#         assert generated_algorithm is not None, "Algorithm must be defined."

#         # if (
#         #     config.regenerate_policy == RegenerationPolicy.merge
#         #     and AbstractionLevel.show_algorithm(config.abstraction_level)  # if hiding algorithm, don't manually merge
#         #     and node.algorithm is not None
#         #     and node.algorithm != generated_algorithm
#         # ):
#         #     response = todo.merge(
#         #         TodoMergeString(
#         #             node_id=node.id,
#         #             phase=Phase.algorithm,
#         #             message="Updating algorithm",
#         #             old=node.algorithm,
#         #             new=generated_algorithm,
#         #         )
#         #     )
#         #     assert response is not None, "Response must be defined."
#         #     assert isinstance(response.result, str), "Response must be a string."
#         #     generated_algorithm = response.result

#         assert generated_algorithm is not None, "Algorithm must be defined."

#         build_cache = node.build_cache.update(
#             algorithm=AlgorithmCache(
#                 in_=AlgorithmIn(
#                     requirements=node.requirements, preconditions=preconditions
#                 ),
#                 out_algorithm=generated_algorithm,
#             )
#         )

#         new_node = node.update(**build_cache.algorithm.out(), build_cache=build_cache)

#         # Any checks, and just return current algorithm value

#         # assistant = Assistant(
#         #     "inconsistent-algorithm",
#         #     label=new_node.label,
#         #     preconditions=create_preconditions(graph, new_node),
#         #     postconditions=new_node.requirements,
#         #     algorithm=new_node.algorithm,
#         # )

#         # class Warning(BaseModel):
#         #     warnings: List[str] = Field(
#         #         default_factory=list, description="Warning messages."
#         #     )

#         # completion = assistant.model_completion(Warning)

#         # for warning in completion.warnings:
#         #     todo.warn(
#         #         TodoWarning(phase=Phase.algorithm, node_id=new_node.id, message=warning)
#         #     )

#         return new_node


# @node_pass(required_phase=Phase.algorithm, target_phase=Phase.code)
# def compile(pass_config: PassConfig, graph: DataFlowGraph, node: Node) -> Node:
#     with logger("Compile step"):

#         assert node.requirements is not None, "Requirements must be defined."
#         assert node.algorithm is not None, "Algorithm must be defined."
#         assert (
#             node.function_parameters is not None
#         ), "Function parameters must be defined."

#         parameter_types = {param.name: param.type for param in node.function_parameters}

#         diff = ""
#         if node.build_cache.compile is not None:
#             if node.build_cache.compile.matches(
#                 signature=node.signature_str(),
#                 parameter_types=parameter_types,
#                 requirements=node.requirements,
#                 algorithm=node.algorithm,
#             ):
#                 if node.code == node.build_cache.compile.out_code:
#                     log("Using cache for algorithm.")
#                     return node.update(**node.build_cache.compile.out())
#                 elif node.code is not None:
#                     node.warn(
#                         phase=Phase.code,
#                         message="Requirements or algorithm may have changed -- regenerate code.",
#                     )
#             else:
#                 diff_map = node.build_cache.compile.diff(
#                     node.signature_str(),
#                     parameter_types,
#                     node.requirements,
#                     node.algorithm,
#                 )
#                 diff = f"Here are the changes to reflect in the new code:\n```\n{diff_map}\n```\n"
#                 summarize_changes(node, Phase.code, diff_map)

#         code = node.code
#         requirements = node.requirements
#         algorithm = node.algorithm
#         parameter_type_str = "\n".join(
#             [f"* {name}: {t.type_description()}" for name, t in parameter_types.items()]
#         )

#         completion = complete_node_pass_query(
#             "compile",
#             {
#                 "signature": node.signature_str(),
#                 "parameter_types": parameter_type_str,
#                 "diff": diff,
#             },
#             pass_config,
#             node,
#             NodeView(
#                 node_fields=[
#                     "description",
#                     "requirements",
#                     "algorithm",
#                     "function_name",
#                     "function_parameters",
#                     "function_return_type",
#                     "code",
#                 ]
#             ),
#             NodeView(node_fields=["code"]),
#         )

#         code = completion.code
#         assert code is not None, "Code must be defined."

#         # go back and add assertions for requirements -- could do in one step,
#         # but want to keep the steps separate for now...
#         completion = complete_node_pass_query(
#             "compile-assertions",
#             {
#                 "function_return_var": node.function_result_var,
#                 "diff": diff,
#             },
#             pass_config,
#             node.update(code=code),
#             NodeView(
#                 node_fields=[
#                     "requirements",
#                     "algorithm",
#                     "code",
#                 ]
#             ),
#             NodeView(node_fields=["code"]),
#         )

#         code = completion.code
#         assert code is not None, "Code must be defined."

#         # if (
#         #     config.regenerate_policy == RegenerationPolicy.merge
#         #     and AbstractionLevel.show_code(config.abstraction_level)  # if hiding code, don't manually merge
#         #     and node.code is not None
#         #     and node.code != code
#         # ):
#         #     result = todo.merge(
#         #         TodoMergeList(
#         #             kind="lines",
#         #             node_id=node.id,
#         #             phase=Phase.code,
#         #             message="Updating Code",
#         #             old=node.code,
#         #             new=code,
#         #         )
#         #     )
#         #     assert result is not None, "Result must be defined."
#         #     assert isinstance(result.result, list), "Result must be a list."
#         #     code = result.result
#         #     assert code is not None, "Code must be defined."

#         build_cache = node.build_cache.update(
#             compile=CompileCache(
#                 in_=CompileIn(
#                     signature=node.signature_str(),
#                     parameter_types=parameter_types,
#                     requirements=requirements,
#                     algorithm=algorithm,
#                 ),
#                 out_code=code,
#             )
#         )

#         new_node = node.update(**build_cache.compile.out(), build_cache=build_cache)

#         return new_node


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

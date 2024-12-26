# import textwrap
# import traceback
# from typing import List, Optional, Tuple

# from pydantic import BaseModel, Field, create_model

# from flowco.assistant.assistant import Assistant
# from flowco.builder.build import PassConfig
# from flowco.dataflow.phase import Phase
# from flowco.dataflow.dfg import DataFlowGraph, Node, SanityCheck, SanityCheckResult
# from flowco.dataflow.graph_completions import (
#     make_node_like,
#     node_like_model,
#     restrict_model_fields,
# )
# from flowco.dataflow.tests import TestCase
# from flowco.page.error_messages import error_message, execution_error_message
# from flowco.page.notebook import (
#     DataFlowNotebookExecutionError,
#     DataflowNotebookTest,
#     NotebookConfig,
# )
# from flowco.util.config import config
# from flowco.util.errors import FlowcoError
# from flowco.util.output import logger, message, warn
# from flowco.util.stopper import Stopper
# from flowco.util.text import strip_ansi


# def check_sanity_checks(
#     pass_config: PassConfig,
#     graph: DataFlowGraph,
#     node: Node,
#     test_uid: Optional[str] = None,
# ) -> Tuple[Node, bool]:
#     assert node.sanity_checks is not None, f"Node {node.id} must have tests"
#     okay = True

#     sanity_checks = node.sanity_checks

#     if test_uid is not None:
#         test_indices = [i for i, ut in enumerate(sanity_checks) if ut.uuid == test_uid]
#     else:
#         test_indices = range(len(sanity_checks))

#     for i in test_indices:
#         try:
#             new_node = _repair_node_test(
#                 pass_config, graph, node, i, max_retries=pass_config.max_retries
#             )
#             if config.diff:
#                 diff = node.diff(new_node)
#                 if diff:
#                     message(diff)
#             node = new_node
#             outcome = SanityCheckResult(outcome=None)
#         except StopIteration:
#             return node, False
#         except Exception as e:
#             okay = False
#             m = f'**{node.pill}:** Failed a test for: "{sanity_checks[i].requirement}"'
#             todo.error(
#                 TodoError(
#                     phase=Phase.tests_checked,
#                     node_id=node.id,
#                     message=m,
#                 )
#             )
#             outcome = SanityCheckResult(outcome=strip_ansi(str(e)))

#         assert node is not None, "Repair must return a node"

#         node = node.update(
#             sanity_check_results={
#                 **node.sanity_check_results,
#                 sanity_checks[i].uuid: outcome,
#             }
#         )

#     return node, okay


# def _repair_node_test(
#     pass_config: PassConfig,
#     graph: DataFlowGraph,
#     node: Node,
#     test_index: int,
#     max_retries: int,
# ) -> Node:
#     assistant = Assistant("repair-system")

#     retries = 0

#     initial_node = make_node_like(
#         node,
#         node_like_model(
#             "description",
#             "function_name",
#             "function_parameters",
#             "function_return_type",
#             "requirements",
#         ),
#     )
#     assert node.sanity_checks is not None, f"Node {node.id} must have tests"

#     while True:
#         if Stopper.should_stop():
#             raise StopIteration()

#         try:
#             assert node.sanity_checks is not None, f"Node {node.id} must have tests"
#             DataflowNotebookTest.run_test(
#                 NotebookConfig(
#                     file_name="unittest.ipynb",
#                     dfg=graph.with_node(node),
#                     data_files=pass_config.spec_files,
#                 ),
#                 node.sanity_checks[test_index],
#             )
#             return node
#         except DataFlowNotebookExecutionError as e:
#             retries += 1
#             message(execution_error_message(f"Test failure in {node.pill}.", e, node))
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
#                     "The attempted repairs are not working.  Ask the user a question to help you understand and fix the problem.",
#                 )
#                 message(f"A Question: {assistant.str_completion()}")

#                 raise e

#             message(f"Repair attempt {retries} of {pass_config.max_retries}")

#             if retries == 1:
#                 assistant.add_prompt_by_key(
#                     "repair-one-test",
#                     node=initial_node,
#                     code=node.code_str(),
#                     test=str(node.sanity_checks[test_index]),  # type: ignore
#                     error=strip_ansi(str(e)),
#                 )
#             else:
#                 assistant.add_prompt_by_key(
#                     "repair-one-test-followup",
#                     error=strip_ansi(str(e)),
#                 )

#             SanityCheckCompletion = restrict_model_fields(
#                 SanityCheck,
#                 "requirement",
#                 "definition",
#                 "call",
#             )

#             TestCompletion = create_model(
#                 "TestCompletion",
#                 explanation=(
#                     str,
#                     Field(description="The explanation of the repair that was made."),
#                 ),
#                 code=(
#                     Optional[List[str]],
#                     Field(
#                         default=None,
#                         description="The corrected function for this computation stage of the data flow graph, stored as a list of source lines. The signature should match the function_name, function_parameters, and function_return_type fields.",
#                     ),
#                 ),
#                 test=(
#                     Optional[SanityCheckCompletion],
#                     Field(  # Use the dynamically created UnitCompletion model
#                         default=None,
#                         description="The corrected test for this computation stage of the data flow graph.",
#                     ),
#                 ),
#             )

#             response = assistant.model_completion(TestCompletion)

#             message(
#                 "\n".join(
#                     textwrap.wrap(
#                         f"Explanation of repair: {response.explanation}",  # type: ignore
#                         subsequent_indent=" " * 4,
#                     )
#                 )
#             )
#             if response.code:
#                 node = node.update(code=response.code)
#             if response.test:
#                 sanity_check = SanityCheck(
#                     uuid=node.sanity_checks[test_index].uuid,
#                     requirement=response.test.requirement,
#                     definition=response.test.definition,
#                     call=response.test.call,
#                 )
#                 node = node.update(sanity_checks=node.sanity_checks[:test_index] + [sanity_check] + node.sanity_checks[test_index + 1 :])  # type: ignore

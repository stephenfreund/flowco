from __future__ import annotations

import json
from typing import Callable, Iterable, List, Literal, Tuple
from flowco.assistant.openai import OpenAIAssistant
from flowco.assistant.stream import StreamingAssistantWithFunctionCalls
from flowco.builder.graph_completions import messages_for_graph, messages_for_node
from flowco.dataflow.dfg import DataFlowGraph, Geometry, Node
from flowco.dataflow.extended_type import ExtendedType
from flowco.dataflow.phase import Phase
from flowco.page.ama_node import AskMeAnythingNode
from flowco.page.page import Page
from flowco.pythonshell.shells import PythonShells
from flowco.session.session import session
from flowco.util.config import config
from flowco.util.errors import FlowcoError
from flowco.util.output import error, log, logger
from pydantic import BaseModel, Field


def check_node_consistency(
    assistant: AskMeAnythingNode, node: Node
) -> Iterable[Tuple[str, Node]]:

    count = 0

    node = node.lower_phase(Phase.requirements)

    yield ("Checking consistency with inputs.", node)

    class ParameterWarnings(BaseModel):
        warnings: List[str] = Field(
            description="Warnings about the parameters of the function."
        )

    prompt = "The parameters and preconditions cannot be changed.  Examine the node's label, requirements and create a 5-10 word warning for each inconsistency."

    completion = assistant.model_completion(
        prompt=prompt, node=node, response_model=ParameterWarnings
    )

    assert completion is not None, "Completion must be defined."
    count += len(completion.warnings)

    if completion.warnings:
        items = "\n".join([f"* {warning}" for warning in completion.warnings])
        message = f"Inconsistencies with inputs:\n{items}\n"
        node = node.warn(phase=Phase.requirements, message=message)

    yield (
        f"Checking consistency of requirements {'and code' if assistant.show_code else ''}.",
        node,
    )

    class ConsistencyWarning(BaseModel):
        warning: str = Field(description="Warning about the node.")
        fix: str = Field(description="Fix for the warning.")

    class ConsistencyWarnings(BaseModel):
        warnings: List[ConsistencyWarning] = Field(
            description="Warnings about the node."
        )

    if assistant.show_code:
        prompt = config.get_prompt("ama_node_editor_sync")
    else:
        prompt = config.get_prompt("ama_node_editor_sync_no_code")

    completion = assistant.model_completion(
        prompt=prompt, node=node, response_model=ConsistencyWarnings
    )
    count += len(completion.warnings)

    for warning in completion.warnings:
        message = f"**Problem:** {warning.warning}  **Suggested Fix:** {warning.fix}"
        node = node.warn(phase=Phase.requirements, message=message)

    if assistant.show_code:
        node = node.update(phase=Phase.requirements)
    else:
        node = node.update(phase=Phase.code)

    yield (f"I found {count} inconsistencies.  Ask to to fix them if you like!", node)

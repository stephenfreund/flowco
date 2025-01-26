from __future__ import annotations

from typing import Callable, Iterable, List, Literal, Tuple

import openai
from flowco.assistant.openai import OpenAIAssistant
from flowco.assistant.stream import StreamingAssistantWithFunctionCalls
from flowco.builder.graph_completions import messages_for_graph, messages_for_node
from flowco.dataflow.dfg import DataFlowGraph, Geometry, Node
from flowco.dataflow.extended_type import ExtendedType
from flowco.dataflow.phase import Phase
from flowco.page.page import Page
from flowco.pythonshell.shells import PythonShells
from flowco.session.session import session
from flowco.util.config import config
from flowco.util.errors import FlowcoError
from flowco.util.output import error, log, logger
from pydantic import BaseModel, Field


class VisibleMessage(BaseModel):
    role: str
    content: str


ReturnType = Tuple[str, str | None]


class update_node(BaseModel):
    label: str = Field(
        description="The new label of the node.  Keep in sync with the requirements, algorithm, and code."
    )
    requirements: List[str] = Field(
        description="A list of requirements that must be true of the return value for the function.  Describe the representation of the return value as well."
    )
    function_return_type: ExtendedType = Field(
        description="The return type of the node."
    )
    code: List[str] = Field(
        description="The code for the node.  Only modify if there is already an code.  The code should be a list of strings, one for each line of code.  The signature must match the original version, except for the return type"
    )


class update_node_requirements(BaseModel):
    label: str = Field(
        description="The new label of the node.  Keep in sync with the requirements, algorithm, and code."
    )
    requirements: List[str] = Field(
        description="A list of requirements that must be true of the return value for the function.  Describe the representation of the return value as well."
    )
    function_return_type: ExtendedType = Field(
        description="The return type of the node."
    )


class AskMeAnythingNode:

    def __init__(self, dfg: DataFlowGraph, show_code: bool):
        self.dfg = dfg
        self.show_code = show_code

        if show_code:
            prompt = "ama_node_editor"
            functions = [
                (
                    self.update_node,
                    openai.pydantic_function_tool(update_node)["function"],
                )
            ]
        else:
            prompt = "ama_node_editor_no_code"
            functions = [
                (
                    self.update_node_requirements,
                    openai.pydantic_function_tool(update_node_requirements)["function"],
                )
            ]

        self.assistant = StreamingAssistantWithFunctionCalls(
            functions,
            ["system-prompt", prompt],
            imports="",
        )
        self.completion_node = None
        self.visible_messages = []

    def update_node(
        self,
        label: str | None = None,
        requirements: List[str] | None = None,
        function_return_type: ExtendedType | None = None,
        code: List[str] | None = None,
    ) -> ReturnType:
        node = self.completion_node
        assert node is not None, "Node must be set before calling update_node"
        mods = []
        log(f"update_node: {label}, {requirements}, {function_return_type}, {code}")
        if code and code != node.code:
            log(f"Updating code to {code}")
            node = node.update(code=code, phase=Phase.algorithm)
            mods.append("code")

        if requirements and requirements != node.requirements:
            log(f"Updating requirements to {requirements}")
            node = node.update(
                requirements=requirements,
                phase=Phase.clean,
            )
            mods.append("requirements")

        if function_return_type:
            function_return_type = ExtendedType.model_validate(function_return_type)
            if function_return_type != node.function_return_type:
                log(
                    f"Updating function_return_type from {node.function_return_type} to {function_return_type}"
                )
                node = node.update(
                    function_return_type=function_return_type,
                    phase=Phase.clean,
                )
            if "requirements" not in mods:
                mods.append("requirements")
        if label and label != node.label:
            log(f"Updating label to {label}")
            node = node.update(label=label, phase=Phase.clean)
            mods.append("label")

        if config.x_trust_ama:
            if "requirements" in mods:
                node = node.update(cache=node.cache.update(Phase.requirements, node))
                node = node.update(phase=Phase.requirements)
            if "algorithm" in mods:
                node = node.update(cache=node.cache.update(Phase.algorithm, node))
                node = node.update(phase=Phase.algorithm)
            if "code" in mods:
                node = node.update(cache=node.cache.update(Phase.code, node))
                node = node.update(phase=Phase.code)

        mod_str = ", ".join(reversed(mods))

        self.completion_node = node

        return (
            f"Updated {mod_str} for {node.pill}",
            node.model_dump_json(
                include=set(
                    [
                        "id",
                        "pill",
                        "label",
                        "predecessors",
                        "requirements",
                        "function_return_type",
                        "code",
                    ]
                ),
                indent=2,
            ),
        )

    def update_node_requirements(
        self,
        label: str | None = None,
        requirements: List[str] | None = None,
        function_return_type: ExtendedType | None = None,
    ) -> ReturnType:
        node = self.completion_node
        assert node is not None, "Node must be set before calling update_node"
        mods = []
        log(
            f"update_node_requirements: {label}, {requirements}, {function_return_type}"
        )
        if requirements and requirements != node.requirements:
            log(f"Updating requirements to {requirements}")
            node = node.update(
                requirements=requirements,
                phase=Phase.clean,
            )
            mods.append("requirements")

        if function_return_type:
            function_return_type = ExtendedType.model_validate(function_return_type)
            if function_return_type != node.function_return_type:
                log(
                    f"Updating function_return_type from {node.function_return_type} to {function_return_type}"
                )
                node = node.update(
                    function_return_type=function_return_type,
                    phase=Phase.clean,
                )
            if "requirements" not in mods:
                mods.append("requirements")
        if label and label != node.label:
            log(f"Updating label to {label}")
            node = node.update(label=label, phase=Phase.clean)
            mods.append("label")

        if config.x_trust_ama:
            if "requirements" in mods:
                node = node.update(cache=node.cache.update(Phase.requirements, node))
                node = node.update(phase=Phase.requirements)
            if "algorithm" in mods:
                node = node.update(cache=node.cache.update(Phase.algorithm, node))
                node = node.update(phase=Phase.algorithm)

        mod_str = ", ".join(reversed(mods))

        self.completion_node = node

        return (
            f"Updated {mod_str} for {node.pill}",
            node.model_dump_json(
                include=set(
                    [
                        "id",
                        "pill",
                        "label",
                        "predecessors",
                        "requirements",
                        "function_return_type",
                        "code",
                    ]
                ),
                indent=2,
            ),
        )

    def complete(
        self, prompt: str, node: Node, show_prompt: bool = True
    ) -> Iterable[str]:
        yield from self._complete(prompt, node, show_prompt)

    def _complete(
        self,
        prompt: str,
        node: Node,
        show_prompt: bool,
    ) -> Iterable[str]:

        markdown = ""

        if self.completion_node is None:
            self.assistant.add_message(
                "user",
                messages_for_graph(
                    graph=self.dfg,
                    graph_fields=["edges", "description"],
                    node_fields=["requirements"],
                ),
            )

            image = self.dfg.to_image_prompt_messages()
            self.assistant.add_message("user", image)

        self.completion_node = node
        self.assistant.add_message(
            "user",
            messages_for_node(
                node=self.completion_node,
                node_fields=(
                    [
                        "id",
                        "pill",
                        "label",
                        "function_parameters",
                        "preconditions",
                        "requirements",
                        "function_return_type",
                        "function_result_var",
                    ]
                    + (["code"] if self.show_code else [])
                ),
            ),
        )

        if show_prompt:
            self.visible_messages += [VisibleMessage(role="user", content=prompt)]

        yield "Working...\n\n"

        self.assistant.add_message("user", prompt)

        for x in self.assistant.str_completion():
            markdown += x
            yield x

        self.visible_messages += [VisibleMessage(role="assistant", content=markdown)]

    def updated_node(self) -> Node | None:
        return self.completion_node

    def last_message(self) -> VisibleMessage:
        return self.visible_messages[-1]

    def __len__(self) -> int:
        return len(self.visible_messages)

    def messages(self) -> Iterable[VisibleMessage]:
        for message in self.visible_messages:
            yield message

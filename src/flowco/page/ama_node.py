from __future__ import annotations

from typing import Annotated, Iterable, List

from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)
from flowco.assistant.flowco_assistant import flowco_assistant
from flowco.builder.graph_completions import json_for_graph_view, json_for_node_view
from flowco.dataflow.dfg import DataFlowGraph, Node
from flowco.dataflow.extended_type import ExtendedType
from flowco.dataflow.phase import Phase
from flowco.util.config import config
from flowco.util.output import log
from pydantic import BaseModel

from flowco.llm.assistant import ToolCallResult


class VisibleMessage(BaseModel):
    role: str
    content: str


class AskMeAnythingNode:

    def __init__(self, dfg: DataFlowGraph, show_code: bool, is_plot_node: bool):
        self.dfg = dfg
        self.show_code = show_code

        if is_plot_node:
            if show_code:
                prompt = "ama_node_editor-for-plot"
                functions = [
                    self.update_node_for_plot,
                ]
            else:
                prompt = "ama_node_editor_no_code-for-plot"
                functions = [
                    self.update_node_requirements_for_plot,
                ]
        else:
            if show_code:
                prompt = "ama_node_editor"
                functions = [
                    self.update_node,
                ]
            else:
                prompt = "ama_node_editor_no_code"
                functions = [
                    self.update_node_requirements,
                ]

        self.assistant = flowco_assistant(
            f"ama-node",
            prompt_key=prompt)
        self.assistant.set_functions(functions)
        self.completion_node = None
        self.visible_messages = []

    def update_node(
        self,
        label: Annotated[
            str,
            "The new label of the node.",
        ],
        requirements: Annotated[
            List[str],
            "A list of requirements that must be true of the return value for the function.  Describe the representation of the return value as well.",
        ],
        function_return_type: Annotated[ExtendedType, "The return type of the node."],
        code: Annotated[
            List[str] | None,
            "The code for the node.  Only modify if there is already code.  The code should be a list of strings, one for each line of code.  The signature must match the original version, except for the return type",
        ],
    ) -> ToolCallResult:
        """
        Use this to modify the node.  The label, requirements, function_return_type, and code must be kept in sync.
        Make as few changes as possible.
        """
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

        if config().x_trust_ama:
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

        return ToolCallResult(
            user_message=f"Updated {mod_str} for {node.pill}",
            content=ChatCompletionContentPartTextParam(
                type="text",
                text=node.model_dump_json(
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
            ),
        )

    def update_node_requirements(
        self,
        label: Annotated[
            str,
            "The new label of the node.  Keep in sync with the requirements, algorithm, and code.",
        ],
        requirements: Annotated[
            List[str],
            "A list of requirements that must be true of the return value for the function.  Describe the representation of the return value as well.",
        ],
        function_return_type: Annotated[ExtendedType, "The return type of the node."],
    ) -> ToolCallResult:
        """
        Use this to modify the node.  The label, requirements, and function_return_type must be kept in sync.
        Make as few changes as possible.
        """
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

        if config().x_trust_ama:
            if "requirements" in mods:
                node = node.update(cache=node.cache.update(Phase.requirements, node))
                node = node.update(phase=Phase.requirements)
            if "algorithm" in mods:
                node = node.update(cache=node.cache.update(Phase.algorithm, node))
                node = node.update(phase=Phase.algorithm)

        mod_str = ", ".join(reversed(mods))

        self.completion_node = node

        return ToolCallResult(
            user_message=f"Updated {mod_str} for {node.pill}",
            content=ChatCompletionContentPartTextParam(
                type="text",
                text=node.model_dump_json(
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
            ),
        )

    def update_node_for_plot(
        self,
        label: Annotated[
            str,
            "The new label of the node.",
        ],
        requirements: Annotated[
            List[str],
            "A list of requirements that must be true for this node.",
        ],
        code: Annotated[
            List[str] | None,
            "The code for the node.  Only modify if there is already code.  The code should be a list of strings, one for each line of code.  The signature must match the original version.",
        ],
    ) -> ToolCallResult:
        """
        Use this to modify the node.  The label, requirements and code must be kept in sync.
        Make as few changes as possible.
        """
        node = self.completion_node
        assert node is not None, "Node must be set before calling update_node"
        mods = []
        log(f"update_node: {label}, {requirements}, {code}")
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

        if label and label != node.label:
            log(f"Updating label to {label}")
            node = node.update(label=label, phase=Phase.clean)
            mods.append("label")

        if config().x_trust_ama:
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

        return ToolCallResult(
            user_message=f"Updated {mod_str} for {node.pill}",
            content=ChatCompletionContentPartTextParam(
                type="text",
                text=node.model_dump_json(
                    include=set(
                        [
                            "id",
                            "pill",
                            "label",
                            "predecessors",
                            "requirements",
                            "code",
                        ]
                    ),
                    indent=2,
                ),
            ),
        )

    def update_node_requirements_for_plot(
        self,
        label: Annotated[
            str,
            "The new label of the node.  Keep in sync with the requirements, algorithm, and code.",
        ],
        requirements: Annotated[
            List[str],
            "A list of requirements that must be true.",
        ],
    ) -> ToolCallResult:
        """
        Use this to modify the node.  The label and requirements must be kept in sync.
        Make as few changes as possible.
        """
        node = self.completion_node
        assert node is not None, "Node must be set before calling update_node"
        mods = []
        log(f"update_node_requirements: {label}, {requirements}")
        if requirements and requirements != node.requirements:
            log(f"Updating requirements to {requirements}")
            node = node.update(
                requirements=requirements,
                phase=Phase.clean,
            )
            mods.append("requirements")

        if label and label != node.label:
            log(f"Updating label to {label}")
            node = node.update(label=label, phase=Phase.clean)
            mods.append("label")

        if config().x_trust_ama:
            if "requirements" in mods:
                node = node.update(cache=node.cache.update(Phase.requirements, node))
                node = node.update(phase=Phase.requirements)
            if "algorithm" in mods:
                node = node.update(cache=node.cache.update(Phase.algorithm, node))
                node = node.update(phase=Phase.algorithm)

        mod_str = ", ".join(reversed(mods))

        self.completion_node = node

        return ToolCallResult(
            user_message=f"Updated {mod_str} for {node.pill}",
            content=ChatCompletionContentPartTextParam(
                type="text",
                text=node.model_dump_json(
                    include=set(
                        [
                            "id",
                            "pill",
                            "label",
                            "predecessors",
                            "requirements",
                            "code",
                        ]
                    ),
                    indent=2,
                ),
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
            self.assistant.add_text("user", "Here is the current dataflow graph.")
            self.assistant.add_json(
                "user",
                json_for_graph_view(
                    graph=self.dfg,
                    graph_fields=["edges", "description"],
                    node_fields=["requirements"],
                ),
            )

            image = self.dfg.to_image_url()
            if image:
                self.assistant.add_text(
                    "user", "Here is a picture of the dataflow graph."
                )
                self.assistant.add_image("user", image)

        self.completion_node = node
        self.assistant.add_text("user", f"Here is the current node: {node.pill}")
        self.assistant.add_json(
            "user",
            json_for_node_view(
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

        self.assistant.add_text("user", prompt)

        for x in self.assistant.stream():
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

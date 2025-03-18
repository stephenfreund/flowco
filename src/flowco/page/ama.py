from __future__ import annotations

import traceback
from typing import Annotated, Iterable, List, Literal

from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)
from regex import P


from flowco.assistant.flowco_assistant import flowco_assistant, flowco_assistant_fast
from flowco.builder.graph_completions import json_for_graph_view
from flowco.builder.synthesize import create_parameters, create_preconditions
from flowco.dataflow.dfg import Geometry, NodeKind
from flowco.dataflow.extended_type import ExtendedType
from flowco.dataflow.phase import Phase
from flowco.page.page import Page
from flowco.page.tables import GlobalTables
from flowco.pythonshell.shells import PythonShells
from flowco.session.session import session
from flowco.ui.mx_diagram import DiagramGroup
from flowco.util.config import config
from flowco.util.output import error, log, logger
from pydantic import BaseModel


from flowco.dataflow.dfg_update import (
    mxDiagramUpdate,
    DiagramEdgeUpdate,
    DiagramNodeUpdate,
    update_dataflow_graph,
)
from flowco.util.text import strip_ansi
from llm.assistant import ToolCallResult


class VisibleMessage(BaseModel):
    role: str
    content: str
    is_error: bool = False


class QuestionKind(BaseModel):
    kind: Literal["Explain"] | Literal["Modify"]


class AskMeAnything:

    def __init__(self, page: Page):
        self.page = page
        self.reset()
        self.visible_messages: List[VisibleMessage] = []

    def reset(self):
        """Reset internals"""
        self.assistant = flowco_assistant(prompt_key="ama_general")
        self.shell = None
        self.completion_dfg = None

    def python_eval(
        self, code: Annotated[str, "The Python code to evaluate"]
    ) -> ToolCallResult:
        """
        Evaluate python code.  You may assume numpy, scipy, and pandas are available.",
        """
        init_code = ""
        for node in self.page.dfg.nodes:
            result = node.result
            if (
                result is not None
                and result.result is not None
                and node.function_return_type is not None
                and not node.function_return_type.is_None_type()
            ):
                value, _ = result.result.to_repr()
                init_code += f"{node.function_result_var} = {value}\n"

        tables = GlobalTables.from_dfg(self.page.dfg)
        init_code += "\n".join(tables.function_defs())

        shell_code = f"{init_code}\n{code}"
        with logger("python_eval"):
            try:
                log(f"Code:\n{shell_code}")
                result = session.get("shells", PythonShells).run(shell_code)
                result_output = result.as_result_output()
                if result_output is not None:
                    log(f"Result:\n{result_output}")
                    return ToolCallResult(
                        user_message=f"**:blue[Okay, I ran some code:]**\n```\n{code}\n```\n",
                        content=(
                            result_output.to_content_part()
                            if result_output is not None
                            else None
                        ),
                    )

                else:
                    return ToolCallResult(
                        user_message=f"**:blue[Okay, I ran some code:]**\n```\n{code}\n```\n. It produced no output",
                        content=None,
                    )
            except Exception as e:
                error(f"Error running code: {e}")
                e_str = strip_ansi(str(e).splitlines()[-1])
                return ToolCallResult(
                    user_message=f"**:red[I had an error running this code:]**\n```\n{code}\n```\n\n**:red[Error:]**\n```\n{e_str}\n```\n",
                    content=ChatCompletionContentPartTextParam(type="text", text=e_str),
                )

    def inspect(
        self, id: Annotated[str, "The id of the node whose output to inspect"]
    ) -> ToolCallResult:
        """
        Inspect the output for a node in the diagram, including any generated plots.
        """
        log(f"inspect: {id}")
        if id not in self.page.dfg.node_ids():
            return ToolCallResult(
                user_message=f"**:red[Node {id} does not exist]**", content=None
            )
        node = self.page.dfg[id]
        result = node.result
        if result is None:
            return ToolCallResult(
                user_message=f"**:red[No output or result for {node.pill}]**",
                content=None,
            )
        else:
            if result.output is not None:
                return ToolCallResult(
                    user_message=f"**:blue[I inspected the output for {node.pill}]**",
                    content=result.output.to_content_part(),
                )
            elif result.result is not None:
                return ToolCallResult(
                    user_message=f"**:blue[I inspected the result for {node.pill}]**",
                    content=result.result.to_content_part(),
                )
            else:
                return ToolCallResult(
                    user_message=f"**:red[No output or result for {node.pill}]**",
                    content=None,
                )

    def update_node(
        self,
        id: Annotated[str, "The id of the node to update"],
        pill: Annotated[
            str,
            "The new pill of the node.",
        ],
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
        log(f"update_node: {id}, {requirements},  {code}")
        dfg = self.page.dfg

        if dfg[id] is None:
            return ToolCallResult(
                user_message=f"**:red[Node {id} does not exist]**", content=None
            )

        node = dfg[id]
        mods = []

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
            if (
                node.function_return_type is not None
                and function_return_type != node.function_return_type
            ):
                log(
                    f"Updating function_return_type from {node.function_return_type.to_markdown(True)} to {function_return_type.to_markdown(True)}"
                )
                node = node.update(
                    function_return_type=function_return_type,
                    phase=Phase.clean,
                )
            mods.append("return-type")
        if label and label != node.label:
            log(f"Updating label to {label}")
            node = node.update(label=label, phase=Phase.clean)
            mods.append("label")
        if pill and pill != node.pill:
            log(f"Updating pill to {pill}")
            node = node.update(pill=pill, phase=Phase.clean)
            mods.append("pill")

        if node.phase == Phase.clean:
            dfg = dfg.lower_phase_with_successors(node.id, Phase.clean)

        if config().x_trust_ama:
            if "requirements" in mods or "return-type" in mods:
                node = node.update(cache=node.cache.update(Phase.requirements, node))
                node = node.update(phase=Phase.requirements)
            if "code" in mods:
                node = node.update(cache=node.cache.update(Phase.code, node))
                node = node.update(phase=Phase.code)

        dfg = dfg.with_node(node).reduce_phases_to_below_target(node.id, node.phase)
        self.page.update_dfg(dfg)

        mod_str = ", ".join(reversed(mods))

        return ToolCallResult(
            user_message=f"**:blue[I updated {mod_str} for {node.pill}]**",
            content=ChatCompletionContentPartTextParam(
                type="text", text=node.model_dump_json(indent=2)
            ),
        )

    def add_node_for_a_compute_step(
        self,
        id: Annotated[
            str, "A unique id for the new node.  No spaces or special characters."
        ],
        predecessors: Annotated[List[str], "The ids of the predecessor nodes"],
        label: Annotated[str, "The label of the new node"],
        requirements: Annotated[
            List[str],
            "A list of requirements that must be true of the return value for the function.  Describe the representation of the return value as well.",
        ],
        function_return_type: Annotated[ExtendedType, "The return type of the node."],
    ) -> ToolCallResult:
        """
        Add a node to compute a value to the diagram.  Do not provide code.  Nodes should represent one small step in a pipeline.
        Provide a list of nodes that should point to the new node.
        Provide a unique id for the node, a list of predecessor nodes, a label, and a list of requirements that must be true of the return value for the function.
        Describe the representation of the return value as well.
        """
        log(
            f"add_node: {id}, {predecessors}, {label}, {requirements}, {function_return_type}"
        )
        dfg = self.page.dfg

        pill = "tmp-pill"

        if dfg.node_for_pill(pill) is not None:
            return ToolCallResult(
                user_message=f"**:red[Node with pill {pill} already exists]**",
                content=None,
            )

        for pred in predecessors:
            if dfg[pred] is None:
                return ToolCallResult(
                    user_message=f"**:red[predecessor {pred} does not exist]**",
                    content=None,
                )

        pill = dfg.make_pill(label)
        geometry = Geometry(x=0, y=0, width=0, height=0)
        output_geometry = geometry.translate(geometry.width + 100, 0).resize(120, 80)
        node_updates = {
            x.id: DiagramNodeUpdate(
                id=x.id,
                pill=x.pill,
                label=x.label,
                geometry=x.geometry,
                output_geometry=x.output_geometry,
                is_locked=x.is_locked,
                force_show_output=x.force_show_output,
            )
            for x in dfg.nodes
        } | {
            id: DiagramNodeUpdate(
                id=id,
                pill=pill,
                label=label,
                geometry=geometry,
                output_geometry=output_geometry,
                is_locked=False,
                force_show_output=False,
            )
        }
        edge_updates = {
            x.id: DiagramEdgeUpdate(id=x.id, src=x.src, dst=x.dst) for x in dfg.edges
        } | {
            f"{src}-{id}": DiagramEdgeUpdate(id=f"{src}-{id}", src=src, dst=id)
            for src in predecessors
        }

        dfg = update_dataflow_graph(
            dfg,
            mxDiagramUpdate(
                version=dfg.version,
                nodes=node_updates,
                edges=edge_updates,
                groups=[
                    DiagramGroup(
                        id=x.id,
                        label=x.label,
                        is_collapsed=x.is_collapsed,
                        collapsed_geometry=x.collapsed_geometry,
                        parent_group=x.parent_group,
                        nodes=x.nodes,
                    )
                    for x in dfg.groups
                ],
            ),
        )

        node = dfg[id]

        node = node.update(
            requirements=requirements,
            function_return_type=ExtendedType.model_validate(function_return_type),
            kind=NodeKind.compute,
        )

        dfg = dfg.with_node(node).reduce_phases_to_below_target(node.id, node.phase)
        self.page.update_dfg(dfg)

        src_pills = ", ".join(dfg[x].pill for x in predecessors)
        if src_pills:
            message = f"I add a new node {node.pill}, and connected these nodes to it: {src_pills}"
        else:
            message = f"I add a new node {node.pill}"
        return ToolCallResult(
            user_message=f"**:blue[{message}]**",
            content=ChatCompletionContentPartTextParam(
                type="text", text=node.model_dump_json(indent=2)
            ),
        )

    def add_node_for_a_plot(
        self,
        id: Annotated[
            str, "A unique id for the new node.  No spaces or special characters."
        ],
        predecessors: Annotated[List[str], "The ids of the predecessor nodes"],
        label: Annotated[str, "The label of the new node"],
        requirements: Annotated[
            List[str],
            "A list of requirements that must be true of the return value for the function.  Describe the representation of the return value as well.",
        ],
    ) -> ToolCallResult:
        """
        Add a node to make a plot to the diagram.  Do not provide code.
        Provide a list of nodes that should point to the new node.
        Provide a unique id for the node, a list of predecessor nodes, a label, and a list of requirements for the plot.
        """
        log(f"add_node: {id}, {predecessors}, {label}, {requirements}")
        dfg = self.page.dfg

        pill = "tmp-pill"

        if dfg.node_for_pill(pill) is not None:
            return ToolCallResult(
                user_message=f"**:red[Node with pill {pill} already exists]**",
                content=None,
            )

        for pred in predecessors:
            if dfg[pred] is None:
                return ToolCallResult(
                    user_message=f"**:red[predecessor {pred} does not exist]**",
                    content=None,
                )

        pill = dfg.make_pill(label)
        geometry = Geometry(x=0, y=0, width=0, height=0)
        output_geometry = geometry.translate(geometry.width + 100, 0).resize(120, 80)
        node_updates = {
            x.id: DiagramNodeUpdate(
                id=x.id,
                pill=x.pill,
                label=x.label,
                geometry=x.geometry,
                output_geometry=x.output_geometry,
                is_locked=x.is_locked,
                force_show_output=x.force_show_output,
            )
            for x in dfg.nodes
        } | {
            id: DiagramNodeUpdate(
                id=id,
                pill=pill,
                label=label,
                geometry=geometry,
                output_geometry=output_geometry,
                is_locked=False,
                force_show_output=False,
            )
        }
        edge_updates = {
            x.id: DiagramEdgeUpdate(id=x.id, src=x.src, dst=x.dst) for x in dfg.edges
        } | {
            f"{src}-{id}": DiagramEdgeUpdate(id=f"{src}-{id}", src=src, dst=id)
            for src in predecessors
        }

        dfg = update_dataflow_graph(
            dfg,
            mxDiagramUpdate(
                version=dfg.version,
                nodes=node_updates,
                edges=edge_updates,
                groups=[
                    DiagramGroup(
                        id=x.id,
                        label=x.label,
                        is_collapsed=x.is_collapsed,
                        collapsed_geometry=x.collapsed_geometry,
                        parent_group=x.parent_group,
                        nodes=x.nodes,
                    )
                    for x in dfg.groups
                ],
            ),
        )

        node = dfg[id]
        node = node.update(
            requirements=requirements,
            function_return_type=ExtendedType.from_value(None),
            kind=NodeKind.plot,
        )
        dfg = dfg.with_node(node)

        dfg = dfg.with_node(node).reduce_phases_to_below_target(node.id, node.phase)

        src_pills = ", ".join(dfg[x].pill for x in predecessors)
        if src_pills:
            message = f"I add a new node {node.pill}, and connected these nodes to it: {src_pills}"
        else:
            message = f"I add a new node {node.pill}"
        return ToolCallResult(
            user_message=f"**:blue[{message}]**",
            content=ChatCompletionContentPartTextParam(
                type="text", text=node.model_dump_json(indent=2)
            ),
        )

    def add_edge(
        self,
        src_id: Annotated[str, "The id of the src node"],
        dst_id: Annotated[str, "The id of the dst node"],
    ) -> ToolCallResult:
        """
        Add an edge to the diagram
        """
        log(f"add_edge: {src_id}, {dst_id}")
        dfg = self.page.dfg

        for id in [src_id, dst_id]:
            if dfg[id] is None:
                return ToolCallResult(
                    user_message=f"**:red[Node {id} does not exist]", content=None
                )

        dfg = dfg.with_new_edge(src_id, dst_id).lower_phase_with_successors(
            dst_id, Phase.clean
        )
        self.page.update_dfg(dfg)

        # find id for that edge
        edge = dfg.edge_for_nodes(src_id, dst_id)

        assert edge is not None, f"Edge not found for {src_id} to {dst_id}"

        src_pill = dfg[src_id].pill
        dst_pill = dfg[dst_id].pill

        return ToolCallResult(
            user_message=f"**:blue[I added a new edge from {src_pill} to {dst_pill}]**",
            content=ChatCompletionContentPartTextParam(
                type="text", text=edge.model_dump_json(indent=2)
            ),
        )

    def remove_node(
        self, id: Annotated[str, "The id of the node to remove"]
    ) -> ToolCallResult:
        """
        Remove a node from the diagram.
        """
        log(f"remove_node: {id}")
        dfg = self.page.dfg

        if dfg[id] is None:
            return ToolCallResult(
                user_message=f"**:red[Node {id} does not exist]**", content=None
            )

        node = dfg[id]
        dfg_update = mxDiagramUpdate(
            version=dfg.version,
            nodes={
                x.id: DiagramNodeUpdate(
                    id=x.id,
                    pill=x.pill,
                    label=x.label,
                    geometry=x.geometry,
                    output_geometry=x.output_geometry,
                    is_locked=x.is_locked,
                    force_show_output=x.force_show_output,
                )
                for x in dfg.nodes
                if x.id != id
            },
            edges={
                x.id: DiagramEdgeUpdate(id=x.id, src=x.src, dst=x.dst)
                for x in dfg.edges
                if not (x.src == id or x.dst == id)
            },
            groups=[
                DiagramGroup(
                    id=x.id,
                    label=x.label,
                    is_collapsed=x.is_collapsed,
                    collapsed_geometry=x.collapsed_geometry,
                    parent_group=x.parent_group,
                    nodes=[y for y in x.nodes if y != id],
                )
                for x in dfg.groups
            ],
        )

        dfg = update_dataflow_graph(dfg, dfg_update)
        dfg = dfg.with_node(node).reduce_phases_to_below_target(node.id, Phase.clean)
        return ToolCallResult(
            user_message=f"**:blue[I removed node {node.pill}]**", content=None
        )

    def remove_edge(
        self, id: Annotated[str, "The id of the edge to remove"]
    ) -> ToolCallResult:
        """
        Remove an edge from the diagram
        """
        log(f"remove_edge: {id}")
        dfg = self.page.dfg

        edge_to_remove = dfg.get_edge(id)

        if edge_to_remove is None:
            return ToolCallResult(
                user_message="**:red[Edge has already been removed]**", content=None
            )

        dfg_update = mxDiagramUpdate(
            version=dfg.version,
            nodes={
                x.id: DiagramNodeUpdate(
                    id=x.id,
                    pill=x.pill,
                    label=x.label,
                    geometry=x.geometry,
                    output_geometry=x.output_geometry,
                    is_locked=x.is_locked,
                    force_show_output=x.force_show_output,
                )
                for x in dfg.nodes
            },
            edges={
                x.id: DiagramEdgeUpdate(id=x.id, src=x.src, dst=x.dst)
                for x in dfg.edges
                if x.id != id
            },
            groups=[
                DiagramGroup(
                    id=x.id,
                    label=x.label,
                    is_collapsed=x.is_collapsed,
                    collapsed_geometry=x.collapsed_geometry,
                    parent_group=x.parent_group,
                    nodes=x.nodes,
                )
                for x in dfg.groups
            ],
        )

        src_pill = dfg[edge_to_remove.src].pill
        dst_pill = dfg[edge_to_remove.dst].pill

        dfg = update_dataflow_graph(dfg, dfg_update).lower_phase_with_successors(
            dst_pill, Phase.clean
        )
        self.page.update_dfg(dfg)

        return ToolCallResult(
            user_message=f"**:blue[I removed edge from {src_pill} to {dst_pill}]**",
            content=None,
        )

    def classify_question(self, question: str) -> str:
        assistant = flowco_assistant_fast(prompt_key="classify_ama_prompt")

        for message in self.visible_messages[-4:]:
            assistant.add_content_parts(
                role=message.role,
                content=ChatCompletionContentPartTextParam(
                    type="text", text=message.content
                ),
            )
        assistant.add_text("user", f"Classify this prompt:\n```\n{question}\n```\n")
        return str(assistant.model_completion(QuestionKind).kind)

    def complete(self, prompt: str, selected_node: str | None = None) -> Iterable[str]:
        try:
            kind = self.classify_question(prompt)

            with logger(f"AMA: {kind}"):
                if kind == "Explain":
                    yield from self._complete(
                        "ama_explain",
                        [self.python_eval, self.inspect],
                        prompt,
                        selected_node,
                    )

                elif kind == "Modify":
                    original_dfg = self.page.dfg
                    yield from self._complete(
                        "ama_modify",
                        [
                            self.python_eval,
                            self.inspect,
                            self.add_node_for_a_compute_step,
                            self.add_node_for_a_plot,
                            self.add_edge,
                            self.remove_node,
                            self.remove_edge,
                            self.update_node,
                        ],
                        prompt,
                        selected_node,
                    )
                else:
                    raise ValueError(f"Unknown kind: {kind}")
        except Exception as e:
            error(e)
            self.visible_messages += [
                VisibleMessage(
                    role="assistant",
                    content=f"Error: {e}\n{traceback.format_exc()}",
                    is_error=True,
                )
            ]

    def _complete(
        self, system_prompt, functions, prompt: str, selected_node: str | None = None
    ) -> Iterable[str]:

        markdown = ""

        self.assistant.set_functions(functions)

        if (
            self.completion_dfg is None
            or self.completion_dfg.nodes != self.page.dfg.nodes
            or self.completion_dfg.edges != self.page.dfg.edges
        ):
            log("Recomputing completion DFG prompts")
            self.completion_dfg = self.page.dfg
            locals = "The following variables are already defined.  You may use them in any code you run via a function call.\n"

            for node in self.page.dfg.nodes:
                result = node.result
                if (
                    result is not None
                    and result.result is not None
                    and node.function_return_type is not None
                    and not node.function_return_type.is_None_type()
                ):
                    # type_description = node.function_return_type.type_description()
                    locals += f"`{node.function_result_var} : {node.function_return_type.to_python_type()}` is {node.function_return_type.description}\n\n"

            tables = GlobalTables.from_dfg(self.page.dfg)
            locals += "\nYou have access to these files:\n" + str(
                tables.as_preconditions()
            )

            image = self.page.dfg.to_image_url()
            if image is not None:
                self.assistant.add_text("user", "Here is the current diagram:")
                self.assistant.add_image("user", url=image)

            node_fields = [
                "id",
                "pill",
                "label",
                "requirements",
                "function_return_type",
                "code",
                "messages",
            ]

            self.assistant.add_text("user", "Here is the current data flow graph:")
            self.assistant.add_json(
                "user",
                json_for_graph_view(
                    self.page.dfg, graph_fields=["edges"], node_fields=node_fields
                ),
            )

            self.assistant.add_text("user", locals)

        if selected_node is not None:
            self.assistant.add_text(
                "user",
                f"The currently selected node in the diagram is: `{selected_node}`",
            )

        self.visible_messages += [VisibleMessage(role="user", content=prompt)]
        self.assistant.add_text("system", config().get_prompt(system_prompt))
        self.assistant.add_text("user", prompt)

        for x in self.assistant.stream():
            markdown += x
            yield x

        with_embedded_images = self.assistant.replace_placeholders_with_base64_images(
            markdown
        )
        with_embedded_images = self.page.dfg.replace_placeholders_with_base64_images(
            with_embedded_images
        )

        self.visible_messages += [
            VisibleMessage(role="assistant", content=with_embedded_images)
        ]

    def last_message(self) -> VisibleMessage:
        return self.visible_messages[-1]

    def __len__(self) -> int:
        return len(self.visible_messages)

    def messages(self) -> Iterable[VisibleMessage]:
        for message in self.visible_messages:
            yield message

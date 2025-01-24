from __future__ import annotations
import traceback
import black
from collections import deque
import os
import textwrap
from typing import Any, Dict, List, Literal, Optional, OrderedDict, Set, Union
from graphviz import Digraph
from pydantic import BaseModel, Field

import base64

import openai

from flowco.builder.cache import BuildCache
from flowco.dataflow.checks import Check, CheckOutcomes
from flowco.dataflow.extended_type import ExtendedType
from flowco.dataflow.function_call import FunctionCall
from flowco.dataflow.parameter import Parameter
from flowco.dataflow.phase import Phase
from flowco.dataflow.tests import (
    UnitTest,
)
from flowco.page.output import NodeResult, OutputType
from flowco.util.config import config
from flowco.util.errors import FlowcoError
from flowco.util.text import (
    black_format,
    format_basemodel,
    pill_to_function_name,
    pill_to_python_name,
    pill_to_result_var_name,
)
from flowco.util.output import error, logger, log

import jsonmerge

import deepdiff


class FailOnDifferentValueStrategy(jsonmerge.strategies.Strategy):
    def merge(self, walk, base, head, schema, **kwargs):
        if base.val != head.val:
            raise jsonmerge.JSONMergeError(
                f"Merge conflict: tried to merge different value '{head}' into '{base}'"
            )
        return base


class NodeLike(BaseModel):
    """
    Any class that has id,pill,label,predecessors fields, and also any other fields from Node.
    """


class Geometry(BaseModel):
    x: float
    y: float
    width: float
    height: float

    def translate(self, dx: float, dy: float):
        return Geometry(
            x=self.x + dx, y=self.y + dy, width=self.width, height=self.height
        )

    def resize(self, width: float, height: float):
        return Geometry(x=self.x, y=self.y, width=width, height=height)


class NodeMessage(BaseModel):
    level: Literal["info", "warning", "error"] = Field(
        description="The level of the message."
    )
    phase: Phase = Field(
        description="The phase of the computation stage generating this message."
    )
    text: str = Field(description="The message generated during the build process.")

    def title(self) -> str:
        return f"{self.level.title()}@{self.phase}"

    def message(self) -> str:
        return self.text

    def __str__(self) -> str:
        return f"[{self.level.title()}@{self.phase}]: {self.text}"


class NodeQuestions(BaseModel):
    phase: Phase = Field(
        description="The phase of the computation stage generating this question."
    )
    questions: List[str] = Field(description="The assistant in need of answers.")


class Node(NodeLike, BaseModel):
    """
    A node in a data flow graph.
    """

    def __init__(self, **data):

        code = data.get("code", None)
        if code is not None:
            data["code"] = black_format(code)

        super().__init__(**data)

    # From Diagram:
    id: str = Field(
        description="A unique identifier for this computation stage. ids must never change.",
    )

    pill: str = Field(description="A phrase to idenfity this node.")

    label: str = Field(
        description="A short summary of what this stage of the computation does.",
    )

    geometry: Geometry = Field(
        description="The geometry of the node in the diagram.",
    )
    output_geometry: Optional[Geometry] = Field(
        default=None,
        description="The geometry of the output of the node in the diagram.",
    )

    is_locked: bool = Field(
        default=False,
        description="Whether the node is locked and cannot be modified by the LLM.",
    )

    is_output_visible: bool = Field(
        default=True,
        description="Whether the output of the node is visible in the diagram.",
    )

    # Update with pill

    function_name: str = Field(
        description="The name of the function implementing this computation stage.",
    )

    function_result_var: str = Field(
        description="The variable name to store the result of this computation stage.",
    )

    # From diagram to set order as soon as we create the node.
    predecessors: List[str] = Field(
        description="The ids of the nodes that are predecessors to this node in the data flow graph.",
    )

    # Managed by build
    phase: Phase = Field(
        default=Phase.clean,
        description="The phase of the computation stage for this node in the data flow graph.",
    )

    # Requirements Phase: From preds
    function_parameters: Optional[List[Parameter]] = Field(
        default=None,
        description="The parameters of the function implementing this computation stage.",
    )

    preconditions: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="A map from predecessor to a list of preconditions that must be true for the function to be called.",
    )

    # Requirements Phase: generated
    requirements: Optional[List[str]] = Field(
        default=None,
        description="A list of requirements that must be true of the return value for the function.  Describe the representation of the return value as well.",
    )

    description: Optional[str] = Field(
        default=None,
        description="A 3-5 sentence description of what this stage of the computation does.",
    )

    function_return_type: Optional[ExtendedType] = Field(
        default=None,
        # NOTE: Do not put description here, or scheme includes an allOf, which OpenAI cannot handle...
        #  description="The return type of the function implementing this computation stage.",
    )

    function_computed_value: Optional[str] = Field(
        default=None,
        description="A description of computed value of the function implementing this computation stage.",
    )

    # Algorithm Phase
    algorithm: Optional[List[str]] = Field(
        default=None,
        description="The algorithm used to generate the output.  Use Markdown text to describe the algorithm.",
    )

    # Compile Phase
    code: Optional[List[str]] = Field(
        default=None,
        description="The function for this computation stage of the data flow graph, stored as a list of source lines.  The signature should match the function_name, function_parameters, and function_return_type fields.  Don't include newlines in the strings.",
    )

    #####

    assertions: Optional[List[str]] = Field(
        default=None, description="A list of assertions that must be true at run time."
    )

    assertion_checks: Optional[OrderedDict[str, Check]] = Field(
        default_factory=OrderedDict, description="The generated assertions."
    )

    assertion_outcomes: Optional[CheckOutcomes] = Field(
        default_factory=CheckOutcomes, description="The outcomes of the assertions."
    )

    #####

    unit_tests: Optional[List[UnitTest]] = Field(
        default=None, description="A list of unit tests that the node must pass."
    )

    ###

    messages: List[NodeMessage] = Field(
        default_factory=list,
        description="A list of messages generated during the build process.",
    )

    build_status: Optional[str] = Field(
        default=None, description="The status of the build process for this node."
    )

    # Runnable Phase
    result: Optional[NodeResult] = Field(
        default=None,
        description="The result of the computation stage.",
    )

    cache: BuildCache = Field(
        default_factory=BuildCache,
        description="A cache of the build process for this node.",
    )

    def semantically_eq(self, other: "Node") -> bool:
        return (
            self.id == other.id
            and self.pill == other.pill
            and self.label == other.label
            and self.predecessors == other.predecessors
            and self.geometry == other.geometry
            and self.output_geometry == other.output_geometry
            and self.is_locked == other.is_locked
            and self.is_output_visible == other.is_output_visible
            and self.requirements == other.requirements
            and self.function_return_type == other.function_return_type
            and self.algorithm == other.algorithm
            and self.code == other.code
            and self.result == other.result
            and self.assertions == other.assertions
            and self.assertion_checks == other.assertion_checks
            and self.assertion_outcomes == other.assertion_outcomes
        )

    def diff(self, other: "Node") -> dict:
        return deepdiff.DeepDiff(self, other)

    def signature_str(self) -> str:
        assert (
            self.function_name is not None
            and self.function_parameters is not None
            and self.function_return_type is not None
        ), "Signature must be generated before it can be printed."

        params = ", ".join([str(p) for p in self.function_parameters])
        return f"{self.function_name}({params}) -> {self.function_return_type.to_python_type()}"

    def code_str(self) -> str:
        assert (
            self.function_name is not None
            and self.function_parameters is not None
            and self.function_return_type is not None
        ), "Signature must be generated before code can be printed."
        assert self.code is not None, "Code must be generated before it can be printed."
        return "\n".join(self.code)

    @classmethod
    def get_merge_schema(cls):
        properties = {
            key: {"mergeStrategy": "overwrite"} for key in cls.model_fields.keys()
        }
        properties["id"] = {"mergeStrategy": "FailOnDifferentValueStrategy"}
        properties["label"] = {"mergeStrategy": "FailOnDifferentValueStrategy"}
        properties["predecessors"] = {"mergeStrategy": "FailOnDifferentValueStrategy"}
        return {"properties": properties}

    def merge(self, other: NodeLike) -> "Node":
        """
        Anything with a subset of the Node representation can be merged.
        """
        schema = self.get_merge_schema()
        merger = jsonmerge.Merger(
            schema,
            strategies={"FailOnDifferentValueStrategy": FailOnDifferentValueStrategy()},
        )
        result = Node.model_validate(
            merger.merge(self.model_dump(), other.model_dump())
        )
        if result == self:
            return self
        else:
            return result

    def update(self, **kwargs) -> "Node":
        assert set(kwargs.keys()).issubset(
            self.model_fields.keys() - {"id", "predecessors"}
        ), f"Invalid kwargs for updating a node: {set(kwargs.keys()).difference(self.model_fields.keys() - {'id', 'predecessors'})}"

        # if code is updated, we need to reset the cache
        code = kwargs.get("code", None)
        if code is not None:
            kwargs["code"] = black_format(code)

        new_node = self.model_copy(update=kwargs)

        if new_node == self:
            return self
        else:
            return new_node

    def lower_phase(self, target_phase: Phase) -> "Node":
        # """
        # Set the phase at target, and remove anything filled in by later phases.
        # """

        # node be already be at phase or lower...
        target_phase = min(target_phase, self.phase)

        messages = [
            message for message in (self.messages or []) if message.phase < target_phase
        ]

        return self.update(phase=target_phase, messages=messages)

    def get_generated_image(self) -> str | None:
        if self.result is not None:
            node_output = self.result.output
            if node_output is not None and node_output.output_type == OutputType.image:
                return node_output.data
        return None

    def warn(self, phase: Phase, message: str) -> Node:
        if self.messages is None:
            self.messages = []
        messages = self.messages + [
            NodeMessage(level="warning", phase=phase, text=message)
        ]
        return self.update(messages=messages)

    def error(self, phase: Phase, message: str):
        if self.messages is None:
            self.messages = []
        messages = self.messages + [
            NodeMessage(level="error", phase=phase, text=message)
        ]
        return self.update(messages=messages)

    def info(self, phase: Phase, message: str):
        if self.messages is None:
            self.messages = []
        messages = self.messages + [
            NodeMessage(level="info", phase=phase, text=message)
        ]
        return self.update(messages=messages)

    def filter_messages(
        self, phase: Phase, level: str | None = None
    ) -> List[NodeMessage]:
        if level is not None:
            return [
                message
                for message in self.messages
                if message.phase == phase and message.level == level
            ]
        else:
            return [message for message in self.messages if message.phase == phase]

    # def ask(self, phase: Phase, assistant: OpenAIAssistant) -> Node:
    #     return self.update(questions=NodeQuestions(phase=phase, assistant=assistant))

    def reset(self, reset_requirements=False) -> Node:
        if reset_requirements:
            node = self.update(requirements=None)
        else:
            node = self
        return node.update(
            phase=Phase.clean,
            cache=BuildCache(),
            function_parameters=None,
            preconditions=None,
            requirements=None,
            function_computed_value=None,
            function_return_type=None,
            function_result_var=pill_to_result_var_name(self.pill),
            function_name=pill_to_function_name(self.pill),
            algorithm=None,
            description=None,
            code=None,
            assertion_checks=None,
            assertion_outcomes=None,
            result=None,
            messages=[],
            build_status=None,
        )

    def to_markdown(self) -> str:

        keys = [
            "pill",
            "label",
            "messages",
            "requirements",
            "description",
            "code",
            "result",
            "assertions",
        ]
        if self.algorithm is not None:
            keys.append("algorithm")
        md = ""
        if "pill" in keys:
            md += f"#### {self.pill}\n\n"
        if "label" in keys:
            md += f"{self.label}\n\n"
        if "messages" in keys and self.messages is not None:
            for level in ["error", "warning", "info"]:
                for message in self.messages:
                    if message.level == level:
                        md += f'\n\n<div markdown="1" class="message {level}">\n{message.message().rstrip()}\n</div>\n\n'
        if "requirements" in keys and self.requirements is not None:
            requirements = "\n".join([f"* {x}" for x in self.requirements])
            md += f"**Requirements**\n\n{requirements}\n\n"
        if "description" in keys and self.description is not None:
            md += f"**Description**\n\n{self.description}\n\n"
        if "function_return_type" in keys and self.function_return_type is not None:
            md += f"**Function Return Type** `{self.function_return_type}`\n\n"
        if "algorithm" in keys and self.algorithm is not None:
            algorithm = "\n".join([f"* {x}" for x in self.algorithm])
            md += f"**Algorithm**\n\n{algorithm}\n\n"
        if "code" in keys and self.code is not None:
            code = "\n".join(self.code)
            md += f"**Code** \n```python\n{code}\n```\n\n"
        if "result" in keys:
            md += f"**Output** \n\n"

            if self.function_return_type is not None:
                md += self.function_return_type.description
                md += "\n\n"

            if self.result is not None:
                clipped = None
                if (
                    self.function_return_type is not None
                    and not self.function_return_type.is_None_type()
                ):
                    text = self.result.pp_result_text(clip=15)
                    if text is not None:
                        clipped = f"<pre>{text}</pre>"

                if not clipped:
                    text = self.result.pp_output_text(clip=15)
                    if text is not None:
                        clipped = f"<pre>{text}</pre>"

                if not clipped:
                    image_url = self.result.output_image()
                    if image_url is not None:
                        base64encoded = image_url.split(",", maxsplit=1)
                        image_data = base64encoded[0] + ";base64," + base64encoded[1]
                        clipped = f"![{self.pill}]({image_data})"

                md += f"{clipped}\n\n"

        if "assertions" in keys and self.assertions is not None:
            assertions = "\n".join([f"* {x}" for x in self.assertions])
            md += f"**Checks**\n\n{assertions}\n\n"

        return md


NodeOrNodeList = Optional[Union[str, List[str]]]


class Edge(BaseModel, frozen=True):
    id: str
    src: str
    dst: str


class GraphLike(BaseModel):
    """
    Any class with a list of nodes that are NodeLike, and also
    any other fields from Graph.
    """

    pass


####


class DataFlowGraph(GraphLike, BaseModel):
    version: int = Field(description="The version of the data flow graph.")
    description: str = Field(
        default="",
        description="A detailed description of what this dataflow graph computes.  Include any assumptions or constraints used when generating the code.",
    )
    nodes: List[Node] = Field(
        default_factory=list,
        description="The nodes in the graph.",
    )
    edges: List[Edge] = Field(
        default_factory=list,
        description="The edges in the graph.",
    )

    image: Optional[str] = Field(
        default=None,
        description="A base64-encoded PNG image representing the data flow graph.",
    )

    def __init__(self, **data):

        # Temporary patch to ensure version is always present in old files
        if "version" not in data:
            data["version"] = 0

        super().__init__(**data)
        self.ensure_valid()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DataFlowGraph):
            return False
        return (
            self.version == other.version
            and self.description == other.description
            and self.nodes == other.nodes
            and self.edges == other.edges
        )

    def semantically_eq(self, other: DataFlowGraph) -> bool:
        if not isinstance(other, DataFlowGraph):
            return False
        return (
            self.description == other.description
            and self.edges == other.edges
            and self.node_ids() == other.node_ids()
            and all(
                self[node_id].semantically_eq(other[node_id])
                for node_id in self.node_ids()
            )
        )

    @classmethod
    def hard_reset(cls, **data: Any) -> "DataFlowGraph":
        with logger("Hard reset"):
            description = data.get("description", "")
            nodes = data.get("nodes", [])
            edges = data.get("edges", [])

            new_nodes = []
            for node in nodes:
                with logger(f"Node {node['id']}..."):
                    new_node = Node(
                        id=node["id"],
                        pill=node["pill"],
                        label=node["label"],
                        geometry=Geometry(**node["geometry"]),
                        output_geometry=Geometry(**node["output_geometry"]),
                        is_locked=node.get("is_locked", False),
                        is_output_visible=node.get("is_output_visible", True),
                        function_name=node["function_name"],
                        function_result_var=node["function_result_var"],
                        predecessors=node["predecessors"],
                        assertions=node.get("assertions", None),
                        phase=Phase.clean,
                        cache=BuildCache(),
                        requirements=node.get("requirements", None),
                        code=node.get("code", None),
                    )

                    new_nodes.append(new_node)

            return cls(version=0, description=description, nodes=new_nodes, edges=edges)

    def ensure_valid(self):
        """
        Check that the graph is valid.
        """
        if not all(
            edge.src in [node.id for node in self.nodes]
            and edge.dst in [node.id for node in self.nodes]
            for edge in self.edges
        ):
            raise ValueError("All edges must be between nodes in the graph")

    def __str__(self) -> str:
        return format_basemodel(self, order=["description", "nodes", "edges"])

    def diff(self, other: "DataFlowGraph") -> dict:
        return deepdiff.DeepDiff(self, other)

    def node_ids(self) -> list[str]:
        return [node.id for node in self.nodes]

    def __getitem__(self, key) -> Node:
        for node in self.nodes:
            if node.id == key:
                return node
        raise KeyError(f"Node with id {key} not found")

    def get_node(self, key) -> Optional[Node]:
        for node in self.nodes:
            if node.id == key:
                return node
        return None

    def get_edge(self, key) -> Optional[Edge]:
        for edge in self.edges:
            if edge.id == key:
                return edge
        return None

    def successors(self, node_id: str) -> List[str]:
        return [edge.dst for edge in self.edges if edge.src == node_id]

    def listify_node_ids(self, node_ids: NodeOrNodeList = None) -> List[str]:
        if node_ids is None:
            node_ids = self.node_ids()
        elif isinstance(node_ids, str):
            node_ids = [node_ids]

        ordered_node_ids = []
        for node in self.topological_sort():
            if node in node_ids:
                ordered_node_ids.append(node)

        return ordered_node_ids

    def with_node(self, node: Node) -> "DataFlowGraph":
        assert node.id in self.node_ids(), f"Node with id {node.id} must already exist"
        old_pill = self[node.id].pill
        new_nodes = [node if node.id == n.id else n for n in self.nodes]
        dfg = self.update(nodes=new_nodes)
        if old_pill != node.pill:
            old_name = pill_to_python_name(old_pill)
            new_name = pill_to_python_name(node.pill)
            dfg = dfg.alpha_rename(old_name, new_name)
            # dfg = dfg.alpha_rename(f"compute_{old_name}", f"compute_{new_name}")
            # dfg = dfg.alpha_rename(f"{old_name}_result", f"{new_name}_result")
        return dfg

    def with_new_node(self, node: Node) -> "DataFlowGraph":
        assert (
            node.id not in self.node_ids()
        ), f"Node with id {node.id} must not already exist"
        new_nodes = self.nodes + [node]
        return self.update(nodes=new_nodes)

    def edge_is_transitively_implied(self, src_id: str, dst_id: str) -> bool:
        """
        Check if there is a path from src to dst.
        """
        if src_id == dst_id:
            return True

        for edge in self.edges:
            if edge.src == src_id and edge.dst == dst_id:
                return True
            if edge.src == src_id:
                if self.edge_is_transitively_implied(edge.dst, dst_id):
                    return True
        return False

    def with_new_edge(self, src_id: str, dst_id: str) -> "DataFlowGraph":
        for edge in self.edges:
            if edge.src == src_id and edge.dst == dst_id:
                return self

        # if dst is successor of src, skip adding
        if self.edge_is_transitively_implied(dst_id, src_id):
            return self

        edge = Edge(id=f"{src_id}->{dst_id}", src=src_id, dst=dst_id)
        new_edges = self.edges + [edge]
        dfg = self.update(edges=new_edges)
        return dfg.lower_phase_with_successors(src_id, Phase.clean)

    def edge_for_nodes(self, src_id: str, dst_id: str) -> Optional[Edge]:
        for edge in self.edges:
            if edge.src == src_id and edge.dst == dst_id:
                return edge
        return None

    def topological_sort(self) -> List[str]:
        try:
            # Create a dictionary to store the indegree of each node
            indegree: Dict[str, int] = {node.id: 0 for node in self.nodes}

            # Create a dictionary to store adjacency list of the graph
            adj_list: Dict[str, List[str]] = {node.id: [] for node in self.nodes}

            # Populate the indegree and adjacency list
            for edge in self.edges:
                adj_list[edge.src].append(edge.dst)
                indegree[edge.dst] += 1

            # Queue to store nodes with indegree 0
            zero_indegree_queue = deque(
                [node.id for node in self.nodes if indegree[node.id] == 0]
            )

            # List to store the topological order
            topo_order = []

            while zero_indegree_queue:
                current_node = zero_indegree_queue.popleft()
                topo_order.append(current_node)

                for neighbor in adj_list[current_node]:
                    indegree[neighbor] -= 1
                    if indegree[neighbor] == 0:
                        zero_indegree_queue.append(neighbor)

            # Check if topological sort is possible (i.e., no cycles)
            if len(topo_order) != len(self.nodes):
                raise ValueError(
                    "Graph has at least one cycle, topological sort not possible"
                )

            return topo_order
        except Exception as e:
            raise e
            # raise FlowcoError(f"Error in topological sort: {e}") from None

    def phases(self) -> Dict[str, Phase]:
        return {node.id: node.phase for node in self.nodes}

    def min_phase(self) -> Phase:
        return min(self.phases().values())

    def node_for_pill(self, pill: str) -> Optional[Node]:
        for node in self.nodes:
            if node.pill == pill:
                return node
        return None

    def make_pill(self, label, exclude_pills: List[str] | None = None) -> str:
        if exclude_pills is None:
            exclude_pills = [x.pill for x in self.nodes]
        prompt = textwrap.dedent(
            f"""
                Summarize the following text with two words:
                ```
                {label}
                ```
                Title-case each word and hyphenate them, eg: `Make-Plot`.
                Do not include any other text in your response.  
                Do not use any of the following: {', '.join(exclude_pills)}.
                """
        )
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=10,
        )

        # Extract the generated text
        pill = completion.choices[0].message.content
        if pill is None:
            raise FlowcoError("No pill generated")
        log(f"'{pill}'")
        return pill

    def update_node_pill(self, node) -> "Node":
        with logger("Generating pill"):
            exclude_pills: List[str] = [x.pill for x in self.nodes if x != node]
            pill = self.make_pill(node.label, exclude_pills=exclude_pills)
            function_name = pill_to_function_name(pill)
            function_result_var = pill_to_result_var_name(pill)

            return node.update(
                pill=pill,
                function_name=function_name,
                function_result_var=function_result_var,
            )

    def alpha_rename(self, from_str: str, to_str: str) -> "DataFlowGraph":
        """
        Replace all occurrences of `from_str` with `to_str` in all string fields of all nodes.

        Args:
            from_str (str): The substring to be replaced.
            to_str (str): The substring to replace with.

        Returns:
            DataFlowGraph: A new DataFlowGraph instance with the replacements made.
        """

        def replace_in_obj(obj):
            if obj is None:
                return None
            if isinstance(obj, str):
                return obj.replace(from_str, to_str)
            elif isinstance(obj, BaseModel):
                if isinstance(obj, Edge):
                    return obj
                elif isinstance(obj, Node):
                    # Recursively replace in Pydantic models
                    return obj.model_copy(
                        update={
                            field: replace_in_obj(getattr(obj, field))
                            for field in obj.model_fields.keys()
                            if field not in ["id", "predecessors"]
                            and getattr(obj, field) is not None
                        }
                    )
                else:
                    # Recursively replace in Pydantic models
                    return obj.model_copy(
                        update={
                            field: replace_in_obj(getattr(obj, field))
                            for field in obj.model_fields.keys()
                            if getattr(obj, field) is not None
                        }
                    )

            elif isinstance(obj, list):
                return [replace_in_obj(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: replace_in_obj(value) for key, value in obj.items()}
            else:
                return obj  # For other types, return as is

        new_nodes = []
        for node in self.nodes:
            updated_node = replace_in_obj(node)
            new_nodes.append(updated_node)

        return self.update(nodes=new_nodes)

    def subgraph_of_predecessors(self, node_id: str) -> "DataFlowGraph":
        """
        Return a subgraph of the graph that includes the node and all its predecessors.
        """
        node_ids = self[node_id].predecessors + [node_id]
        return DataFlowGraph(
            description=self.description,
            nodes=[node for node in self.nodes if node.id in node_ids],
            edges=[
                edge
                for edge in self.edges
                if edge.src in node_ids and edge.dst in node_ids
            ],
        )

    def get_merge_schema(self):
        return {
            "properties": {
                "nodes": {
                    "mergeStrategy": "arrayMergeById",
                    "mergeOptions": {"idRef": "id"},
                }
                | Node.get_merge_schema(),
                "edges": {"mergeStrategy": "FailOnDifferentValueStrategy"},
                "description": {"mergeStrategy": "overwrite"},
            }
        }

    def merge(self, other: GraphLike) -> "DataFlowGraph":
        """
        Anything with a subset of the Graph representation can be merged.
        """
        # Careful - graphlikes may not have all entries...
        if hasattr(other, "nodes") and not set(
            [node.id for node in other.nodes]  # type: ignore
        ).issubset(
            set(self.node_ids())
        ):  # type: ignore
            raise ValueError(f"Other graph must have a subset of the node ids to merge: {other.nodes} not in {self.node_ids()}")  # type: ignore
        if hasattr(other, "edges") and not set(other.edges).issubset(set(self.edges)):  # type: ignore
            raise ValueError(
                f"Other graph must have a subset of edges as the old graph: {other.edges} not in {self.edges}"  # type: ignore
            )

        schema = self.get_merge_schema()
        merger = jsonmerge.Merger(
            schema,
            strategies={"FailOnDifferentValueStrategy": FailOnDifferentValueStrategy()},
        )
        result = DataFlowGraph.model_validate(
            merger.merge(self.model_dump(), other.model_dump())
        )

        if result == self:
            return self
        else:
            return result

    def update(self, **kwargs) -> "DataFlowGraph":
        new_graph = self.model_copy(update=kwargs)

        # test if the visual image for new_graph would be different: different nodes or edges, nodes have moved, etc.
        if (
            set(self.node_ids()) != set(new_graph.node_ids())
            or self.edges != new_graph.edges
            or any(self[n].label != new_graph[n].label for n in self.node_ids())
            or any(self[n].geometry != new_graph[n].geometry for n in self.node_ids())
        ):
            new_graph.image = None

        if new_graph == self:
            return self
        else:
            new_graph.version = self.version + 1
            return new_graph

    def update_node(self, node_id: str, **kwargs) -> "DataFlowGraph":
        """
        Update a node in the graph.
        """
        node = self[node_id].update(**kwargs)
        return self.with_node(node)

    def with_phase(self, node_ids: NodeOrNodeList, phase: Phase) -> "DataFlowGraph":
        node_ids = self.listify_node_ids(node_ids)
        new_nodes = [
            node.update(phase=phase) if node.id in node_ids else node
            for node in self.nodes
        ]
        return self.update(nodes=new_nodes)

    def successors(self, node_id: str) -> Set[str]:
        direct = {edge.dst for edge in self.edges if edge.src == node_id}
        for node in self.nodes:
            if node.id in direct:
                direct |= self.successors(node.id)
        return direct

    def lower_phase_with_successors(
        self, node_ids: NodeOrNodeList, target_phase: Phase
    ) -> "DataFlowGraph":
        to_change = self.listify_node_ids(node_ids)

        # Could compute successors directly too...
        # to_change = set(to_change) | {
        #     x
        #     for x in self.node_ids()
        #     for node_id in to_change
        #     if node_id in self[x].predecessors
        # }

        succs = set(to_change)
        for node_id in to_change:
            succs |= self.successors(node_id)

        new_nodes = [
            (self[node].lower_phase(target_phase) if node in succs else self[node])
            for node in self.node_ids()
        ]

        kwargs: Dict[str, Any] = {"nodes": new_nodes}
        return self.update(**kwargs)

    def reduce_phases_to_below_target(
        self, node_ids: NodeOrNodeList, target_phase: Phase | None
    ) -> "DataFlowGraph":
        new_graph = self.lower_phase_with_successors(
            node_ids,
            Phase(target_phase - 1) if target_phase is not None else Phase.clean,
        )
        return new_graph

    def invalidate_build_cache(self, node_ids: NodeOrNodeList) -> "DataFlowGraph":
        with logger(f"invalidate_build_cache {node_ids}"):
            node_ids = self.listify_node_ids(node_ids)
            new_nodes = [
                (node.update(cache=BuildCache()) if node.id in node_ids else node)
                for node in self.nodes
            ]
            return self.update(nodes=new_nodes)

    def reset(self, reset_requirements=False) -> "DataFlowGraph":
        new_nodes = [node.reset(reset_requirements) for node in self.nodes]
        return self.update(nodes=new_nodes, version=self.version + 1)

    def make_driver(self):

        driver = []
        for node_id in self.topological_sort():
            node = self.get_node(node_id)
            assert node, f"Node {node_id} not found in graph"
            assert node.function_name, f"Node {node_id} has no function name"
            preds = [self.get_node(x) for x in node.predecessors]
            assert all(preds), f"Node {node_id} has missing predecessors"
            driver += [
                FunctionCall(
                    node_id=node.id,
                    function_name=node.function_name,
                    arguments=[f"{x.function_result_var}" for x in preds],
                    result=node.function_result_var,
                )
            ]
        return driver

    def to_image_prompt_messages(self) -> List[Dict[str, Any]]:

        if not config.x_no_dfg_image_in_prompt:
            if self.image is None:
                with logger("Generating image"):
                    self.image = dataflow_graph_to_image(self)

            if self.image is not None:
                return [
                    {
                        "type": "text",
                        "text": f"Here is the dataflow diagram.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{self.image}",
                            "detail": "high",
                        },
                    },
                ]

        return []

    def outputs_to_prompt_messages(dfg: DataFlowGraph) -> List[Dict[str, Any]]:
        messages = []
        for node_id in dfg.topological_sort():
            node = dfg[node_id]
            result = node.result
            if result is not None:
                if result.result is not None:

                    # guard against the result value being something we can't handle
                    try:
                        repr, clipped = result.result.to_repr(10)
                    except Exception as e:
                        error(e)
                        repr, clipped = str(result.result), False

                    if clipped:
                        messages.append(
                            {
                                "type": "text",
                                "text": f"The variable `{node.function_result_var}` has this value.  We show only the first 10 elements of any aggregate value:\n```\n{repr}\n```",
                            }
                        )
                    else:
                        messages.append(
                            {
                                "type": "text",
                                "text": f"The variable `{node.function_result_var}` has this value:\n```\n{repr}\n```",
                            }
                        )

                if result.output is not None:
                    messages.append(
                        {
                            "type": "text",
                            "text": f"Here is the output for {node_id}.",
                        }
                    )
                    output = result.output
                    if output.output_type == OutputType.text:
                        messages.append({"type": "text", "text": output.data})
                    elif output.output_type == OutputType.image:
                        messages.append(
                            {
                                "type": "text",
                                "text": f"You may include the following image in your report with this: `![node_output]({node_id}.png)`",
                            }
                        )
                        messages.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": output.data.replace(
                                        "data:image/png,", "data:image/png;base64,"
                                    ),
                                    "detail": "high",
                                },
                            }
                        )
                    else:
                        messages.append(
                            {
                                "type": "text",
                                "text": "Unknown output type: {self.output_type}: {self.data}",
                            }
                        )
        return messages

    def replace_placeholders_with_base64_images(
        dfg: DataFlowGraph, markdown: str
    ) -> str:
        for node_id in dfg.topological_sort():
            image = dfg[node_id].get_generated_image()
            if image is not None:
                image = image.replace("data:image/png,", "data:image/png;base64,")
                markdown = markdown.replace(
                    f"![node_output]({node_id}.png)", f'\n\n<img src="{image}"/>\n\n'
                )
        return markdown

    def normalize_ids_to_pills(self) -> "DataFlowGraph":
        """
        Returns a new DataFlowGraph instance with each node's `id` renamed to its `pill`.
        All references (predecessors, edges) are updated accordingly.

        Returns:
            DataFlowGraph: The updated data flow graph with renamed IDs.

        Raises:
            ValueError: If there are duplicate pills which would result in duplicate IDs.
        """
        # Step 1: Create a mapping from old id to pill
        id_to_pill: Dict[str, str] = {
            node.id: pill_to_python_name(node.pill) for node in self.nodes
        }

        # Step 2: Ensure that all pills are unique to prevent ID conflicts
        pills = list(id_to_pill.values())
        if len(pills) != len(set(pills)):
            duplicates = set([pill for pill in pills if pills.count(pill) > 1])
            raise ValueError(
                f"Pills must be unique to rename IDs. Duplicate pills found: {duplicates}"
            )

        # Step 3: Create new nodes with updated ids
        new_nodes = []
        for node in self.nodes:
            new_node = node.model_copy()
            new_node.id = id_to_pill[node.id]
            new_node.pill = id_to_pill[node.id]
            # Update predecessors in the new node
            updated_predecessors = [
                id_to_pill[pred_id] for pred_id in node.predecessors
            ]
            new_node.predecessors = updated_predecessors
            new_nodes.append(new_node)

        # Step 4: Create new edges with updated src and dst
        new_edges = []
        for edge in self.edges:
            if edge.src not in id_to_pill or edge.dst not in id_to_pill:
                raise ValueError(
                    f"Edge references undefined node IDs: src='{edge.src}', dst='{edge.dst}'."
                )
            new_edge = Edge(
                id=f"{id_to_pill[edge.src]}->{id_to_pill[edge.dst]}",
                src=id_to_pill[edge.src],
                dst=id_to_pill[edge.dst],
            )
            new_edges.append(new_edge)

        # Step 5: Construct and return the new DataFlowGraph instance
        return self.model_copy(
            update={
                "nodes": new_nodes,
                "edges": new_edges,
            }
        )

    def to_markdown(self) -> str:
        """
        Generates a Markdown representation of the data flow graph, including nodes, edges, and results.
        """
        md = ""
        md += f"**Version:** {self.version}\n\n"

        with logger("Generating image"):
            self.image = dataflow_graph_to_image(self)

        if self.image is not None:
            md += f"![Data Flow Graph Image](data:image/png;base64,{self.image})\n\n"

        md += f"**Description:**\n\n{self.description}\n\n"

        # Nodes Section
        for node in self.nodes:
            md += node.to_markdown()

        return md


def dataflow_graph_to_image(dfg: DataFlowGraph) -> str:
    """
    Convert a DataFlowGraph instance into a base64-encoded PNG image.
    """
    dot = Digraph(format="png")

    # Node styles
    node_style = {
        "shape": "rect",
        "style": "filled",
        "fontcolor": "brown",
        "fillcolor": "lemonchiffon",
        "fontsize": "10",
        "penwidth": "1",
    }

    # Add nodes
    for node in dfg.nodes:
        label = "<br/>".join(textwrap.wrap(node.label, width=50))
        label = f"<<b>{node.pill}:</b><br/>{label}>"
        dot.node(node.id, label=label, **node_style)

    # Edge styles
    edge_style = {
        "color": "darkblue",
        "fontsize": "10",
        "fontcolor": "darkblue",
        "penwidth": "1",
    }

    # Add edges
    for edge in dfg.edges:
        dot.edge(
            edge.src,
            edge.dst,
            label="",  # f" {dfg[edge.src].function_result_var}",
            **edge_style,
        )

    temp_files = []
    # Create a temporary directory to store image files
    # with tempfile.TemporaryDirectory() as temp_dir:
    for node in dfg.nodes:
        # Check if the node has no incoming edges
        if node.result is not None:
            if (
                node.function_return_type is not None
                and not node.function_return_type.is_None_type()
            ):
                if (text := node.result.pp_result_text(clip=15)) is not None:
                    dot.node(
                        node.id + "-output",
                        label=f"{text}",
                        shape="none",
                        fontsize="8pt",
                    )
            elif (text := node.result.pp_output_text(clip=15)) is not None:
                dot.node(
                    node.id + "-output",
                    label=f"{text}",
                    shape="none",
                    fontsize="8pt",
                )
            elif (image_url := node.result.output_image()) is not None:
                base64encoded = image_url.split(",", maxsplit=1)[1]
                image_data = base64.b64decode(base64encoded)

                image_filename = f"{node.id}-output.png"
                image_path = image_filename  # os.path.join(temp_dir, image_filename)

                log(f"Saving image to {image_path}")
                with open(image_path, "wb") as image_file:
                    image_file.write(image_data)
                temp_files.append(image_path)

                dot.node(
                    node.id + "-output",
                    label="",
                    image=image_path,
                    width="3",
                    height="3",
                    imagescale="true",
                    fixedsize="true",
                    shape="none",
                )

            with dot.subgraph() as s:
                s.attr(rank="same")  # Ensure nodes are on the same horizontal level
                s.node(node.id)
                s.node(node.id + "-output")

            edge_style = {"style": "solid"}  # Example edge style
            dot.edge(
                node.id,
                node.id + "-output",
                **edge_style,
                color="red",  # Example edge color
            )

    png_data = dot.pipe(format="png")
    base64_image = base64.b64encode(png_data).decode("utf-8")

    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except Exception as e:
            error(f"Error deleting temporary file", e)

    return base64_image

import difflib
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from flowco.builder.build import BuildEngine
from flowco.builder.pass_config import PassConfig
from flowco.dataflow.dfg import DataFlowGraph, Edge, Geometry, Node
from flowco.dataflow.phase import Phase
from flowco.util.config import AbstractionLevel

from pydantic import BaseModel, Field

from flowco.util.errors import FlowcoError
from flowco.util.output import logger, message
from flowco.util.text import pill_to_python_name
from flowco.util.yes_no import YesNoPrompt


class FlowthonNode(BaseModel):
    pill: str
    uses: List[str]
    label: str
    requirements: List[str] | None
    algorithm: Optional[List[str]] | None
    code: Optional[List[str]] | None

    def to_json(self, level: AbstractionLevel) -> Dict[str, Any]:
        map = {
            "uses": self.uses,
            "label": self.label,
        }
        if self.requirements:
            map["requirements"] = self.requirements
        if self.algorithm and level in [
            AbstractionLevel.algorithm,
            AbstractionLevel.code,
        ]:
            map["algorithm"] = self.algorithm
        if self.code and level in [AbstractionLevel.code]:
            map["code"] = self.code
        return map

    @classmethod
    def from_json(cls, pill: str, node_data: dict) -> "FlowthonNode":
        assert isinstance(pill, str), f"Expected str, got {type(pill)}"
        assert isinstance(node_data, dict), f"Expected dict, got {type(node_data)}"
        assert all(
            key in ["uses", "label", "requirements", "algorithm", "code"]
            for key in node_data
        ), f"Missing keys in {node_data}"

        # assert that uses is a list of strings and that all strings are valid pills in the nodes
        assert isinstance(
            node_data["uses"], list
        ), f"Expected list, got {type(node_data['uses'])}"
        assert all(
            isinstance(x, str) for x in node_data["uses"]
        ), f"Expected list of str, got {node_data['uses']}"

        # assert label is a string
        assert isinstance(
            node_data["label"], str
        ), f"Expected str, got {type(node_data['label'])}"

        if "requirements" in node_data:
            assert isinstance(
                node_data["requirements"], list
            ), f"Expected list, got {type(node_data['requirements'])}"
            assert all(
                isinstance(x, str) for x in node_data["requirements"]
            ), f"Expected list of str, got {node_data['requirements']}"

        if "algorithm" in node_data:
            assert isinstance(
                node_data["algorithm"], list
            ), f"Expected list, got {type(node_data['algorithm'])}"
            assert all(
                isinstance(x, str) for x in node_data["algorithm"]
            ), f"Expected list of str, got {node_data['algorithm']}"

        if "code" in node_data:
            assert isinstance(
                node_data["code"], list
            ), f"Expected str, got {type(node_data['code'])}"
            assert all(
                isinstance(x, str) for x in node_data["code"]
            ), f"Expected list of str, got {node_data['code']}"
            node_data["code"] = node_data["code"]

        return cls(
            pill=pill,
            uses=node_data.get("uses", []),
            label=node_data.get("label", ""),
            requirements=node_data.get("requirements", None),
            algorithm=node_data.get("algorithm", None),
            code=node_data.get("code", None),
        )


class FlowthonGraph(BaseModel):
    nodes: Dict[str, FlowthonNode] = Field(
        description="List of nodes in the graph.",
    )

    def __init__(self, **data):
        super().__init__(**data)

        # verify that all uses in all nodes are valid pills in the nodes
        for node in self.nodes.values():
            for use in node.uses:
                assert use in self.nodes

    @classmethod
    def from_dfg(cls, dfg: DataFlowGraph) -> Tuple[DataFlowGraph, "FlowthonGraph"]:

        dfg = dfg.normalize_ids_to_pills()

        # assert all nodes had ids matching pills
        for node in dfg.nodes:
            assert node.id == node.pill, f"{node.id} != {node.pill}"

        # assert all edges had ids matching pills
        for edge in dfg.edges:
            assert edge.src in dfg.node_ids(), f"{edge.src} not in {dfg.node_ids()}"
            assert edge.dst in dfg.node_ids(), f"{edge.dst} not in {dfg.node_ids()}"
            assert (
                edge.id == f"{edge.src}->{edge.dst}"
            ), f"{edge.id} != {edge.src}-->{edge.dst}"

        nodes = {}
        for node_id in dfg.topological_sort():
            node = dfg[node_id]
            predecessors = [dfg[n].pill for n in node.predecessors]
            nodes[node.pill] = FlowthonNode(
                pill=node.pill,
                uses=predecessors,
                label=node.label,
                requirements=node.requirements,
                algorithm=node.algorithm,
                code=node.code,
            )
        return dfg, cls(nodes=nodes)

    def update(self, *nodes) -> "FlowthonGraph":
        """
        Functional update
        """
        updated = self.model_copy(
            update={"nodes": self.nodes | {node.id: node for node in nodes}}
        )
        if updated == self:
            return self
        else:
            return updated

    def merge(
        self,
        pass_config: PassConfig,
        dfg: DataFlowGraph | None = None,
        interactive=False,
    ) -> DataFlowGraph:
        """
        Merge the editable graph into the DataFlowGraph:
        - Add new nodes
        - Update existing nodes
        - Remove nodes that are not in the editable graph
        - Update edges
        """
        if dfg is None:
            dfg = DataFlowGraph(nodes=[], edges=[], version=0)

        new_nodes = []
        new_edges = []

        for node in self.nodes.values():
            if node.pill not in dfg.node_ids():
                python_name = pill_to_python_name(node.pill)
                new_node = Node(
                    id=node.pill,
                    pill=node.pill,
                    label=node.label,
                    predecessors=[],
                    geometry=Geometry(x=0, y=0, width=0, height=0),
                    output_geometry=Geometry(x=0, y=0, width=0, height=0),
                    function_name=f"compute_{python_name}",
                    function_result_var=f"result_{python_name}",
                    requirements=node.requirements,
                    algorithm=node.algorithm,
                    code=node.code,
                )

                message(f"Adding new node {new_node.pill}")

            else:
                original = dfg[node.pill]
                new_node = original.model_copy(
                    update={
                        "predecessors": [],
                        "phase": Phase.clean,
                    }
                )

                new_node = new_node.update(
                    label=node.label,
                    requirements=(
                        node.requirements
                        if node.requirements is not None
                        else new_node.requirements
                    ),
                    algorithm=(
                        node.algorithm
                        if node.algorithm is not None
                        else new_node.algorithm
                    ),
                    code=node.code if node.code is not None else new_node.code,
                )

                edits = []
                if node.label != original.label:
                    edits.append("label")
                if node.requirements and node.requirements != original.requirements:
                    edits.append("requirements")
                if node.algorithm and node.algorithm != original.algorithm:
                    edits.append("algorithm")
                if node.code and node.code != original.code:
                    edits.append("code")

                if edits:
                    message(
                        f"Modifying existing node {new_node.pill}: {', '.join(edits)}"
                    )

            new_nodes.append(new_node)
            for pred in node.uses:
                new_edges.append(
                    Edge(id=f"{pred}->{node.pill}", src=pred, dst=node.pill)
                )

        new_graph = DataFlowGraph(
            description=dfg.description,
            nodes=new_nodes,
            edges=new_edges,
            version=dfg.version + 1,
        )

        def predecessors(node: Node) -> List[str]:
            direct_preds = {edge.src for edge in new_graph.edges if edge.dst == node.id}
            indirect_preds = {
                p for pred in direct_preds for p in predecessors(new_graph[pred])
            }
            preds = list(direct_preds | indirect_preds)
            preds = sorted(preds, key=lambda x: new_graph[x].pill)
            return preds

        for node_id in new_graph.topological_sort():
            new_graph[node_id].predecessors = predecessors(new_graph[node_id])
            if (
                node_id in dfg
                and new_graph[node_id].predecessors != dfg[node_id].predecessors
            ):
                message(
                    f"Modifying existing node {new_graph[node_id].pill}: predecessors"
                )

        new_graph = self.rebuild(pass_config, dfg, new_graph, interactive)

        return new_graph

    def rebuild(
        self,
        pass_config: PassConfig,
        original: DataFlowGraph,
        dfg: DataFlowGraph,
        interactive,
    ) -> DataFlowGraph:

        with logger("Making build engine"):
            engine = BuildEngine.get_builder()

            attributes = {
                Phase.requirements: "requirements",
                Phase.algorithm: "algorithm",
                Phase.code: "code",
            }

            yes_no_prompt = YesNoPrompt(not interactive)

            with logger("Building"):
                for build_updated in engine.build_with_worklist(
                    pass_config, dfg, Phase.run_checked, None
                ):
                    new_dfg = build_updated.new_graph
                    updated_node = build_updated.updated_node

                    prefix = f"[{str(updated_node.phase)}]"
                    prefix = f"{prefix:<14} {updated_node.pill}"
                    prefix = f" {prefix:.<55}"

                    if updated_node.phase in attributes:
                        key = attributes[updated_node.phase]
                        edited = getattr(self.nodes[updated_node.id], key)
                        new = getattr(updated_node, key)

                        if updated_node.id in original.node_ids():
                            old = getattr(original[updated_node.id], key)
                        else:
                            old = None

                        if old == new:
                            message(f"{prefix} unchanged")
                        elif edited is None:
                            message(f"{prefix} generated")
                        elif edited == new:
                            message(f"{prefix} taken from user")
                        else:
                            message(f"{prefix} taken from user and modified")
                            diff = "\n".join(difflib.ndiff(edited, new))
                            print(textwrap.indent(diff, "    "))
                            print()
                            if yes_no_prompt.ask("Accept this change"):
                                setattr(self.nodes[updated_node.id], key, new)
                                print()
                            else:
                                print()
                                raise FlowcoError("User rejected changes")

                    else:
                        message(f"{prefix} done")

            return new_dfg

    def to_json(self, level: AbstractionLevel) -> Dict[str, Any]:
        return {pill: node.to_json(level) for pill, node in self.nodes.items()}

    @classmethod
    def from_json(cls, data: dict) -> "FlowthonGraph":
        return cls(
            nodes={
                pill: FlowthonNode.from_json(pill, node_data)
                for pill, node_data in data.items()
            }
        )

from dataclasses import dataclass
import importlib
import inspect
import pprint
import queue
from threading import current_thread
import traceback
from flowco.builder.pass_config import PassConfig
from flowco.dataflow.phase import Phase
from flowco.session.session import session
from flowco.util.output import error, log, warn, logger
from flowco.util.stopper import Stopper
from typing import (
    Callable,
    Dict,
    Iterator,
    List,
)
import concurrent.futures

from pydantic import BaseModel
from flowco.dataflow.dfg import (
    DataFlowGraph,
    Node,
)
from flowco.util.text import function_name_to_title
from flowco.util.output import log, warn, error, buffer_output

from flowco.util.config import config


class BuildUpdate(BaseModel):
    steps_remaining: int
    steps_total: int
    new_graph: DataFlowGraph
    updated_node: Node | None = None


PassFunction = Callable[[PassConfig, DataFlowGraph, Node], Node]


class WorkItem(BaseModel):
    node_id: str
    target_phase: Phase

    class Config:
        frozen = True
        use_enum_values = True

    def __hash__(self):
        return hash((self.node_id, self.target_phase))

    def __str__(self):
        return f"{self.node_id}@{Phase(self.target_phase)}"

    def __repr__(self):
        return str(self)


phase_to_message = {
    Phase.clean: "Cleaning",
    Phase.requirements: "Making requirements",
    Phase.algorithm: "Making algorithm",
    Phase.code: "Making code",
    Phase.runnable: "Making runnable",
    Phase.run_checked: "Running",
    Phase.assertions_code: "Making checks",
    Phase.assertions_checked: "Checking",
}


class Pass:
    def __init__(self, pass_function: PassFunction):
        self.pass_function: PassFunction = pass_function
        assert hasattr(
            pass_function, "_is_node_pass"
        ), "Pass must be a node or page pass."

    def title(self) -> str:
        return function_name_to_title(self.pass_function.__name__)

    def full_name(self) -> str:
        return f"{self.pass_function.__module__}.{self.pass_function.__qualname__}"

    def required_phase(self) -> Phase:
        return self.pass_function._required_phase  # type: ignore

    def target_phase(self) -> Phase:
        return self.pass_function._target_phase  # type: ignore


class PassFailedException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class BuildEngine:

    def __init__(self: "BuildEngine"):
        self.passes_by_target: Dict[Phase, Pass] = {}
        self.passes_by_required: Dict[Phase, Pass] = {}

    def add_pass(self, callable: PassFunction):
        p = Pass(callable)
        assert (
            p.required_phase() not in self.passes_by_required
        ), f"Pass already exists for this required phase: {p.required_phase()} for {p.full_name()}."

        assert (
            p.target_phase() not in self.passes_by_target
        ), f"Pass already exists for this target: {p.target_phase()} for {p.full_name()}."

        self.passes_by_required[p.required_phase()] = p
        self.passes_by_target[p.target_phase()] = p

        # if we want to reach a target phase with no pass,
        # we'll use pass that can reach it by overshooting
        # by as little as possible
        for phase in range(Phase.clean, p.required_phase()):
            if (
                phase not in self.passes_by_target
                or p.target_phase() < self.passes_by_target[phase].target_phase()
            ):
                self.passes_by_target[phase] = p

    def __str__(self):
        return "\n".join(
            [
                f"""{str(k):>17} -> {str(v.target_phase()):<17}  {"'"+v.title()+"'":<20}  {v.full_name()}"""
                for k, v in sorted(self.passes_by_required.items(), key=lambda x: x[0])
            ]
        )

    def make_worklist(
        self,
        graph: DataFlowGraph,
        target_phase: Phase,
        node_ids: List[str] | str | None,
    ) -> Dict[WorkItem, List[WorkItem]]:
        preds: Dict[WorkItem, List[WorkItem]] = {}

        def add_work_item(work_item: WorkItem):
            if work_item in preds:
                log(f"{work_item} already done")
                return

            if graph[work_item.node_id].phase >= work_item.target_phase:
                log(f"{work_item} already at phase")
                return

            preds[work_item] = []
            with logger(f"Adding {work_item}"):
                node_id = work_item.node_id
                target_phase = work_item.target_phase

                if target_phase != Phase.clean:
                    with logger(
                        f"Looking for pass to reach target {str(target_phase)}"
                    ):
                        pass_to_reach_target = self.passes_by_target[target_phase]
                        required_phase = pass_to_reach_target.required_phase()

                    immediate_pred = WorkItem(
                        node_id=node_id, target_phase=required_phase
                    )
                    if graph[node_id].phase < required_phase:
                        preds[work_item].append(immediate_pred)
                        add_work_item(immediate_pred)

                    pred_required_phase = (
                        pass_to_reach_target.pass_function._pred_required_phase
                    )
                    if pred_required_phase is not None:
                        for pred in graph[node_id].predecessors:
                            if graph[pred].phase < pred_required_phase:
                                pred_item = WorkItem(
                                    node_id=pred, target_phase=pred_required_phase
                                )
                                preds[work_item].append(pred_item)
                                add_work_item(pred_item)

        node_ids = graph.listify_node_ids(node_ids)
        log("nodes", node_ids)
        log("target_phase", target_phase)
        with logger("making worklist"):
            for node_id in node_ids:
                add_work_item(WorkItem(node_id=node_id, target_phase=target_phase))

        return preds

    def build_with_worklist(
        self,
        pass_config: PassConfig,
        graph: DataFlowGraph,
        target_phase: Phase,
        node_ids: List[str] | str | None,
    ) -> Iterator[BuildUpdate]:

        graph = graph.update(nodes=[node.update(messages=[]) for node in graph.nodes])

        if config.sequential:
            yield from self.build_with_worklist_sequential(
                pass_config, graph, target_phase, node_ids
            )
        else:
            yield from self.build_with_worklist_parallel(
                pass_config, graph, target_phase, node_ids
            )

    def build_with_worklist_sequential(
        self,
        pass_config: PassConfig,
        graph: DataFlowGraph,
        target_phase: Phase,
        node_ids: List[str] | str | None,
    ) -> Iterator[BuildUpdate]:

        @dataclass
        class NodeResult:
            item: WorkItem
            node: Node
            new_node: Node
            buffer: str

        worklist = self.make_worklist(graph, target_phase, node_ids)

        with logger(f"Building: {len(worklist.keys())} steps"):

            def process_workitem(graph: DataFlowGraph, work_item: WorkItem):
                node_id = work_item.node_id
                node = graph[node_id]

                with logger(f"{node_id}:{node.pill}"):
                    if session.get("stopper", Stopper).should_stop():
                        return NodeResult(work_item, node, node, "Stopped")

                    node.build_status = phase_to_message.get(
                        work_item.target_phase, "???"
                    )

                    target_phase = work_item.target_phase
                    if node.phase >= target_phase:
                        return NodeResult(
                            work_item,
                            node,
                            node,
                            f"Target phase is {target_phase} but node is already at {node.phase}",
                        )
                    else:
                        log(
                            f"Node {node_id} at phase {node.phase}, target is {Phase(target_phase)}"
                        )

                    pass_to_reach_target = self.passes_by_target[target_phase]
                    required_phase = pass_to_reach_target.required_phase()

                    # check requirements
                    if node.phase < required_phase:
                        return NodeResult(
                            work_item,
                            node,
                            node,
                            f"Node {node_id} must be at {required_phase} to reach {Phase(target_phase)}",
                        )
                    else:
                        log(f"Node {node_id} at required phase {required_phase}")

                    pred_required_phase = (
                        pass_to_reach_target.pass_function._pred_required_phase
                    )
                    if pred_required_phase is not None:
                        for pred in graph[node_id].predecessors:
                            if graph[pred].phase < pred_required_phase:
                                return NodeResult(
                                    work_item,
                                    node,
                                    node,
                                    f"Node {pred} must be at {pred_required_phase} to build {node_id}",
                                )

                    with logger(f"{node.pill}: {pass_to_reach_target.title()}"):
                        new_node = pass_to_reach_target.pass_function(
                            pass_config, graph, node
                        )
                    assert isinstance(
                        new_node, Node
                    ), f"Node pass {pass_to_reach_target.title()} must return a Node object."
                    if new_node.phase < target_phase:
                        warn(
                            f"Pass {pass_to_reach_target.title()} did not update phase for node {node.pill} to {target_phase}"
                        )
                        node.error(
                            phase=target_phase,
                            message=f"{pass_to_reach_target.title()} failed on `{node.pill}`",
                        )
                return NodeResult(work_item, node, new_node, "Done")

            new_graph = graph
            done = set()
            in_flight = []

            def submit_items(current_worklist: Dict[WorkItem, List[WorkItem]]):
                ready = [
                    item
                    for item in current_worklist
                    if len(current_worklist[item]) == 0
                ]
                log(f"Submitting {ready}")
                current_worklist = {
                    k: v for k, v in current_worklist.items() if k not in ready
                }
                for work_item in ready:
                    in_flight.append(work_item)
                return current_worklist

            log("workllist:", pprint.pformat(worklist, sort_dicts=True, indent=2))
            current_worklist = submit_items(worklist)

            with logger("Running worklist"):
                stopper = session.get("stopper", Stopper)
                while len(done) < len(worklist) and not stopper.should_stop():
                    try:
                        log(f"Building: {len(worklist.keys()) - len(done)} steps")
                        item = in_flight.pop(0)
                        result = process_workitem(new_graph, item)
                        done |= {item}
                        with logger(f"{item} Complete"):
                            log(result.buffer)
                            new_node = result.new_node
                            new_node.build_status = None
                            if new_node.phase < item.target_phase:
                                warn(
                                    f"Pass did not update phase for node {new_node.pill} to {target_phase}"
                                )

                            new_graph = new_graph.with_node(new_node)
                            current_worklist = {
                                k: [x for x in v if x != item]
                                for k, v in current_worklist.items()
                            }
                            current_worklist = submit_items(current_worklist)
                            b = BuildUpdate(
                                steps_remaining=len(worklist.keys()) - len(done),
                                steps_total=len(worklist.keys()),
                                new_graph=new_graph,
                                updated_node=new_node,
                            )
                            yield b
                    except Exception as e:
                        error(e)
                        raise e

                new_graph = new_graph.update(
                    nodes=[x.update(build_status=None) for x in new_graph.nodes]
                )
                yield BuildUpdate(
                    steps_remaining=len(worklist.keys()) - len(done),
                    steps_total=len(worklist.keys()),
                    new_graph=new_graph,
                    updated_node=None,
                )
                log("Worklist done")

    def build_with_worklist_parallel(
        self,
        pass_config: PassConfig,
        graph: DataFlowGraph,
        target_phase: Phase,
        node_ids: List[str] | str | None,
    ) -> Iterator[BuildUpdate]:

        @dataclass
        class NodeResult:
            item: WorkItem
            node: Node
            new_node: Node
            buffer: str

        import streamlit as st

        st.session_state.borp = 3

        worklist = self.make_worklist(graph, target_phase, node_ids)
        result_queue = queue.Queue[NodeResult]()

        with logger(f"Building: {len(worklist.keys())} steps"):

            def process_workitem(graph: DataFlowGraph, work_item: WorkItem):
                node_id = work_item.node_id
                node = graph[node_id]

                with buffer_output(f"{node.pill}") as buffer:
                    with logger(f"Starting"):
                        if session.get("stopper", Stopper).should_stop():
                            result_queue.put(
                                NodeResult(work_item, node, node, "Stopped")
                            )
                            return

                        node.build_status = phase_to_message.get(
                            work_item.target_phase, "???"
                        )

                        target_phase = work_item.target_phase
                        if node.phase >= target_phase:
                            result_queue.put(
                                NodeResult(
                                    work_item,
                                    node,
                                    node,
                                    f"Target phase is {target_phase} but node is already at {node.phase}",
                                )
                            )
                            return
                        else:
                            log(
                                f"Node {node_id} at phase {node.phase}, target is {Phase(target_phase)}"
                            )

                        pass_to_reach_target = self.passes_by_target[target_phase]
                        required_phase = pass_to_reach_target.required_phase()

                        # check requirements
                        if node.phase < required_phase:
                            result_queue.put(
                                NodeResult(
                                    work_item,
                                    node,
                                    node,
                                    f"Node {node_id} must be at {required_phase} to reach {Phase(target_phase)}",
                                )
                            )
                            return
                        else:
                            log(f"Node {node_id} at required phase {required_phase}")

                        pred_required_phase = (
                            pass_to_reach_target.pass_function._pred_required_phase
                        )
                        if pred_required_phase is not None:
                            for pred in graph[node_id].predecessors:
                                if graph[pred].phase < pred_required_phase:
                                    result_queue.put(
                                        NodeResult(
                                            work_item,
                                            node,
                                            node,
                                            f"Node {pred} must be at {pred_required_phase} to build {node_id}",
                                        )
                                    )
                                    return

                        log(f"Node {node_id} ready to build")

                        try:
                            log(f"{node.pill}: {pass_to_reach_target.title()}")
                            new_node = pass_to_reach_target.pass_function(
                                pass_config, graph, node
                            )
                            assert isinstance(
                                new_node, Node
                            ), f"Node pass {pass_to_reach_target.title()} must return a Node object, not a {type(new_node)}."
                            if new_node.phase < target_phase:
                                warn(
                                    f"Pass {pass_to_reach_target.title()} did not update phase for node {node.pill} to {target_phase}"
                                )
                                node.error(
                                    phase=target_phase,
                                    message=f"{pass_to_reach_target.title()} failed on `{node.pill}`",
                                )
                        except Exception as e:
                            error(f"Error processing node {node_id}: {e}")
                            error(f"Full report: {traceback.format_exc()}")
                            new_node = node.error(
                                phase=target_phase,
                                message=f"Internal Error: {e}\n```\n{traceback.format_exc()}\n```\n",
                            )
                        log("Done", node_id)
                result_queue.put(
                    NodeResult(work_item, node, new_node, buffer.get_text())
                )
                return

            def init_worker(session):
                setattr(current_thread(), "flowco_session", session)

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=16,
                initializer=init_worker,
                initargs=(session.get_session(),),
            ) as executor:

                new_graph = graph
                done = set()
                in_flight = set()

                def submit_items(current_worklist: Dict[WorkItem, List[WorkItem]]):
                    ready = [
                        item
                        for item in current_worklist
                        if len(current_worklist[item]) == 0
                    ]
                    log(f"Submitting {ready}")
                    current_worklist = {
                        k: v for k, v in current_worklist.items() if k not in ready
                    }
                    for work_item in ready:
                        executor.submit(process_workitem, new_graph, work_item)
                        in_flight.add(work_item)
                    return current_worklist

                log("workllist:", pprint.pformat(worklist, sort_dicts=True, indent=2))
                current_worklist = submit_items(worklist)

                with logger("Running worklist"):
                    stopper = session.get("stopper", Stopper)
                    while len(done) < len(worklist) and not stopper.should_stop():
                        try:
                            log(f"Remaining: {len(worklist.keys()) - len(done)} steps")
                            result = result_queue.get()
                            result_queue.task_done()
                            item = result.item
                            done |= {item}
                            in_flight -= {item}
                            with logger(f"{item} Complete"):
                                log(result.buffer)
                                new_node = result.new_node
                                new_node.build_status = None
                                if new_node.phase < item.target_phase:
                                    warn(
                                        f"Pass did not update phase for node {new_node.pill} to {target_phase}"
                                    )
                                new_graph = new_graph.with_node(new_node)
                                current_worklist = {
                                    k: [x for x in v if x != item]
                                    for k, v in current_worklist.items()
                                }
                                current_worklist = submit_items(current_worklist)
                                b = BuildUpdate(
                                    steps_remaining=len(worklist.keys()) - len(done),
                                    steps_total=len(worklist.keys()),
                                    new_graph=new_graph,
                                    updated_node=new_node,
                                )
                                yield b
                        except Exception as e:
                            error(e)
                            raise e
                    if stopper.should_stop():
                        executor.shutdown(wait=False, cancel_futures=True)
                        log(f"Stopping!")

                    new_graph = new_graph.update(
                        nodes=[x.update(build_status=None) for x in new_graph.nodes]
                    )
                    yield BuildUpdate(
                        steps_remaining=len(worklist.keys()) - len(done),
                        steps_total=len(worklist.keys()),
                        new_graph=new_graph,
                        updated_node=None,
                    )
                    log("Worklist done")

    @staticmethod
    def import_function(full_name):
        """
        Note: Only Functions!  No Methods.
        """
        module_name, func_name = full_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        return func

    @staticmethod
    def get_builder(passes_key: str | None = None) -> "BuildEngine":
        with logger(f"Making build engine: {passes_key}"):
            builder = BuildEngine()
            builder_passes = config.get_build_passes(passes_key=passes_key)
            for qual_name in builder_passes:
                builder.add_pass(BuildEngine.import_function(qual_name))
            return builder


def assert_pass_has_signature_and_types(func, is_node_pass):
    parameters = inspect.signature(func).parameters
    if is_node_pass:
        assert (
            len(parameters) == 3
            and parameters["pass_config"].annotation == PassConfig
            and parameters["graph"].annotation == DataFlowGraph
            and parameters["node"].annotation == Node
        ), f"Node pass {func.__name__} must have signature (pass_config: PassConfig, graph: DataFlowGraph, node: Node)."
        assert (
            inspect.signature(func).return_annotation == Node
        ), f"Node pass {func.__name__} must return a Node object."
    else:
        assert (
            len(parameters) == 2
            and parameters["pass_config"].annotation == PassConfig
            and parameters["graph"].annotation == DataFlowGraph
        ), f"Graph pass {func.__name__} must have signature (pass_config: PassConfig, graph: DataFlowGraph), not {parameters}"
        assert (
            inspect.signature(func).return_annotation == DataFlowGraph
        ), f"Graph pass {func.__name__} must return a Graph object."


def node_pass(
    required_phase: Phase, target_phase: Phase, pred_required_phase: Phase | None = None
):
    def decorator(func):
        assert_pass_has_signature_and_types(func, True)
        func._is_node_pass = True
        func._required_phase = required_phase
        func._target_phase = target_phase
        func._pred_required_phase = pred_required_phase
        return func

    return decorator

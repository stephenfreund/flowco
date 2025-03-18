from __future__ import annotations

import functools
import json
import os
import threading
from typing import Iterable, List, Optional, TypeVar, Union

import markdown
from pydantic import BaseModel, Field, PrivateAttr
from pydantic_core import ValidationError

from flowco.dataflow.phase import Phase
from flowco.dataflow.dfg import DataFlowGraph
from flowco.builder.build import BuildEngine, BuildUpdate, PassConfig
from flowco.page.tables import GlobalTables
from flowco.session.session_file_system import fs_exists, fs_read, fs_write
from flowco.util.config import AbstractionLevel, config
from flowco.util.errors import FlowcoError
from flowco.util.output import error, log, logger
from flowco.util.text import (
    format_basemodel,
)
from flowthon.flowthon import FlowthonNode, FlowthonProgram

T = TypeVar("T", bound=Union[BaseModel, str])


class PageListener:
    def page_saved(self, page: "Page"):
        pass


class UndoStack:
    def __init__(self):
        self.undo_stack: List[DataFlowGraph] = []
        self.redo_stack: List[DataFlowGraph] = []
        self.atomic_depth = 0
        self.lock = threading.RLock()

    def clear(self):
        with self.lock:
            self.undo_stack.clear()
            self.redo_stack.clear()

    def __str__(self):
        with self.lock:
            undos = [hash(dfg.model_dump_json()) % 1000 for dfg in self.undo_stack]
            redos = [hash(dfg.model_dump_json()) % 1000 for dfg in self.redo_stack]
            return f"UndoStack(undos={undos}, redos={redos})"

    def push(self, dfg: DataFlowGraph):
        with self.lock:
            if self.atomic_depth == 0 and (
                not self.undo_stack or not dfg.semantically_eq(self.undo_stack[-1])
            ):
                with logger("Pushing to undo stack"):
                    # if not self.undo_stack:
                    #     log("First push")
                    # # else:
                    # log(f"Changes: {self.undo_stack[-1].diff(dfg).affected_paths}")
                    self.undo_stack.append(dfg)
                    self.redo_stack.clear()

    # Only possible if atomic_depth == 0, but we don't check that here
    # so the UI can refresh properly even when atomic_depth > 0
    def can_undo(self) -> bool:
        with self.lock:
            return len(self.undo_stack) > 0

    # Only possible if atomic_depth == 0, but we don't check that here
    # so the UI can refresh properly even when atomic_depth > 0
    def can_redo(self) -> bool:
        with self.lock:
            return len(self.redo_stack) > 0

    # No op if in atomic operation or nothing left
    def undo(self, current: DataFlowGraph):
        with self.lock:
            if self.atomic_depth == 0 and self.undo_stack:
                dfg = self.undo_stack.pop()
                dfg = dfg.update(
                    nodes=[node.update(build_status=None) for node in dfg.nodes],
                    version=dfg.version + 1,
                )
                self.redo_stack.append(current)
                return dfg
            else:
                return current

    # No op if in atomic operation or nothing left
    def redo(self, current: DataFlowGraph):
        with self.lock:
            if self.atomic_depth == 0 and self.redo_stack:
                dfg = self.redo_stack.pop()
                dfg = dfg.update(version=dfg.version + 1)
                self.undo_stack.append(current)
                return dfg
            else:
                return current

    def inc(self, current: DataFlowGraph):
        with self.lock:
            if self.atomic_depth == 0:
                self.push(current)
            self.atomic_depth += 1

    def dec(self):
        with self.lock:
            self.atomic_depth -= 1


class Page(BaseModel, extra="allow"):
    # fixed part
    file_name: str = Field(frozen=True)

    # generation part
    dfg: DataFlowGraph

    # listeners for changes
    _listeners: List[PageListener] = PrivateAttr(default=[])
    _undo_stack: UndoStack = PrivateAttr(default_factory=UndoStack)

    def __init__(self, **data):
        super().__init__(**data)

    def __setattr__(self, name, value):
        current = self.__getattribute__(name)

        super().__setattr__(name, value)

        if name in ["dfg"] and current is not value:
            self.save()

    def atomic(self):
        return self

    @staticmethod
    def atomic_method(f):
        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            with self:
                return f(self, *args, **kwargs)

        return wrapper

    def __enter__(self):
        self.atomic_enter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.atomic_exit()

    def atomic_enter(self):
        self._undo_stack.inc(self.dfg)

    def atomic_exit(self):
        self._undo_stack.dec()

    def undo(self) -> None:
        self.dfg = self._undo_stack.undo(self.dfg)

    def redo(self) -> None:
        self.dfg = self._undo_stack.redo(self.dfg)

    def can_undo(self) -> bool:
        return self._undo_stack.can_undo()

    def can_redo(self) -> bool:
        return self._undo_stack.can_redo()

    def add_listener(self, listener: PageListener):
        self._listeners.append(listener)

    def save(self):
        with logger(f"Saving page {self.file_name}"):
            for listener in self._listeners:
                listener.page_saved(self)
            fs_write(self.file_name, self.model_dump_json(indent=2))

    def __str__(self):
        return format_basemodel(
            self,
            [
                "file_name",
                "dfg",
            ],
            drop_missing=True,
        )

    @classmethod
    def create(cls, file_name: str):
        with logger(f"Creating page"):

            with logger("Creating dataflow graph"):
                dfg = DataFlowGraph(version=0)

            page = cls(
                file_name=file_name,
                dfg=dfg,
            )
            page.save()
            return page

    @classmethod
    def from_file(cls, file_name: str):
        with logger(f"Loading page"):
            if not fs_exists(file_name):
                raise FlowcoError(f"Page {file_name} does not exist.")

            data = json.loads(fs_read(file_name))

            # Include the filename in the initialization data
            data["file_name"] = file_name

            # Initialize the model using the file data
            try:
                page = cls(**data)
                page.ensure_valid()
            except ValidationError as e:
                error(f"Page {file_name} is not valid", e)
                error("Doing hard reset.")

                if "dfg" not in data or data["dfg"] is None:
                    data["dfg"] = DataFlowGraph(version=0)
                else:
                    data["dfg"] = DataFlowGraph.hard_reset(**data["dfg"])

                page = Page(
                    file_name=data["file_name"],
                    dfg=data["dfg"],
                )
                page.save()
        return page

    def ensure_valid(self):
        with logger("Ensuring page is valid..."):
            if not fs_exists(self.file_name):
                raise FlowcoError(f"Page {self.file_name} does not exist.")

            self.dfg.ensure_valid()

    def check_up_to_date(self):
        self.ensure_valid()

    @atomic_method
    def update_dfg(self, dfg: DataFlowGraph):
        """
        This is the *only* way that a user action or build action should directly update the dfg.
        """
        initial_dfg = dfg
        if self.dfg != dfg:
            # set any node that has a predecessor at Phase.clean to also be clean
            for node_id in dfg.topological_sort():
                changed_preds = [
                    initial_dfg[pred]
                    for pred in initial_dfg[node_id].predecessors
                    if initial_dfg[pred].phase == Phase.clean
                ]
                if changed_preds:
                    dfg[node_id].warn(
                        phase=Phase.requirements,
                        message=f"Must re-evaluate requirements because of changes to: {', '.join([f'{p.pill}' for p in changed_preds])}",
                    )
                    dfg = dfg.with_phase(node_id, Phase.clean)
            self._undo_stack.push(self.dfg)
            self.dfg = dfg

    @atomic_method
    def reset(self, reset_requirements: bool = False):
        self.update_dfg(self.dfg.reset(reset_requirements=reset_requirements))
        self._undo_stack.clear()

    @atomic_method
    def clear_outputs(self):
        self.update_dfg(self.dfg.clear_outputs())
        self.clean(phase=Phase.runnable)
        self.save()

    @atomic_method
    def clean(self, node_id: Optional[str] = None, phase: Phase = Phase.clean):
        """
        Clean the page or a specific node.  Call this before building to
        regenerate all the node information.
        """
        with logger("Clean"):
            if node_id is None:
                with logger("Cleaning page"):
                    new_dfg = self.dfg.lower_phase_with_successors(node_id, phase)
                    self.update_dfg(new_dfg)
            else:
                with logger(f"Cleaning node {node_id}"):
                    new_dfg = self.dfg.lower_phase_with_successors(node_id, phase)
                    self.update_dfg(new_dfg)

    # @atomic
    # def invalidate_build_cache(self, target: Phase, node_id: Optional[str] = None):
    #     with logger("Clear Caches"):
    #         new_dfg = self.dfg.invalidate_build_cache(node_id, target)
    #         self.update_dfg(new_dfg)

    @atomic_method
    def reduce_phases_to_below_target(
        self, node_id: Optional[str], target: Optional[Phase] = None
    ):
        with logger("Reducing Phases"):
            new_dfg = self.dfg.reduce_phases_to_below_target(node_id, target)
            log(f"Changes: {self.dfg.diff(new_dfg)}")
            self.update_dfg(new_dfg)

    @atomic_method
    def invalidate_build_cache(self, node_id, phase: Phase) -> None:
        with logger("Invalidating Build Cache"):
            new_dfg = self.dfg.invalidate_build_cache(node_id)
            log(f"Changes: {self.dfg.diff(new_dfg)}")
            self.update_dfg(new_dfg)

    def base_build_config(self, repair: bool) -> PassConfig:

        return PassConfig(
            tables=GlobalTables.from_dfg(self.dfg),
            max_retries=config().retries if repair else 0,
        )

    @atomic_method
    def build(
        self,
        engine: BuildEngine,
        target_phase: Phase,
        repair: bool,
        target_node_id: Optional[str] = None,
    ) -> Iterable[BuildUpdate]:

        self.check_up_to_date()

        build_config = self.base_build_config(repair=repair)

        with logger("Building"):
            for build_updated in engine.build_with_worklist(
                build_config, self.dfg, target_phase, target_node_id
            ):
                self.update_dfg(build_updated.new_graph)
                yield build_updated

    def to_markdown(self) -> str:
        return f"""# {self.file_name}\n\n{self.dfg.to_markdown()}\n\n"""

    def to_html(self) -> str:
        md = self.to_markdown()
        html_content = markdown.markdown(
            md,
            extensions=[
                "extra",  # Includes several extensions like tables, fenced code, etc.
                "codehilite",  # Adds syntax highlighting to code blocks
                "toc",  # Generates a table of contents
                "sane_lists",  # Improves list handling
                "smarty",  # Converts quotes and dashes to smart quotes and dashes
            ],
        )

        with open(os.path.join(os.path.dirname(__file__), "template.html"), "r") as f:
            html = f.read()

        return html.format(content=html_content, title=self.file_name)

    @atomic_method
    def to_flowthon(self) -> FlowthonProgram:
        dfg = self.dfg
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
                assertions=node.assertions,
            )

        self.update_dfg(dfg)
        return FlowthonProgram(tables=GlobalTables.from_dfg(dfg), nodes=nodes)

    @atomic_method
    def merge_flowthon(
        self, flowthon: FlowthonProgram, rebuild=True, interactive=True
    ) -> None:
        with logger("Merging"):
            build_config = self.base_build_config(repair=True)
            new_dfg = flowthon.merge(
                build_config, self.dfg, rebuild=rebuild, interactive=interactive
            )
            self.update_dfg(new_dfg)

    @atomic_method
    def user_edit_graph_description(self, description: str) -> None:
        if description != self.dfg.description:
            self.update_dfg(self.dfg.update(description=description))

    @atomic_method
    def user_edit_node_assertions(self, node_id: str, assertions: List[str]) -> None:
        node = self.dfg[node_id]
        if assertions != node.assertions:
            self.update_dfg(self.dfg.with_node(node.update(assertions=assertions)))
            self.update_dfg(
                self.dfg.reduce_phases_to_below_target(node.id, Phase.assertions_code)
            )

    @atomic_method
    def user_build_node_to_phase(
        self, node_id: str, phase: Phase, repair=False
    ) -> None:
        builder = BuildEngine.get_builder()
        for _ in self.build(builder, phase, repair=repair, target_node_id=node_id):
            pass

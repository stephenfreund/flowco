from __future__ import annotations

import functools
import json
import os
import threading
from typing import Iterable, List, Optional, TypeVar, Union

import markdown
from pydantic import BaseModel, Field, PrivateAttr
from pydantic_core import ValidationError

from flowco.dataflow.editor import FlowthonGraph
from flowco.dataflow.phase import Phase
from flowco.dataflow.dfg import DataFlowGraph
from flowco.builder.build import BuildEngine, BuildUpdate, PassConfig
from flowco.page.tables import GlobalTables
from flowco.session.session import session
from flowco.session.session_file_system import fs_exists, fs_read, fs_write
from flowco.util.config import AbstractionLevel, config
from flowco.util.errors import FlowcoError
from flowco.util.output import error, log, logger
from flowco.util.text import (
    format_basemodel,
)

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
                    if not self.undo_stack:
                        log("First push")
                    else:
                        log(f"{self.undo_stack[-1].diff(dfg)}")
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

    # No op if in atomic operation or nothing left
    def redo(self, current: DataFlowGraph):
        with self.lock:
            if self.atomic_depth == 0 and self.redo_stack:
                dfg = self.redo_stack.pop()
                dfg = dfg.update(version=dfg.version + 1)
                self.undo_stack.append(current)
            return dfg

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

    # csv files
    tables: GlobalTables = GlobalTables()

    # listeners for changes
    _listeners: List[PageListener] = PrivateAttr(default=[])
    _undo_stack: UndoStack = PrivateAttr(default_factory=UndoStack)

    def __init__(self, **data):
        super().__init__(**data)

    def __setattr__(self, name, value):
        current = self.__getattribute__(name)

        super().__setattr__(name, value)

        if name in ["dfg", "tables"] and current is not value:
            self.save()

    def atomic(self):
        return self

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
            ["file_name", "dfg", "tables"],
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
                error(f"Page {file_name} is not valid: {e}.")
                error("Doing hard reset.")

                if "dfg" not in data or data["dfg"] is None:
                    data["dfg"] = DataFlowGraph(version=0)
                else:
                    data["dfg"] = DataFlowGraph.hard_reset(**data["dfg"])

                page = Page(
                    file_name=data["file_name"],
                    dfg=data["dfg"],
                    tables=data["tables"],
                )
                page.save()
        return page

    def ensure_valid(self):
        with logger("Ensuring page is valid..."):
            if not fs_exists(self.file_name):
                raise FlowcoError(f"Page {self.file_name} does not exist.")

            self.dfg.ensure_valid()

    def add_table(self, file_path: str):
        with logger(f"Adding file"):
            self.tables = self.tables.add(file_path)
        self.clean()

    def remove_table(self, file_path: str):
        with logger(f"Removing file"):
            self.tables = self.tables.remove(file_path)
        self.clean()

    def update_tables(self, tables: GlobalTables):
        if tables != self.tables:
            self.tables = tables
            self.clean()

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
            tables=self.tables,
            max_retries=config.retries if repair else 0,
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
    def to_flowthon(self, level: AbstractionLevel, file_name: str) -> None:
        dfg, editable = FlowthonGraph.from_dfg(self.dfg)
        self.update_dfg(dfg)

        # make file name by replacing the extension .flowco, with .flowthon
        with open(file_name, "w") as f:
            rep = {
                "tables": self.tables.model_dump(),
                "graph": editable.to_json(level),
            }
            json_str = json.dumps(rep, indent=2)
            f.write(json_str)

        # editor = os.environ.get('EDITOR', "nano")
        # subprocess.run([editor, temp_path.name], check=True)

    @atomic_method
    def merge_flowthon(self, file_name, interactive=True) -> None:
        with logger("Merging"):
            with open(file_name, "r") as f:
                new_json = f.read()

            rep = json.loads(new_json)
            self.tables = GlobalTables(**rep["tables"])
            new_editable = FlowthonGraph.from_json(rep["graph"])
            build_config = self.base_build_config(repair=True)
            new_dfg = new_editable.merge(
                build_config, self.dfg, interactive=interactive
            )
            self.update_dfg(new_dfg)

            if any(node.code for node in new_dfg.nodes):
                level = AbstractionLevel.code
            elif any(node.algorithm for node in new_dfg.nodes):
                level = AbstractionLevel.algorithm
            else:
                level = AbstractionLevel.spec

            with open(file_name, "w") as f:
                rep = {
                    "tables": self.tables.model_dump(),
                    "graph": new_editable.to_json(level),
                }
                json_str = json.dumps(rep, indent=2)
                f.write(json_str)

    # @atomic
    # def add_bugs(self, node_ids: Optional[List[str]]) -> None:
    #     """
    #     Sketchy -- not updated recently.
    #     """
    #     with logger("Creating assistant"):
    #         assistant = Assistant("add-bugs")
    #         assistant.add_json_object(
    #             "Here is the dataflow graph matching the diagram",
    #             self.dfg.model_dump(),
    #         )

    #     with logger("Running assistant"):
    #         # TODO New completion model
    #         new_graph = graph_completion(
    #             assistant,
    #             DataFlowGraph,
    #         )

    #     with logger("Updating graph"):
    #         self.dfg = self.dfg.update(**new_graph.model_dump()).with_phase(
    #             node_ids=None, phase=Phase.tests
    #         )

    # def check_unit_test(self, node_id: str, test_uid: str) -> None:
    #     with logger(f"Check Unit Test {node_id} {test_uid}"):
    #         self.check_up_to_date()
    #         self.build(
    #             BuildEngine.get_builder(),
    #             target_phase=Phase.code,
    #             repair=True,
    #             target_node_id=node_id,
    #         )

    #         build_config = self.base_build_config(repair=True)

    #         new_node, _ = check_unit_tests(
    #             build_config,
    #             self.dfg,
    #             self.dfg[node_id],
    #             test_uid,
    #         )
    #         self.update_dfg(self.dfg.with_node(new_node))

    # @atomic
    # def remove_unit_test(self, node_id, test_uid: str) -> None:
    #     with logger(f"Remove Unit Test {node_id} {test_uid}"):
    #         node = self.dfg[node_id]
    #         unit_tests = node.unit_tests
    #         if unit_tests is None:
    #             return
    #         new_unit_tests = [test for test in unit_tests if test.uuid != test_uid]
    #         new_node = node.update(unit_tests=new_unit_tests)
    #         self.update_dfg(self.dfg.with_node(new_node))

    # @atomic
    # def check_sanity_check(self, node_id: str, test_uid: str) -> None:
    #     with logger(f"Check Sanity Check {node_id} {test_uid}"):
    #         self.check_up_to_date()
    #         self.build(
    #             BuildEngine.get_builder(),
    #             target_phase=Phase.code,
    #             repair=True,
    #             target_node_id=node_id,
    #         )

    #         build_config = self.base_build_config(repair=True)

    #         new_node, _ = check_sanity_checks(
    #             build_config,
    #             self.dfg,
    #             self.dfg[node_id],
    #             test_uid,
    #         )
    #         self.update_dfg(self.dfg.with_node(new_node))

    # def remove_sanity_check(self, node_id, test_uid: str) -> None:
    #     with logger(f"Remove Sanity Check {node_id} {test_uid}"):
    #         node = self.dfg[node_id]
    #         sanity_checks = node.sanity_checks
    #         if sanity_checks is None:
    #             return
    #         new_sanity_checks = [
    #             test for test in sanity_checks if test.uuid != test_uid
    #         ]
    #         new_node = node.update(sanity_checks=new_sanity_checks)
    #         self.update_dfg(self.dfg.with_node(new_node))

    @atomic_method
    def user_edit_graph_description(self, description: str) -> None:
        if description != self.dfg.description:
            self.update_dfg(self.dfg.update(description=description))

from queue import Empty, Queue
import threading
from typing import List
from flowco.builder.build import BuildEngine, BuildUpdate, PassConfig
from flowco.dataflow.dfg import DataFlowGraph
from flowco.dataflow.phase import Phase
from flowco.page.page import Page
from flowco.pythonshell.shells import PythonShells
from flowco.session.session import session
from flowco.util.output import log, logger
from flowco.util.stopper import Stopper


def run(
    engine: BuildEngine,
    page: Page,
    build_config: PassConfig,
    node_ids: str | List[str] | None,
    target_phase: Phase,
    queue: Queue,
):

    # Todo: Need a way to signal a stop so the worklist stops.
    # Get rid ot todo crap and just pass in a callback.

    with session.get("stopper", Stopper):
        with page:
            for build_updated in engine.build_with_worklist(
                build_config, page.dfg, target_phase, node_ids
            ):
                queue.put(build_updated)
            queue.join()

        log("Done building.")


class Builder:

    def __init__(
        self,
        page: Page,
        node_ids: str | List[str] | None = None,
        target_phase: Phase = Phase.run_checked,
        passes_key: str | None = None,
        force: bool = False,
        repair: bool = True,
    ):
        builder = BuildEngine.get_builder(passes_key=passes_key)

        p = builder.passes_by_target[target_phase]
        if force:
            page.clean(phase=p.required_phase())
        build_config = page.base_build_config(repair=repair)

        verb = (
            "Building"
            if target_phase != Phase.run_checked
            else ("Running" if force else "Updating")
        )
        if node_ids is None:
            name = ""
        elif isinstance(node_ids, str):
            name = page.dfg[node_ids].pill
        else:
            name = ", ".join([page.dfg[x].pill for x in node_ids])
        self.message = f"{verb} {name}..."

        self.queue = Queue[BuildUpdate]()
        self.thread = threading.Thread(
            target=run,
            args=(builder, page, build_config, node_ids, target_phase, self.queue),
        )
        setattr(self.thread, "flowco_session", session.get_session())
        self.thread.start()

    def get_message(self):
        return self.message

    def empty(self) -> bool:
        return self.queue.empty()

    def get(self) -> BuildUpdate:
        return self.queue.get(timeout=0.1)

    def update_done(self) -> None:
        self.queue.task_done()

    def is_alive(self) -> bool:
        return self.thread.is_alive()

    def stop(self) -> None:
        with logger("Stopping builder"):
            session.get("stopper", Stopper).stop()

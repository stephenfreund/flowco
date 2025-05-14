from queue import Queue
import threading
from typing import Callable, List
from flowco.builder.build import BuildEngine, BuildUpdate, PassConfig
from flowco.dataflow.phase import Phase
from flowco.page.page import Page
from flowco.session.session import session
from flowco.util.output import log


def run(
    engine: BuildEngine,
    page: Page,
    build_config: PassConfig,
    node_ids: str | List[str] | None,
    target_phase: Phase,
    queue: Queue,
    should_stop: Callable[[], bool],
):

    with page:
        for build_updated in engine.build_with_worklist(
            build_config, page.dfg, target_phase, node_ids, should_stop
        ):
            queue.put(build_updated)

        log("Waiting for no unfinished tasks.")
        with queue.all_tasks_done:
            if queue.unfinished_tasks:
                # Just wait a bit for the build page to clear out pending updates...
                if queue.all_tasks_done.wait(3):
                    log("All tasks done.")
                else:
                    log("Timeout waiting for tasks to finish.")

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
            if node_ids is None:
                page.clean(phase=p.required_phase())
            elif isinstance(node_ids, str):
                page.clean(node_ids, phase=p.required_phase())
            else:
                for node_id in node_ids:
                    page.clean(node_id, phase=p.required_phase())

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

        self.stopped = False

        self.queue = Queue[BuildUpdate]()
        self.thread = threading.Thread(
            target=run,
            args=(
                builder,
                page,
                build_config,
                node_ids,
                target_phase,
                self.queue,
                self.should_stop,
            ),
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

    def should_stop(self) -> bool:
        return self.stopped

    def stop(self) -> None:
        self.stopped = True

import asyncio
import queue
import threading
from flowco.dataflow.dfg import DataFlowGraph, Node
from flowco.page.output import NodeResult
from flowco.page.tables import GlobalTables
from flowco.pythonshell.shell import EvalResult, PythonShell
from concurrent.futures import ThreadPoolExecutor
from flowco.session.session import session


class PythonShells:
    def __init__(self, num_shells: int = 4, restart_threadhold: int = 100):
        self.queue = queue.Queue()
        self.restart_threadhold = restart_threadhold
        self._preload_shells(num_shells)

    def _preload_shells(self, num_shells) -> None:
        def init_worker(session_data):
            setattr(threading.current_thread(), "flowco_session", session_data)

        def load_shell() -> None:
            try:
                self.queue.put(PythonShell())
            except Exception as e:
                print(f"Error loading PythonShell: {e}")

        preloader = ThreadPoolExecutor(
            max_workers=4,
            initializer=init_worker,
            initargs=(session.get_session(),),
        )

        print(
            f"[Preloading {num_shells} PythonShells in background.  restart_threadhold={self.restart_threadhold}]"
        )
        for _ in range(num_shells):
            preloader.submit(load_shell)

    def _get_shell(self) -> PythonShell:
        return self.queue.get()

    async def _restart_and_put(self, shell: PythonShell, session_data) -> None:
        setattr(threading.current_thread(), "flowco_session", session_data)
        await shell.restart()
        self.queue.put(shell)

    def _put_shell(self, shell: PythonShell) -> None:
        # print (f"Shell run count: {shell.run_count()}")
        if shell.run_count() > self.restart_threadhold:
            # print(f"Restarting PythonShell due to usage threshold ({self.restart_threadhold}).")
            threading.Thread(
                target=asyncio.run,
                args=(self._restart_and_put(shell, session.get_session()),),
            ).start()
        else:
            self.queue.put(shell)

    # These must be thread safe entry points:

    def run(self, code: str) -> EvalResult:
        shell = self._get_shell()
        try:
            return shell.run(code)
        finally:
            self._put_shell(shell)

    def run_node(
        self, tables: GlobalTables, dfg: DataFlowGraph, node: Node
    ) -> NodeResult:
        shell = self._get_shell()
        try:
            return shell.run_node(tables, dfg, node)
        finally:
            self._put_shell(shell)

    def run_assertions(
        self, tables: GlobalTables, dfg: DataFlowGraph, node: Node
    ) -> Node:
        shell = self._get_shell()
        try:
            return shell.run_assertions(tables, dfg, node)
        finally:
            self._put_shell(shell)
    

    def close_all(self) -> None:
        while not self.queue.empty():
            shell = self.queue.get()
            shell.close()

    def __del__(self):
        self.close_all()

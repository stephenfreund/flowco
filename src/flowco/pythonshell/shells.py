import atexit
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
from time import sleep
from typing import Dict
from flowco.page.tables import GlobalTables
from flowco.pythonshell.shell import PythonShell
from flowco.util.output import buffer_output, debug, log, logger, session
from concurrent.futures import ThreadPoolExecutor
from typing import List
from flowco.session.session import session


@dataclass
class ShellEntry:
    shell: PythonShell
    last_access_time: datetime = field(default_factory=datetime.utcnow)


class PythonShells:
    def __init__(self):
        """
        Initializes the PythonShells manager.

        Args:
            session: The session object required for initializing workers.
        """
        # Mapping from node_id to ShellEntry
        self.shells: Dict[str, ShellEntry] = {}

        self.lock = threading.Lock()

        def init_worker(session_data):
            setattr(threading.current_thread(), "flowco_session", session_data)

        self.preloader = ThreadPoolExecutor(
            max_workers=16,
            initializer=init_worker,
            initargs=(session.get_session(),),
        )

        # # Event to signal the cleaner thread to stop
        # self._stop_cleaner = threading.Event()

        # # Start the background cleaner thread
        # self.cleaner_thread = threading.Thread(target=self._cleaner, daemon=True)
        # self.cleaner_thread.start()

        # # Ensure that all resources are cleaned up on exit
        # atexit.register(self.close_all)

    # def shutdown_executor(self):
    #     """
    #     Shuts down the thread pool executor.
    #     """
    #     self.preloader.shutdown(wait=False)
    #     log("ThreadPoolExecutor has been shut down.")

    def preload_shells(self, tables: GlobalTables, node_ids: List[str]) -> None:
        """
        Preloads PythonShell instances for all nodes in the given GlobalTables
        using a thread pool with 16 threads.

        Args:
            tables (GlobalTables): The GlobalTables instance to preload shells for.
            node_ids (List[str]): List of node identifiers to preload shells for.
        """

        def load_shell(node_id: str) -> None:
            with buffer_output(
                f"Preloading PythonShell for node '{node_id}'"
            ) as buffer:
                with self.lock:
                    if node_id in self.shells:
                        log(
                            f"PythonShell already exists for node '{node_id}' during preload."
                        )
                        return
                try:
                    shell = PythonShell(tables=tables)
                    with self.lock:
                        if node_id not in self.shells:
                            self.shells[node_id] = ShellEntry(shell=shell)
                            log(f"Preloaded PythonShell for node '{node_id}'.")
                        else:
                            log(
                                f"PythonShell already exists for node '{node_id}' during preload."
                            )
                except Exception as exc:
                    print(f"Node {node_id} generated an exception: {exc}\n{2}")
            debug(f"{buffer.get_text()}")

        with logger("Preloading PythonShells in background"):
            for node_id in node_ids:
                self.preloader.submit(load_shell, node_id)

    def get_shell_for_node(self, tables: GlobalTables, node_id: str) -> PythonShell:
        """
        Retrieves the PythonShell associated with the given node_id.
        If it does not exist, creates a new PythonShell and adds it to the mapping.

        Args:
            tables (GlobalTables): The GlobalTables instance to associate with the shell.
            node_id (str): The unique identifier for the node.

        Returns:
            PythonShell: The PythonShell instance associated with the node.
        """
        with self.lock:
            if node_id not in self.shells:
                self.shells[node_id] = ShellEntry(shell=PythonShell(tables=tables))
                log(f"Created new PythonShell for node '{node_id}'.")
            else:
                entry = self.shells[node_id]
                if entry.shell.tables != tables:
                    entry.shell.close()
                    self.shells[node_id] = ShellEntry(shell=PythonShell(tables=tables))
                    log(
                        f"Recreated PythonShell for node '{node_id}' due to table changes."
                    )
                else:
                    # Update the last access time
                    entry.last_access_time = datetime.utcnow()
                    log(
                        f"PythonShell already exists and is alive for node '{node_id}'."
                    )
            return self.shells[node_id].shell

    def close_shell_for_node(self, node_id: str) -> None:
        """
        Closes the PythonShell associated with the given node_id and removes it from the mapping.

        Args:
            node_id (str): The unique identifier for the node.
        """
        with self.lock:
            if node_id in self.shells:
                shell_entry = self.shells.pop(node_id)
                shell_entry.shell.close()
                log(f"Closed PythonShell for node '{node_id}'.")
            else:
                log(f"No PythonShell found for node '{node_id}' to close.")

    def close_all(self) -> None:
        """
        Closes all PythonShell instances managed by this manager.
        """
        with self.lock:
            for node_id, shell_entry in list(self.shells.items()):
                shell_entry.shell.close()
                log(f"Closed PythonShell for node '{node_id}'.")
            self.shells.clear()
            log("Closed all PythonShell instances via PythonShells manager.")

    # def _cleaner(self):
    #     """
    #     Background thread that purges shells not accessed for more than 10 minutes.
    #     """
    #     purge_interval = 60  # Check every 60 seconds
    #     purge_threshold = timedelta(minutes=10)

    #     while True:
    #         with self.lock:
    #             now = datetime.utcnow()
    #             to_remove = [
    #                 node_id for node_id, entry in self.shells.items()
    #                 if now - entry.last_access_time > purge_threshold
    #             ]
    #             for node_id in to_remove:
    #                 shell_entry = self.shells.pop(node_id)
    #                 shell_entry.shell.close()
    #                 log(f"Purged PythonShell for node '{node_id}' due to inactivity.")
    #         sleep(purge_interval)

    # def __del__(self):
    #     """
    #     Ensures that all PythonShell instances are closed when the PythonShells manager is destroyed.
    #     """
    #     self.close_all()


# def get_shell_for_node(tables: GlobalTables, node_id: str) -> PythonShell:
#     return session.get("shells", PythonShells).get_shell_for_node(tables, node_id)

# def close_shell_for_node(node_id: str) -> None:
#     return session.get("shells", PythonShells).close_shell_for_node(node_id)

# def close_all_shells() -> None:
#     return session.get("shells", PythonShells).close_all()

from dataclasses import dataclass, field
from flowco.page.tables import GlobalTables
from flowco.pythonshell.shells import PythonShells
from flowco.util.config import config


@dataclass
class PassConfig:
    max_retries: int
    tables: GlobalTables
    shells: PythonShells

    def get_shell_for_node(self, node_id: str):
        return self.shells.get_shell_for_node(self.tables, node_id)

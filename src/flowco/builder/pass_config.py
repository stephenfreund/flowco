from dataclasses import dataclass
from flowco.dataflow.dfg import Node, NodeKind
from flowco.page.tables import GlobalTables


@dataclass
class PassConfig:
    max_retries: int
    tables: GlobalTables

    def max_retries_for_node(self, node: Node) -> int:
        if node.kind == NodeKind.table:
            return 0
        return self.max_retries

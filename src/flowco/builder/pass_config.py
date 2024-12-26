from dataclasses import dataclass
from flowco.page.tables import GlobalTables


@dataclass
class PassConfig:
    max_retries: int
    tables: GlobalTables

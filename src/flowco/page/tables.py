from __future__ import annotations
from functools import lru_cache
from io import StringIO
import os
import threading
from typing import Dict, List
import pandas as pd
from pydantic import BaseModel, Field

from flowco.dataflow.dfg import DataFlowGraph, NodeKind
from flowco.dataflow.extended_type import ExtendedType
from flowco.session.session_file_system import fs_read
from flowco.util.output import error, log, warn

import seaborn as sns

description = """\
The global function `{table_function}()` has already been imported and returns a dataframe with the type
```
{type}
```
Here are the first few rows:
```
{head}
```
Use this function whenever you need to access the data in this table.  Do not redefine it.
"""


def file_path_to_table_name(file_path: str) -> str:
    basename = os.path.basename(file_path)
    if basename.endswith(".csv"):
        basename = basename[:-4]
        fixed = "".join([c if c.isalnum() else "_" for c in basename])
        if fixed[0].isdigit():
            fixed = "T" + fixed
        return fixed
    else:
        return basename


@staticmethod
@lru_cache
def sns_contents(file_path) -> str:
    return sns.load_dataset(file_path).to_csv(index=False)


def table_contents(file_path) -> str:
    if file_path.endswith(".csv"):
        return fs_read(file_path, use_cache=True)
    else:
        return sns_contents(file_path)


def table_df(file_path) -> pd.DataFrame:
    return pd.read_csv(StringIO(table_contents(file_path)))


def table_description(file_path) -> str:
    # return a string containing the first few rows of the table and the types of the columns
    df = table_df(file_path)
    return description.format(
        table_function=f"{file_path_to_table_name(file_path)}_table",
        type=ExtendedType.from_value(df),
        head=df.head(),
    )


class GlobalTables(BaseModel):
    tables: List[str] = Field(
        default_factory=list,
        description="list of csv files.",
    )

    def __init__(self, **data):
        try:
            super().__init__(**data)
        except Exception as e:
            warn("Repairing GlobalTables", e)
            tables_dict = data.get("tables", {})
            data["tables"] = list(tables_dict.keys())
            super().__init__(**data)

    def add(self, file_path: str) -> GlobalTables:
        try:
            if file_path.endswith(".csv"):
                contents = fs_read(file_path, use_cache=True)
                df = pd.read_csv(StringIO(contents))
        except:
            raise ValueError(f"Could not read contents for {file_path} as a DataFrame.")

        if file_path in self.tables:
            raise ValueError(f"File {file_path} already included.")

        if not file_path.endswith(".csv") and not file_path in sns.get_dataset_names():
            raise ValueError(f"File {file_path} must be a CSV file.")

        return GlobalTables(tables=[*self.tables, file_path])

    @classmethod
    def from_dfg(cls, dfg: DataFlowGraph) -> GlobalTables:
        gt = GlobalTables(tables=[])
        for table in [
            node.label.split("`")[1]
            for node in dfg.nodes
            if node.kind == NodeKind.table
        ]:
            gt = gt.add(table)
        return gt

    def all_files(self) -> List[str]:
        return self.tables.copy()

    # for ama
    def function_defs(self) -> List[str]:
        return [
            f"def {self.table_name(file_path)}_table() -> pd.DataFrame:\n    return pd.read_csv(StringIO('''{self.table_contents(file_path)}'''))"
            for file_path in self.tables
        ]

    def table_name(self, file_path) -> str:
        return file_path_to_table_name(file_path)

    def table_contents(self, file_path) -> str:
        if file_path.endswith(".csv"):
            return fs_read(file_path, use_cache=True)
        else:
            return sns_contents(file_path)

    def as_preconditions(self) -> Dict[str, List[str]]:
        return {
            f"{self.table_name(file_path)}_table()": [table_description(file_path)]
            for file_path in self.tables
        }

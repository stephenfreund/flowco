from __future__ import annotations
from io import StringIO
import os
from typing import Dict, List
import pandas as pd
from pydantic import BaseModel, Field

from flowco.dataflow.extended_type import ExtendedType
from flowco.session.session_file_system import fs_read
from flowco.util.output import log


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
    assert basename.endswith(".csv")
    basename = basename[:-4]
    fixed = "".join([c if c.isalnum() else "_" for c in basename])
    if fixed[0].isdigit():
        fixed = "T" + fixed
    return fixed


# class Table(BaseModel):
#     file_path: str = Field(description="The file with the data.")

#     def table_name(self) -> str:
#         return file_path_to_table_name(self.file_path)

#     def contents(self) -> str:
#         return fs_read(self.file_path, use_cache=True)

#     def description(self) -> str:
#         # return a string containing the first few rows of the table and the types of the columns
#         df = self.df()
#         return description.format(
#             table_function=f"{self.table_name()}_table",
#             type=ExtendedType.from_value(df),
#             head=df.head(),
#         )

#     def df(self) -> pd.DataFrame:
#         return pd.read_csv(StringIO(self.contents()))


class GlobalTables(BaseModel):
    tables: List[str] = Field(
        default_factory=list,
        description="list of csv files.",
    )

    def __init__(self, **data):
        try:
            super().__init__(**data)
        except Exception as e:
            log("repairing GlobalTables", e)
            tables_dict = data.get("tables", {})
            data["tables"] = list(tables_dict.keys())
            super().__init__(**data)

    def add(self, file_path: str) -> GlobalTables:
        try:
            contents = fs_read(file_path, use_cache=True)
            df = pd.read_csv(StringIO(contents))
        except:
            raise ValueError(f"Could not read contents for {file_path} as a DataFrame.")

        if file_path in self.tables:
            raise ValueError(f"File {file_path} already included.")

        if not file_path.endswith(".csv"):
            raise ValueError(f"File {file_path} must be a CSV file.")

        return GlobalTables(tables=[*self.tables, file_path])

    def all_files(self) -> List[str]:
        return self.tables.copy()

    def remove(self, file_path: str) -> GlobalTables:
        return GlobalTables(
            tables=[table for table in self.tables if table.file_path != file_path]
        )

    def as_preconditions(self) -> Dict[str, List[str]]:
        return {
            f"{self.table_name(file_path)}_table()": [self.table_description(file_path)]
            for file_path in self.tables
        }

    def function_defs(self) -> List[str]:
        return [
            f"def {self.table_name(file_path)}_table() -> pd.DataFrame:\n    return pd.read_csv(StringIO('''{self.table_contents(file_path)}'''))"
            for file_path in self.tables
        ]

    def contains(self, file_path: str) -> bool:
        return file_path in self.tables

    def table_name(self, file_path) -> str:
        return file_path_to_table_name(file_path)

    def table_contents(self, file_path) -> str:
        return fs_read(file_path, use_cache=True)

    def table_description(self, file_path) -> str:
        # return a string containing the first few rows of the table and the types of the columns
        df = self.table_df(file_path)
        return description.format(
            table_function=f"{self.table_name(file_path)}_table",
            type=ExtendedType.from_value(df),
            head=df.head(),
        )

    def table_df(self, file_path) -> pd.DataFrame:
        return pd.read_csv(StringIO(self.table_contents(file_path)))

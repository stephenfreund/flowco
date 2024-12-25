from pydantic import BaseModel
import textwrap
from typing import Dict, TypeVar, Union
import os
import shutil
from tempfile import TemporaryDirectory
from typing import Any, List, Optional

import nbformat as nbf
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from nbconvert import HTMLExporter

from flowco.dataflow.dfg import (
    DataFlowGraph,
)
from flowco.dataflow.tests import TestCase
from flowco.page.api import PageAPI
from flowco.pythonshell.sandbox import Sandbox
from flowco.util.output import log, logger
from flowco.util.stoppable import Stoppable


class DataFlowNotebookExecutionError(CellExecutionError):
    def __init__(
        self,
        traceback: str,
        ename: str,
        evalue: str,
        cell,
        computed_results: Optional[List[Dict[str, Any]]],
    ):
        super().__init__(traceback, ename, evalue)
        self.cell = cell
        if computed_results is None:
            self.computed_results = NodeOutputs()
        else:
            self.computed_results = NodeOutputs.from_outputs(computed_results)

    # class Sandbox:
    #     def __init__(self, files: List[str] = []):
    #         self.sandbox_dir = TemporaryDirectory()
    #         self.files = files
    #         self.restore()

    #     def __enter__(self):
    #         log(f"Entering sandbox {self.sandbox_dir.name}")
    #         self.sandbox_dir.__enter__()
    #         return self

    #     def __exit__(self, exc_type, exc_value, traceback):
    #         log(f"Exiting sandbox {self.sandbox_dir.name}")
    #         return self.sandbox_dir.__exit__(exc_type, exc_value, traceback)

    #     def restore(self):
    #         # remove all files and directories in the sandbox
    #         for file in os.listdir(self.sandbox_dir.name):
    #             file_path = os.path.join(self.sandbox_dir.name, file)
    #             if os.path.isdir(file_path):
    #                 shutil.rmtree(file_path)
    #             else:
    #                 os.remove(file_path)

    #         # copy all files and directories from self.files to the sandbox
    #         for file in self.files:
    #             log(f"Copying file {file} to sandbox")
    #             file_path = os.path.join(self.sandbox_dir.name, os.path.basename(file))
    #             shutil.copy(file, file_path)


class NotebookConfig(BaseModel):
    file_name: str
    api: PageAPI
    dfg: DataFlowGraph
    data_files: List[str] = []


class BaseDataflowNotebook:
    def __init__(self, config: NotebookConfig):
        self.config = config

        self.imports = [
            "import pandas as pd",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "import sklearn",
        ]

        nb = nbf.v4.new_notebook()

        # Add markdown cell with graph description
        dfg = self.config.dfg
        if dfg.description:
            md_cell = nbf.v4.new_markdown_cell(dfg.description)
            nb.cells.append(md_cell)

        import_lines = "\n".join(self.imports) + "\n" + config.api.imports_code()
        # Disable all warnings
        nb.cells.append(
            nbf.v4.new_code_cell(
                f"import warnings\nwarnings.simplefilter('ignore')\n%matplotlib inline\nfrom typing import *\n{import_lines}",
            )
        )

        # With modules, we need to add globals for the input parameters
        for input in config.api.inputs:
            code = f"{input.name} = {input.default_value}"
            nb.cells.append(nbf.v4.new_code_cell(code))

        # Add code cells for each step in the dataflow graph
        for step in dfg.topological_sort():
            node = dfg[step]
            assert node.function_name, f"Node {node.id} must have a signature"
            md_cell = nbf.v4.new_markdown_cell(f"### {node.pill}")
            nb.cells.append(md_cell)

            code_cell = nbf.v4.new_code_cell(node.code_str(), id=node.id)
            nb.cells.append(code_cell)

        self.notebook = nb
        self.add_json_encoding()

        self.file_name = self.config.file_name
        self.write()

    def write(self):
        with open(self.file_name, "w") as f:
            nbf.write(self.notebook, f)

        prefix = self.file_name.replace(".json", "")
        html_file = f"{prefix}.html"
        with open(html_file, "w") as f:
            print(self.as_html(), file=f)

    T = TypeVar("T")

    def listify(self, input_element: Union[T, List[T]]) -> List[T]:
        if isinstance(input_element, list):
            return input_element
        else:
            return [input_element]

    def run(self) -> Dict[str, List[Dict[str, Any]]]:
        outputs = {}
        for cell_id, result in self:
            if Stoppable.should_stop():
                break
            outputs[cell_id] = result
        return outputs

    def __iter__(self):
        with Sandbox(self.config.data_files) as sandbox:
            client = NotebookClient(self.notebook, kernel_name="python3")
            with client.setup_kernel(cwd=sandbox.get_sandbox_path()):
                for i, cell in enumerate(self.notebook.cells):
                    result = None
                    try:
                        if self.pre_execute(sandbox, cell, i):
                            result = client.execute_cell(cell, i)
                            id = self.post_execute(sandbox, cell, i, result)
                            if id:
                                yield (id, self.listify(result.get("outputs", [])))
                    except CellExecutionError as e:
                        self.write()
                        id = self.post_execute(sandbox, cell, i, result)
                        if id:
                            if result is not None:
                                result_outputs = self.listify(result.get("outputs", []))
                            else:
                                result_outputs = []
                            outputs = self.listify(
                                [{"output_type": "stream", "text": str(e)}]
                                + result_outputs
                            )
                            yield (id, outputs)
                        else:
                            outputs = None
                        raise DataFlowNotebookExecutionError(
                            e.traceback, e.ename, e.evalue, cell, outputs
                        ) from None
                self.write()

    def pre_execute(self, sandbox, cell, i):
        """
        Return True if the cell should be executed, False otherwise.
        Any other pre steps.
        """
        return True

    def post_execute(self, sandbox, cell, i, result) -> Optional[str]:
        """
        Return id of the name to store the outputs under, or None if the outputs should not be stored.
        """
        return cell.id

    def add_json_encoding(self):
        md_cell = nbf.v4.new_markdown_cell("# Pickle Encoding")
        self.notebook.cells.append(md_cell)
        code_cell = nbf.v4.new_code_cell(
            textwrap.dedent(
                """\
            import pickle
            import base64
            def encode(value: Any) -> str:
                # Serialize the object using pickle
                pickled_bytes = pickle.dumps(value)
                # Encode the bytes to a base64 string
                encoded_str = base64.b64encode(pickled_bytes).decode('utf-8')
                return encoded_str
            """
            )
        )
        self.notebook.cells.append(code_cell)

    def as_html(self):
        exporter = HTMLExporter()
        body, _ = exporter.from_notebook_node(self.notebook)
        return body


class DataflowNotebookWithDriver(BaseDataflowNotebook):
    def __init__(self, config: NotebookConfig):
        super().__init__(config)

        md_cell = nbf.v4.new_markdown_cell("# Driver")
        self.notebook.cells.append(md_cell)

        # Build map from node id to the cell so we can extract outputs later.
        self.node_outputs = []
        driver = self.config.dfg.make_driver()
        for call in driver:
            node_id = call.node_id
            code = f"{call.result} = {call.function_name}({','.join(call.arguments)})\nencode({call.result})"
            code_cell = nbf.v4.new_code_cell(code, id=f"driver_{node_id}")
            self.notebook.cells.append(code_cell)
            self.node_outputs += [f"driver_{node_id}"]

    def pre_execute(self, sandbox, cell, i):
        """
        Return True if the cell should be executed, False otherwise.
        Any other pre steps.
        """
        return True

    def post_execute(self, sandbox, cell, i, result):
        # print(cell.id, i, result)
        return cell.id.split("_")[1] if cell.id in self.node_outputs else None


class DataflowNotebookTest(BaseDataflowNotebook):
    def __init__(
        self,
        config: NotebookConfig,
        test: TestCase,
    ):
        super().__init__(config)

        md_cell = nbf.v4.new_markdown_cell(f"### Test Code")
        self.notebook.cells.append(md_cell)

        for i, code in enumerate(test.get_code()):
            code_cell = nbf.v4.new_code_cell(code, id=f"test_cell_{i}")
            self.notebook.cells.append(code_cell)

    def pre_execute(self, sandbox, cell, i):
        return True

    def post_execute(self, sandbox, cell, i, result):
        return cell.id if cell.id == "test_cell_0" else None

    @staticmethod
    def run_test(
        config: NotebookConfig,
        test: TestCase,
    ) -> None:  # NodeOutputs:
        with logger("Generating notebook"):
            nb = DataflowNotebookTest(
                config,
                test,
            )
        with logger("Running notebook"):
            outputs = nb.run()
            if outputs is None:
                return NodeOutputs()
            else:
                return NodeOutputs.from_outputs(outputs.get("test_cell_0", []))

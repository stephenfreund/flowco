import inspect
import textwrap
import io
import base64
import traceback
from typing import List
from PIL import Image
import nbformat as nbf
from nbclient import NotebookClient, exceptions as nb_exceptions

from flowco.builder.type_ops import encode
from flowco.dataflow.dfg import Node, DataFlowGraph
from flowco.page.output import NodeResult, OutputType, ResultOutput, ResultValue
from flowco.page.tables import GlobalTables
from flowco.pythonshell.sandbox import Sandbox
from flowco.util.output import debug, error, logger
from pydantic import BaseModel


class EvalResult(BaseModel):
    stdout: str | None
    outputs: List[str]
    plot: str | None

    def as_result_output(self) -> ResultOutput | None:
        if self.plot is not None:
            return ResultOutput(output_type=OutputType.image, data=self.plot)
        elif self.stdout is not None:
            return ResultOutput(output_type=OutputType.text, data=self.stdout)
        elif self.outputs is not None:
            return ResultOutput(
                output_type=OutputType.text, data="\n".join(self.outputs)
            )
        # elif text.strip() != "None" and len(dfg.successors(node.id)) == 0:
        #     output = ResultOutput(output_type=OutputType.text, data=text)
        #     # log(f"Captured text output for node '{node.id}'.")
        else:
            return None


class PythonShell:
    def __init__(self, tables: GlobalTables):
        self.sandbox = Sandbox()
        self.sandbox_dir = self.sandbox.get_sandbox_path()
        self.tables = tables
        self.nb = nbf.v4.new_notebook()

        self.client = NotebookClient(
            self.nb,
            kernel_name="python3",
            timeout=600,
            kernel_manager_kwargs={"path": self.sandbox_dir},
        )

        # log(f"Initializing NotebookClient with sandbox directory: {self.sandbox_dir}")

        try:
            self.client.create_kernel_manager()
            self.client.start_new_kernel()
            self.client.start_new_kernel_client()

            try:
                # Retrieve the source code of the 'encode' function
                encode_src = inspect.getsource(encode)
            except Exception as e:
                error(f"Error retrieving source of 'encode': {e}")
                encode_src = ""

            # Define the import statements to be executed once
            import_code = textwrap.dedent(
                """\
                import pandas as pd
                import numpy as np
                import matplotlib.pyplot as plt
                import seaborn as sns
                import sklearn
                import scipy
                import pickle
                import base64
                from io import StringIO                          
                from typing import *
                
                import warnings
                warnings.simplefilter('ignore')

                import logging
                logging.disable(logging.ERROR)
                
                %matplotlib inline
            """
            )

            # Execute the import statements
            self.run_cell(import_code)
            # log("Imported necessary libraries.")

            for table_def in self.tables.function_defs():
                self.run_cell(table_def)
            # log("Loaded tables into the kernel.")

            # Execute the encode function source code
            if encode_src:
                self.run_cell(encode_src)
                # log("Defined 'encode' function in the kernel.")

        except nb_exceptions.CellExecutionError as cee:
            error(f"Cell execution error during kernel setup: {cee}")
            self.close()
            raise
        except Exception as e:
            error(f"Unexpected error during kernel setup: {e}")
            self.close()
            raise

    def eval_code(self, code: str) -> EvalResult:
        execution_result = self.run_cell(code)
        stdout = ""
        outputs = []
        images = []

        for msg in execution_result.get("outputs", []):
            msg_type = msg.get("output_type", "")

            if msg_type == "execute_result":
                data = msg.get("data", {})
                if "text/plain" in data:
                    text = data["text/plain"]
                    outputs += [text]  # Capture the first execute_result

            elif msg_type == "stream":
                stdout += msg.get("text", "")

            elif msg_type == "display_data":
                data = msg.get("data", {})
                if "image/png" in data:
                    base64_image_data = "data:image/png;base64," + data["image/png"]
                    try:
                        image_binary = base64.b64decode(base64_image_data.split(",")[1])
                        images.append(Image.open(io.BytesIO(image_binary)))
                    except Exception as e:
                        error(f"Error decoding image: {e}")

        # Combine all images vertically into one image
        if images:
            # Calculate the total height and maximum width
            total_height = sum(img.height for img in images)
            max_width = max(img.width for img in images)

            # Create a new blank image with the calculated dimensions
            combined_image = Image.new("RGB", (max_width, total_height), color="white")

            # Paste each image into the combined image
            current_y = 0
            for img in images:
                combined_image.paste(img, (0, current_y))
                current_y += img.height

            # Save the combined image to a bytes buffer
            combined_buf = io.BytesIO()
            combined_image.save(combined_buf, format="PNG")
            combined_buf.seek(0)
            combined_image_data = combined_buf.read()

            # Encode the combined image as a base64 string
            combined_base64_str = "data:image/png;base64," + base64.b64encode(
                combined_image_data
            ).decode("utf-8")

            # Assign to plot (singular) key
            plot = combined_base64_str
        else:
            plot = None  # No plots were generated

        return EvalResult(stdout=stdout, outputs=outputs, plot=plot)

    def run_cell(self, code: str) -> nbf.NotebookNode:
        """
        Run code as a cell in the existing NotebookClient.
        """
        cell = nbf.v4.new_code_cell(code)
        self.nb.cells.append(cell)
        index = len(self.nb.cells) - 1
        try:
            debug(f"Executing cell {index}")
            debug(f"Cell code: {code}")
            result = self.client.execute_cell(cell, index, store_history=False)
            return result
        except nb_exceptions.CellExecutionError as cee:
            error(f"Error executing cell {index}: {cee}")
            raise cee
        except Exception as e:
            error(f"Unexpected error executing cell {index}: {e}")
            raise e

    def eval_node(self, dfg: DataFlowGraph, node: Node) -> NodeResult:
        """
        Evaluate a Node object and capture outputs.

        Args:
            dfg (DataFlowGraph): The data flow graph containing node dependencies.
            node (Node): The Node object to evaluate.

        Returns:
            NodeResult: The result of evaluating the node.
        """
        with logger(f"Evaluating node '{node.id}'"):
            try:
                # Prepare arguments from predecessor nodes
                arguments = []
                for predecessor_id in node.predecessors:
                    predecessor = dfg[predecessor_id]
                    with logger(f"Retrieving predecessor result for {predecessor_id}"):
                        repr_val, _ = predecessor.result.result.to_repr()
                        self.run_cell(f"{predecessor.function_result_var} = {repr_val}")
                        arguments.append(predecessor.function_result_var)

                # Execute the node's code
                self.run_cell("\n".join(node.code))

                # Construct and execute the function call
                code = f"{node.function_result_var} = {node.function_name}({', '.join(arguments)})"
                result = self.eval_code(code)

                # Retrieve the string representation and encoded pickle of the result
                text = self.eval_code(f"print(str({node.function_result_var}))").stdout
                pickle_result = self.eval_code(
                    f"print(encode({node.function_result_var}))"
                ).stdout

                output = result.as_result_output()

                return NodeResult(
                    result=ResultValue(text=text, pickle=pickle_result), output=output
                )

            except Exception as e:
                error(f"Error evaluating node '{node.id}': {e}")
                error(traceback.format_exc())
                raise (e)

    def close(self):
        """
        Gracefully shuts down the kernel.
        """
        try:
            if self.client is not None:
                self.client.kc.shutdown()
            if self.sandbox is not None:
                self.sandbox.cleanup()
        except Exception as e:
            print(f"Error shutting down kernel: {e} {traceback.format_exc()}")

    def __del__(self):
        """
        Ensures the kernel is shut down when the PythonShell instance is destroyed.
        """
        self.close()

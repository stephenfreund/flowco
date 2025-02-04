import inspect
import json
import os
import re
from tempfile import TemporaryDirectory
import textwrap
import io
import base64
import uuid
from typing import Any, Dict, List
from PIL import Image
import nbformat as nbf
from nbclient import NotebookClient, exceptions as nb_exceptions

from flowco.assistant.flowco_assistant import flowco_assistant
from flowco.builder.type_ops import convert_np_float64, decode, encode
from flowco.dataflow.checks import CheckOutcomes, QualitativeCheck
from flowco.dataflow.dfg import Node, DataFlowGraph
from flowco.page.output import NodeResult, OutputType, ResultOutput, ResultValue
from flowco.page.tables import GlobalTables
from flowco.pythonshell.sandbox import Sandbox
from flowco.util.output import debug, error, log, logger
from pydantic import BaseModel

from flowco.util.text import strip_ansi

from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam,
    ImageURL,
)

from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionContentPartParam,
)


class EvalResult(BaseModel):
    stdout: str | None
    outputs: List[str]
    plot: str | None

    def as_result_output(self) -> ResultOutput | None:
        if self.plot is not None:
            return ResultOutput(output_type=OutputType.image, data=self.plot)
        elif self.stdout is not None and len(self.stdout) > 0:
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
    def __init__(self):
        self.sandbox = Sandbox()
        self.sandbox_dir = self.sandbox.get_sandbox_path()
        self.nb = nbf.v4.new_notebook()

        self.client = NotebookClient(
            self.nb,
            kernel_name="python3",
            timeout=600,
            kernel_manager_kwargs={"path": self.sandbox_dir},
        )

        try:
            self.client.create_kernel_manager()
            self.client.start_new_kernel()
            self.client.start_new_kernel_client()
            self._init()
        except nb_exceptions.CellExecutionError as cee:
            debug(f"Cell execution error during kernel setup:", cee)
            self.close()
            raise
        except Exception as e:
            debug(f"Unexpected error during kernel setup:", e)
            self.close()
            raise

    def _init(self):
        # Define the import statements to be executed once
        import_code = textwrap.dedent(
            """\
            import pandas as pd
            import numpy as np
            from numpy import nan
            import matplotlib.pyplot as plt
            import seaborn as sns
            import sklearn
            import scipy
            import pickle
            import base64
            from io import StringIO                          
            from typing import *
            import pprint
            
            import warnings
            warnings.simplefilter('ignore')

            import logging
            logging.disable(logging.ERROR)
            
            %matplotlib inline
        """
        )
        self._run_cell(import_code)

        imported_functions = [encode, decode, convert_np_float64]
        for function in imported_functions:
            src = inspect.getsource(function)
            self._run_cell(src)

    def reset(self) -> None:
        """
        Reset the PythonShell to its initial state.
        """
        self.sandbox.restore()

    def run(self, code: str) -> EvalResult:
        execution_result = self._run_cell(code)
        stdout = ""
        outputs = []
        images = []

        # print(execution_result.get("outputs", []))

        for msg in execution_result.get("outputs", []):
            msg_type = msg.get("output_type", "")

            if msg_type == "error":
                raise ValueError(f"Could not evaluate {code}")

            elif msg_type == "execute_result":
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
                        error(f"Error decoding image", e)

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

    def _run_cell(self, code: str) -> nbf.NotebookNode:
        """
        Run code as a cell in the existing NotebookClient.
        """
        cell = nbf.v4.new_code_cell(code)
        self.nb.cells.append(cell)
        index = len(self.nb.cells) - 1
        # debug(f"Executing cell {index}")
        # log(f"Cell code {index}:\n", code)
        # with logger(f"Executing cell {index}"):
        #     log("Code: ", code[:40])
        result = self.client.execute_cell(cell, index, store_history=False)
        return result

    def load_tables(self, tables: GlobalTables):
        """
        Load table definitions into the PythonShell.
        """
        # for table_def in tables.function_defs():
        #     self._run_cell(table_def)
        function_defs = []
        for file_path in tables.tables:
            with logger(f"Loading table '{file_path}'"):
                with logger("Reading file"):
                    content = tables.table_contents(file_path)

                with logger("Writing to Sandbox"):
                    file_name = self.sandbox.temporary_file(file_path)
                    with open(file_name, "w") as f:
                        f.write(content)

                function_defs += [
                    f"def {tables.table_name(file_path)}_table() -> pd.DataFrame:\n    return pd.read_csv('{file_name}')\n"
                ]

        with logger("Running cell"):
            self._run_cell("\n".join(function_defs))

    def load_parameters(self, dfg: DataFlowGraph, node: Node) -> List[str]:
        # Prepare arguments from predecessor nodes
        arguments = []
        code = []

        with TemporaryDirectory(dir=self.sandbox_dir) as temp_dir:
            temp_files = []

            for predecessor_id in node.predecessors:
                predecessor = dfg[predecessor_id]
                with logger(f"Retrieving predecessor result for {predecessor_id}"):
                    with logger("Loading predecessor result into PythonShell"):
                        with logger("Getting value"):
                            # Create temp file within the temporary directory
                            temp_file_path = os.path.join(
                                temp_dir, uuid.uuid4().hex + ".pickle"
                            )
                            with open(temp_file_path, "wb") as f:
                                assert predecessor.result is not None
                                assert predecessor.result.result is not None
                                # log(
                                #     "Type of pickle:",
                                #     type(decode(predecessor.result.result.pickle)),
                                # )
                                f.write(predecessor.result.result.pickle.encode())

                            # Prepare the code to decode the temp file
                            code.append(
                                f"{predecessor.function_result_var} = decode(open('{temp_file_path}', 'rb').read().decode())"
                            )

                    with logger("Appending argument"):
                        arguments.append(predecessor.function_result_var)

            with logger("Running cell to load parameters"):
                code_str = "\n".join(code)
                # log(f"Code to load parameters:\n{code_str}")
                self._run_cell(code_str)
            # TemporaryDirectory and its contents are automatically cleaned up here

        return arguments

    def load_result_var(self, node: Node):
        """
        Load the result variable from the previous node into the PythonShell.
        """
        if node.result is not None and node.result.result is not None:
            repr_val, _ = node.result.result.to_repr()
            self._run_cell(f"{node.function_result_var} = {repr_val}")
        else:
            self._run_cell(f"{node.function_result_var} = None")

    def run_node(
        self, tables: GlobalTables, dfg: DataFlowGraph, node: Node
    ) -> NodeResult:
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
                with logger("Loading tables"):
                    self.load_tables(tables)
                with logger("Loading parameters"):
                    arguments = self.load_parameters(dfg, node)

                with logger("Running node code"):
                    self._run_cell("\n".join(node.code or []))

                with logger("Calling function"):
                    # Construct and execute the function call
                    code = f"{node.function_result_var} = {node.function_name}({', '.join(arguments)})"
                    result = self.run(code)

                with logger("Capturing output"):
                    text = self.run(
                        f"pprint.pp(convert_np_float64({node.function_result_var}), indent=2, compact=False, sort_dicts=True, underscore_numbers=True)"
                    ).stdout
                with logger("Capturing pickle"):
                    pickle_result = self.run(
                        f"print(encode({node.function_result_var}))"
                    ).stdout

                    output = result.as_result_output()

                return NodeResult(
                    result=ResultValue(text=text, pickle=pickle_result), output=output
                )

            except Exception as e:
                error(f"Error evaluating node '{node.id}'", e)
                raise (e)

    def extract_assertion_error_message(self, error_log):
        """
        Extracts the message following 'AssertionError:' from a given error log string.

        Parameters:
        error_log (str): The multiline string containing the error message.

        Returns:
        str or None: The extracted assertion error message, or None if not found.
        """
        # Define a regex pattern to find 'AssertionError:' followed by the message
        pattern = r"AssertionError:\s*(.*)"

        # Search for the pattern in the error_log
        match = re.search(pattern, error_log)

        if match:
            # Return the captured group which is the error message
            return match.group(1).strip()
        else:
            # Return None or an appropriate message if not found
            return None

    def make_context(self, dfg, node: Node) -> Dict[str, Any]:
        context = {}
        for predecessor_id in node.predecessors:
            predecessor = dfg[predecessor_id]
            context[predecessor.function_result_var] = predecessor.result.result

        if node.result is not None and node.result.result is not None:
            context[node.function_result_var] = node.result.result
        return context

    def _inspect_output(
        self, node: Node, qualitative_checks: Dict[str, QualitativeCheck]
    ) -> Dict[str, Any]:

        if len(qualitative_checks) == 0:
            return {}

        if node.result is not None:
            result_messages = node.result.to_content_parts()
        else:
            result_messages = []

        class InspectionCompletion(BaseModel):
            requirement: str
            error: bool
            message: str | None

        class InspectionsCompletion(BaseModel):
            errors: List[InspectionCompletion]

        requirements = [check.requirement for check in qualitative_checks.values()]

        assistant = flowco_assistant(
            prompt_key="inspect-output",
            requirements=json.dumps(requirements, indent=2),
        )
        assistant.add_text("user", "Here is the output.")
        assistant.add_content_parts("user", result_messages)

        completion = assistant.model_completion(InspectionsCompletion)
        messages = {
            x.requirement: x.message if x.error else None for x in completion.errors
        }
        return {
            assertion: messages[check.requirement]
            for assertion, check in qualitative_checks.items()
        }

    def run_assertions(
        self, tables: GlobalTables, dfg: DataFlowGraph, node: Node
    ) -> Node:
        """
        Evaluate a Node object and capture outputs.

        Args:
            dfg (DataFlowGraph): The data flow graph containing node dependencies.
            node (Node): The Node object to evaluate.
        """

        with logger(f"Evaluating assertions for node '{node.id}'"):
            try:
                self.load_tables(tables)
                arguments = self.load_parameters(dfg, node)
                self.load_result_var(node)

                context = self.make_context(dfg, node)
                outcomes = {}
                assert node.assertion_checks is not None, "No assertion checks found."
                for assertion, check in node.assertion_checks.items():
                    if check.type == "quantitative":
                        with logger(f"Evaluating quantitative assertion {assertion}"):
                            assertion_message = None
                            try:
                                self._run_cell("\n".join(check.code))
                            except nb_exceptions.CellExecutionError as e:
                                if e.ename == "AssertionError":
                                    assertion_message = (
                                        self.extract_assertion_error_message(
                                            strip_ansi(e.traceback)
                                        )
                                    )
                            except Exception as e:
                                raise e
                            log(assertion_message)
                            outcomes[assertion] = assertion_message

                with logger(f"Evaluating qualitative assertions"):
                    qualitative_checks = {
                        assertion: check
                        for assertion, check in node.assertion_checks.items()
                        if check.type == "qualitative"
                    }
                    assertion_messages = self._inspect_output(node, qualitative_checks)
                    outcomes = outcomes | assertion_messages

                node = node.update(
                    assertion_outcomes=CheckOutcomes(outcomes=outcomes, context=context)
                )
                return node

            except Exception as e:
                error(f"Error evaluating node '{node.id}'", e)
                raise (e)

    def run_count(self):
        """
        Return the number of cells executed in the current NotebookClient.
        """
        return len(self.nb.cells)

    async def restart(self):
        self.nb.cells = []
        assert self.client is not None, "No kernel client found."
        assert self.client.km is not None, "No kernel manager found."
        self.client.km.restart_kernel(now=True)
        self._init()

    def close(self):
        """
        Gracefully shuts down the kernel.
        """
        try:
            if self.client is not None:
                if self.client.kc is not None:
                    self.client.kc.shutdown()
            if self.sandbox is not None:
                self.sandbox.cleanup()
        except Exception as e:
            print(f"Error shutting down kernel:", e)

    def __del__(self):
        """
        Ensures the kernel is shut down when the PythonShell instance is destroyed.
        """
        self.close()

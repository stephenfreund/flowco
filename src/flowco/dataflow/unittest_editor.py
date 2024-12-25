from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple
import uuid
from pydantic import BaseModel, Field
from flowco.assistant.assistant import Assistant
from flowco.builder.type_ops import encode, decode
from flowco.builder.build import PassConfig
from flowco.dataflow.extended_type import ExtendedType
from flowco.dataflow.dfg import (
    DataFlowGraph,
    GraphLike,
    Node,
    UnitTest,
    UnitTestValue,
)
from flowco.page.diagram import Diagram
from flowco.page.output import NodeOutputs
from flowco.builder.fine_grained_passes import unit_tests
from flowco.util.config import config
import json


class UnitTestEditor(BaseModel):
    graph: DataFlowGraph
    node: Node
    unit_test: UnitTest

    @staticmethod
    def create(graph: DataFlowGraph, node: Node) -> UnitTestEditor:
        assert (
            node.function_parameters is not None
        ), "Node must have function parameters to create a unit test."
        assert (
            node.function_return_type is not None
        ), "Node must have a function return type to create a unit test."

        # construct reasonable defaults for the inputs and expected output

        inputs = {
            param.name: UnitTestValue(
                type=param.type,
                spec="",
                code=f"{param.type.default_value()}",
                value=encode(param.type.default_value()),
            )
            for param in node.function_parameters
        }

        expected = UnitTestValue(
            type=node.function_return_type,
            spec="",
            code=f"{node.function_return_type.default_value()}",
            value=encode(node.function_return_type.default_value()),
        )

        unit_test = UnitTest(
            uuid=str(uuid.uuid4()),
            requirement="New Test",
            inputs=inputs,
            expected=expected,
        )
        return UnitTestEditor(graph=graph, node=node, unit_test=unit_test)

    def inputs(self) -> List[Tuple[str, UnitTestValue]]:
        """
        Returns the inputs in sorted order
        """
        return sorted(self.unit_test.inputs.items())

    def refresh(self) -> None:
        graph = self.graph
        node = self.node
        preconditions = {
            graph[pred].function_result_var: {
                "type": graph[pred].function_return_type,
                "requirements": graph[pred].requirements
                + [self.unit_test.inputs[graph[pred].function_result_var].spec],
            }
            for pred in node.predecessors
        }

        print(preconditions)

        assistant = Assistant(
            "unit-test-inputs",
            goal=self.unit_test.requirement,
            requirements=preconditions,
        )

        class Result(BaseModel):
            code: str = Field(description="The code that generates the input data.")

        result = assistant.model_completion(Result)
        print(result)
        globals = dict()
        exec(result.code, globals)
        result = globals["create_input"]()
        for key, value in result.items():
            self.update_input(key, value=encode(value))

    def update_input(self, input_name: str, **kwargs) -> None:
        input = self.unit_test.inputs[input_name]
        input = input.update(**kwargs)
        print("WOOF", input_name, input.value)
        self.unit_test = self.unit_test.update(
            inputs={**self.unit_test.inputs, input_name: input}
        )
        print("WOOF", self.unit_test.inputs[input_name])
        print("WOOF", self.unit_test)

    def get_input(self, input_name: str) -> UnitTestValue:
        return self.unit_test.inputs[input_name]

    def update_expected(self, **kwargs) -> None:
        expected = self.unit_test.expected.model_copy(update=kwargs)
        self.unit_test = self.unit_test.update(expected_output=expected)

    def get_expected(self) -> UnitTestValue:
        return self.unit_test.expected

    # def update_spec(self, node: Node, unit_test: UnitTest, spec: Dict[str, Any]) -> UnitTest:

    # def update(self, node: Node, unit_test: UnitTest, parameters: Optional[List[str]] = None, expected_value = False) -> UnitTest:
    #     pass


if __name__ == "__main__":
    import json
    import sys
    import os

    node = Node.model_validate_json(
        """
    {
        "id": "Step-1",
        "pill": "Fortis",
        "label": "Select Fortis",
        "function_name": "compute_fortis",
        "function_result_var": "fortis_result",
        "predecessors": [
            "Step-5"
        ],
        "phase": 5,
        "hints": {
            "requirements": "",
            "locked_requirements": [],
            "algorithm": "",
            "locked_algorithm": null,
            "sanity_checks": [],
            "unit_tests": []
        },
        "description": "This computation step selects the rows from the DataFrame that correspond to the species 'fortis'. It filters the data based on the 'species' column and retains only the relevant rows for further analysis. The output will be a DataFrame containing only the 'fortis' entries from the original dataset, ready for subsequent calculations.",
        "requirements": [
            "The DataFrame must contain the columns: 'species', 'Beak length, mm', and 'Beak depth, mm'.",
            "The DataFrame should have a number of rows that is less than or equal to the number of rows in the original DataFrame.",
            "No missing values (NA) should be present in any of the columns of the resulting DataFrame.",
            "The resulting DataFrame should contain only rows where the 'species' value is 'fortis'."
        ],
        "algorithm": "",
        "function_parameters": [
            {
            "name": "finch_data_result",
            "type": {
                "type": "pd.DataFrame[{\\"species\\": \\"str\\", \\"Beak length, mm\\": \\"float\\", \\"Beak depth, mm\\": \\"float\\"}]"
            }
            }
        ],
        "function_return_type": {
            "type": "pd.DataFrame[{\\"species\\": \\"str\\", \\"Beak length, mm\\": \\"float\\", \\"Beak depth, mm\\": \\"float\\"}]"
        },
        "code": [
            "import pandas as pd",
            "def compute_fortis(finch_data_result: pd.DataFrame) -> pd.DataFrame:",
            "    # Assert preconditions",
            "    assert 'species' in finch_data_result.columns",
            "    assert 'Beak length, mm' in finch_data_result.columns",
            "    assert 'Beak depth, mm' in finch_data_result.columns",
            "    assert finch_data_result.shape[0] <= 407",
            "    # Clean data by removing rows with NA values",
            "    finch_data_result = finch_data_result.dropna(subset=['species', 'Beak length, mm', 'Beak depth, mm'])",
            "    # Filter for species 'fortis'",
            "    finch_fortis = finch_data_result[finch_data_result['species'] == 'fortis']",
            "    # Assert postconditions",
            "    assert finch_fortis.shape[0] <= finch_data_result.shape[0]",
            "    assert finch_fortis.isnull().sum().sum() == 0",
            "    return finch_fortis"
        ],
        "sanity_checks": null,
        "unit_tests": null,
        "sanity_check_results": {},
        "unit_test_results": {},
        "warnings": null
    }
"""
    )
    node2 = Node.model_validate_json(
        """
      {
        "id": "Step-5",
        "pill": "Finch-Data",
        "label": "Load Finch Data from finch_beaks_1975.csv ",
        "function_name": "compute_finch_data",
        "function_result_var": "finch_data_result",
        "predecessors": [],
        "phase": 6,
        "hints": {
          "requirements": "",
          "locked_requirements": [],
          "algorithm": "",
          "locked_algorithm": null,
          "sanity_checks": [],
          "unit_tests": []
        },
        "description": "This computation step involves loading the Finch data from the specified CSV file, `finch_beaks_1975.csv`. The process includes reading the data into a pandas DataFrame, ensuring that all values are properly formatted as numerical data, and identifying and handling any missing or bad values. The cleaned DataFrame will contain three columns: 'species', 'Beak length, mm', and 'Beak depth, mm'.",
        "requirements": [
          "'Beak length, mm' and 'Beak depth, mm' must be of type float and should not contain any obviously bad values (e.g., negative values).",
          "The 'species' column should contain string values representing the species name.",
          "No missing values (NA) should be present in any of the columns of the DataFrame."
        ]
      }
                                     """
    )
    dfg = DataFlowGraph(nodes=[node, node2], edges=[])

    def merge_graph(dfg: GraphLike):
        pass

    def merge_output(outputs: Dict[str, List[Dict[str, Any]]]):
        pass

    pass_config = PassConfig(
        page_name="moo.json",
        spec_files=[],
        diagram=Diagram(),
        partial_updater=merge_graph,
        output_reporter=merge_output,
        max_retries=0,
    )

    if True:
        editor = UnitTestEditor.create(dfg, node)
        print(editor.unit_test)
        editor.unit_test = editor.unit_test.update(
            requirement="Only 'fortis' finches in output"
        )
        config.debug = True
        editor.refresh()

        # print()
        # unit_tests = unit_tests(
        #     pass_config,
        #     dfg,
        #     node
        # )
        # print(unit_tests)
    elif False:
        unit_test = UnitTest.model_validate_json(
            """{
            "uuid": "44a6b018-4469-4888-8e55-abe90146ed42",
            "requirement": "The DataFrame must contain the columns: 'species', 'Beak length, mm', and 'Beak depth, mm'.",
            "inputs": {
                "finch_data_result": {
                "spec": "",
                "value": "pd.DataFrame({'species': ['fortis', 'fortis', 'not_fortis'], 'Beak length, mm': [10.0, 15.0, 12.0], 'Beak depth, mm': [5.0, 6.0, 4.5]})"
                }
            },
            "expected_output": {
                "spec": "",
                "value": "pd.DataFrame({'species': ['fortis', 'fortis'], 'Beak length, mm': [10.0, 15.0], 'Beak depth, mm': [5.0, 6.0]})"
            }
            }"""
        )

        requirements = [
            "'Beak length, mm' and 'Beak depth, mm' must be of type float and should not contain any obviously bad values (e.g., negative values).",
            "The 'species' column should contain string values representing the species name.",
            "No missing values (NA) should be present in any of the columns of the DataFrame.",
        ]

        if len(sys.argv) > 1:
            requirements += " ".join(sys.argv[1:])
        else:
            requirements += [
                "Create 10 rows where the length is 10 and depth is between 5 and 15.  Species should be 'fortis' or 'scandens'."
            ]

        field_type = ExtendedType(
            type="pd.DataFrame[{'species': 'str', 'Beak length, mm': 'float', 'Beak depth, mm': 'float'}]"
        )

        config.model = "gpt-4o"
        assistant = Assistant(
            "unit-test-input", preconditions=requirements, type=field_type
        )

        class Result(BaseModel):
            input_code: str = Field(
                description="The code that generates the input data."
            )

        result = assistant.model_completion(Result)
        print(result)
        globals = dict()
        exec(result.input_code, globals)
        print(globals["create_input"]())
    else:
        unit_test = UnitTest.model_validate_json(
            """{
            "uuid": "44a6b018-4469-4888-8e55-abe90146ed42",
            "requirement": "whatever",
            "inputs": {
                "x": {
                    "spec": "",
                    "value": "0"
                },
                "y": {
                    "spec": "",
                    "value": "1"
                }
            },
            "expected_output": {
                "spec": "",
                "value": "pd.DataFrame({'species': ['fortis', 'fortis'], 'Beak length, mm': [10.0, 15.0], 'Beak depth, mm': [5.0, 6.0]})"
            }
        }"""
        )

        requirements = {
            "x": {
                "type": "List[str]",
                "requirements": ["x is a list of countries.", "Create 3 elements"],
            },
            "y": {
                "type": "pd.DataFrame[{'country': 'str', 'population': 'int'}]",
                "requirements": [
                    "the populations must be non-negative.",
                    "The countries must be present in the list in x.  Create 20",
                ],
            },
        }

        config.model = "gpt-4o"
        assistant = Assistant("unit-test-inputs", requirements=requirements)

        class Result(BaseModel):
            input_code: str = Field(
                description="The code that generates the input data."
            )

        result = assistant.model_completion(Result)
        print(result)
        globals = dict()
        exec(result.input_code, globals)
        print(globals["create_input"]())

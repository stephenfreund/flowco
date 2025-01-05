from __future__ import annotations

from typing import Iterable, List, Literal, Tuple
from flowco.assistant.openai import OpenAIAssistant
from flowco.assistant.stream import StreamingAssistantWithFunctionCalls
from flowco.builder.graph_completions import messages_for_graph
from flowco.dataflow.dfg import Geometry
from flowco.dataflow.extended_type import ExtendedType
from flowco.dataflow.phase import Phase
from flowco.page.page import Page
from flowco.pythonshell.shells import PythonShells
from flowco.session.session import session
from flowco.util.config import config
from flowco.util.errors import FlowcoError
from flowco.util.output import error, log, logger
from pydantic import BaseModel


from flowco.dataflow.dfg_update import (
    mxDiagramUpdate,
    DiagramEdgeUpdate,
    DiagramNodeUpdate,
    update_dataflow_graph,
)


class VisibleMessage(BaseModel):
    role: str
    content: str


class QuestionKind(BaseModel):
    kind: Literal["Explain"] | Literal["Modify"]


ReturnType = Tuple[str, str | None]


class AskMeAnything:

    def __init__(self, page: Page):
        self.page = page
        self.assistant = StreamingAssistantWithFunctionCalls(
            [],
            ["system-prompt", "ama_general"],
            imports="",
        )
        self.shell = None
        self.completion_dfg = None
        self.visible_messages = []

    def python_eval(self, code: str) -> ReturnType:
        """
        {
            "name": "python_eval",
            "description": "Exec python code.  You may assume numpy, scipy, and pandas are available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to evaluate"
                    }
                },
                "required": ["code"]
            }
        }
        """
        init_code = ""
        for node in self.page.dfg.nodes:
            result = node.result
            if (
                result is not None
                and result.result is not None
                and node.function_return_type is not None
                and not node.function_return_type.is_None_type()
            ):
                value, _ = result.result.to_repr()
                init_code += f"{node.function_result_var} = {value}\n"

        init_code += "\n".join(self.page.tables.function_defs())

        result = session.get("shells", PythonShells).run(init_code + "\n" + code)
        result_output = result.as_result_output()
        return "Finished running code", result_output.to_prompt()

    def inspect(self, id: str) -> ReturnType:
        """
        {
            "name": "inspect",
            "description": "Inspect the output for a node in the diagram, including any generated plots.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The id of the node to inspect"
                    }
                },
                "required": ["id"]
            }
        }
        """
        log(f"inspect: {id}")
        if id not in self.page.dfg.node_ids():
            return (f"Node {id} does not exist", None)
        node = self.page.dfg[id]
        result = node.result
        if result is None:
            return (f"Node {node.pill} has no result right now", None)
        else:
            if result.output is not None:
                return (
                    f"Inspected the output for {node.pill}",
                    result.output.to_prompt(),
                )
            else:
                return (
                    f"Inspected the result for {node.pill}",
                    result.result.to_prompt(),
                )

    if config.x_algorithm_phase:

        def update_node(
            self,
            id: str,
            pill: str | None = None,
            label: str | None = None,
            requirements: List[str] | None = None,
            function_return_type: ExtendedType | None = None,
            algorithm: List[str] | None = None,
            code: List[str] | None = None,
        ) -> ReturnType:
            """\
            {"name": "update_node",
            "strict": true,
            "parameters": {"$defs": {"AnyType": {"properties": {"type": {"const": "Any", "title": "Type", "type": "string"
                                }
                            }, "required": ["type"
                            ], "title": "AnyType", "type": "object", "additionalProperties": false
                        }, "BoolType": {"properties": {"type": {"const": "bool", "title": "Type", "type": "string"
                                }
                            }, "required": ["type"
                            ], "title": "BoolType", "type": "object", "additionalProperties": false
                        }, "ExtendedType": {"properties": {"the_type": {"anyOf": [
                                        {"$ref": "#/$defs/IntType"
                                        },
                                        {"$ref": "#/$defs/BoolType"
                                        },
                                        {"$ref": "#/$defs/StrType"
                                        },
                                        {"$ref": "#/$defs/AnyType"
                                        },
                                        {"$ref": "#/$defs/NoneType"
                                        },
                                        {"$ref": "#/$defs/FloatType"
                                        },
                                        {"$ref": "#/$defs/OptionalType"
                                        },
                                        {"$ref": "#/$defs/ListType"
                                        },
                                        {"$ref": "#/$defs/TypedDictType"
                                        },
                                        {"$ref": "#/$defs/TupleType"
                                        },
                                        {"$ref": "#/$defs/SetType"
                                        },
                                        {"$ref": "#/$defs/PDDataFrameType"
                                        },
                                        {"$ref": "#/$defs/PDSeriesType"
                                        },
                                        {"$ref": "#/$defs/NumpyNdarrayType"
                                        }
                                    ], "title": "The Type"
                                }, "description": {"description": "A description of what this type represents. Indicate how to interpret each component of the type.", "title": "Description", "type": "string"
                                }
                            }, "required": ["the_type", "description"
                            ], "title": "ExtendedType", "type": "object", "additionalProperties": false
                        }, "FloatType": {"properties": {"type": {"const": "float", "title": "Type", "type": "string"
                                }
                            }, "required": ["type"
                            ], "title": "FloatType", "type": "object", "additionalProperties": false
                        }, "IntType": {"properties": {"type": {"const": "int", "title": "Type", "type": "string"
                                }
                            }, "required": ["type"
                            ], "title": "IntType", "type": "object", "additionalProperties": false
                        }, "KeyType": {"properties": {"key": {"title": "Key", "type": "string"
                                }, "type": {"anyOf": [
                                        {"$ref": "#/$defs/IntType"
                                        },
                                        {"$ref": "#/$defs/BoolType"
                                        },
                                        {"$ref": "#/$defs/StrType"
                                        },
                                        {"$ref": "#/$defs/AnyType"
                                        },
                                        {"$ref": "#/$defs/NoneType"
                                        },
                                        {"$ref": "#/$defs/FloatType"
                                        },
                                        {"$ref": "#/$defs/OptionalType"
                                        },
                                        {"$ref": "#/$defs/ListType"
                                        },
                                        {"$ref": "#/$defs/TypedDictType"
                                        },
                                        {"$ref": "#/$defs/TupleType"
                                        },
                                        {"$ref": "#/$defs/SetType"
                                        },
                                        {"$ref": "#/$defs/PDDataFrameType"
                                        },
                                        {"$ref": "#/$defs/PDSeriesType"
                                        },
                                        {"$ref": "#/$defs/NumpyNdarrayType"
                                        }
                                    ], "title": "Type"
                                }, "description": {"description": "What this key represents.", "title": "Description", "type": "string"
                                }
                            }, "required": ["key", "type", "description"
                            ], "title": "KeyType", "type": "object", "additionalProperties": false
                        }, "ListType": {"properties": {"type": {"const": "List", "title": "Type", "type": "string"
                                }, "element_type": {"anyOf": [
                                        {"$ref": "#/$defs/IntType"
                                        },
                                        {"$ref": "#/$defs/BoolType"
                                        },
                                        {"$ref": "#/$defs/StrType"
                                        },
                                        {"$ref": "#/$defs/AnyType"
                                        },
                                        {"$ref": "#/$defs/NoneType"
                                        },
                                        {"$ref": "#/$defs/FloatType"
                                        },
                                        {"$ref": "#/$defs/OptionalType"
                                        },
                                        {"$ref": "#/$defs/ListType"
                                        },
                                        {"$ref": "#/$defs/TypedDictType"
                                        },
                                        {"$ref": "#/$defs/TupleType"
                                        },
                                        {"$ref": "#/$defs/SetType"
                                        },
                                        {"$ref": "#/$defs/PDDataFrameType"
                                        },
                                        {"$ref": "#/$defs/PDSeriesType"
                                        },
                                        {"$ref": "#/$defs/NumpyNdarrayType"
                                        }
                                    ], "title": "Element Type"
                                }, "length": {"anyOf": [
                                        {"type": "integer"
                                        },
                                        {"type": "null"
                                        }
                                    ], "description": "The expected length of the list. If None, the length can be arbitrary.", "title": "Length"
                                }
                            }, "required": ["type", "element_type", "length"
                            ], "title": "ListType", "type": "object", "additionalProperties": false
                        }, "NoneType": {"properties": {"type": {"const": "None", "title": "Type", "type": "string"
                                }
                            }, "required": ["type"
                            ], "title": "NoneType", "type": "object", "additionalProperties": false
                        }, "NumpyNdarrayType": {"properties": {"type": {"const": "np.ndarray", "title": "Type", "type": "string"
                                }, "element_type": {"anyOf": [
                                        {"$ref": "#/$defs/IntType"
                                        },
                                        {"$ref": "#/$defs/BoolType"
                                        },
                                        {"$ref": "#/$defs/StrType"
                                        },
                                        {"$ref": "#/$defs/AnyType"
                                        },
                                        {"$ref": "#/$defs/NoneType"
                                        },
                                        {"$ref": "#/$defs/FloatType"
                                        },
                                        {"$ref": "#/$defs/OptionalType"
                                        },
                                        {"$ref": "#/$defs/ListType"
                                        },
                                        {"$ref": "#/$defs/TypedDictType"
                                        },
                                        {"$ref": "#/$defs/TupleType"
                                        },
                                        {"$ref": "#/$defs/SetType"
                                        },
                                        {"$ref": "#/$defs/PDDataFrameType"
                                        },
                                        {"$ref": "#/$defs/PDSeriesType"
                                        },
                                        {"$ref": "#/$defs/NumpyNdarrayType"
                                        }
                                    ], "title": "Element Type"
                                }, "length": {"anyOf": [
                                        {"type": "integer"
                                        },
                                        {"type": "null"
                                        }
                                    ], "description": "The expected length of the list. If None, the length can be arbitrary.", "title": "Length"
                                }
                            }, "required": ["type", "element_type", "length"
                            ], "title": "NumpyNdarrayType", "type": "object", "additionalProperties": false
                        }, "OptionalType": {"properties": {"type": {"const": "Optional", "title": "Type", "type": "string"
                                }, "wrapped_type": {"anyOf": [
                                        {"$ref": "#/$defs/IntType"
                                        },
                                        {"$ref": "#/$defs/BoolType"
                                        },
                                        {"$ref": "#/$defs/StrType"
                                        },
                                        {"$ref": "#/$defs/AnyType"
                                        },
                                        {"$ref": "#/$defs/NoneType"
                                        },
                                        {"$ref": "#/$defs/FloatType"
                                        },
                                        {"$ref": "#/$defs/OptionalType"
                                        },
                                        {"$ref": "#/$defs/ListType"
                                        },
                                        {"$ref": "#/$defs/TypedDictType"
                                        },
                                        {"$ref": "#/$defs/TupleType"
                                        },
                                        {"$ref": "#/$defs/SetType"
                                        },
                                        {"$ref": "#/$defs/PDDataFrameType"
                                        },
                                        {"$ref": "#/$defs/PDSeriesType"
                                        },
                                        {"$ref": "#/$defs/NumpyNdarrayType"
                                        }
                                    ], "title": "Wrapped Type"
                                }
                            }, "required": ["type", "wrapped_type"
                            ], "title": "OptionalType", "type": "object", "additionalProperties": false
                        }, "PDDataFrameType": {"properties": {"type": {"const": "pd.DataFrame", "title": "Type", "type": "string"
                                }, "columns": {"description": "A list of key-value pairs where the key is the column name and the value is the type of the column.", "items": {"$ref": "#/$defs/KeyType"
                                    }, "title": "Columns", "type": "array"
                                }
                            }, "required": ["type", "columns"
                            ], "title": "PDDataFrameType", "type": "object", "additionalProperties": false
                        }, "PDSeriesType": {"properties": {"type": {"const": "pd.Series", "title": "Type", "type": "string"
                                }, "element_type": {"anyOf": [
                                        {"$ref": "#/$defs/IntType"
                                        },
                                        {"$ref": "#/$defs/BoolType"
                                        },
                                        {"$ref": "#/$defs/StrType"
                                        },
                                        {"$ref": "#/$defs/AnyType"
                                        },
                                        {"$ref": "#/$defs/NoneType"
                                        },
                                        {"$ref": "#/$defs/FloatType"
                                        },
                                        {"$ref": "#/$defs/OptionalType"
                                        },
                                        {"$ref": "#/$defs/ListType"
                                        },
                                        {"$ref": "#/$defs/TypedDictType"
                                        },
                                        {"$ref": "#/$defs/TupleType"
                                        },
                                        {"$ref": "#/$defs/SetType"
                                        },
                                        {"$ref": "#/$defs/PDDataFrameType"
                                        },
                                        {"$ref": "#/$defs/PDSeriesType"
                                        },
                                        {"$ref": "#/$defs/NumpyNdarrayType"
                                        }
                                    ], "title": "Element Type"
                                }
                            }, "required": ["type", "element_type"
                            ], "title": "PDSeriesType", "type": "object", "additionalProperties": false
                        }, "SetType": {"properties": {"type": {"const": "Set", "title": "Type", "type": "string"
                                }, "element_type": {"anyOf": [
                                        {"$ref": "#/$defs/IntType"
                                        },
                                        {"$ref": "#/$defs/BoolType"
                                        },
                                        {"$ref": "#/$defs/StrType"
                                        },
                                        {"$ref": "#/$defs/AnyType"
                                        },
                                        {"$ref": "#/$defs/NoneType"
                                        },
                                        {"$ref": "#/$defs/FloatType"
                                        },
                                        {"$ref": "#/$defs/OptionalType"
                                        },
                                        {"$ref": "#/$defs/ListType"
                                        },
                                        {"$ref": "#/$defs/TypedDictType"
                                        },
                                        {"$ref": "#/$defs/TupleType"
                                        },
                                        {"$ref": "#/$defs/SetType"
                                        },
                                        {"$ref": "#/$defs/PDDataFrameType"
                                        },
                                        {"$ref": "#/$defs/PDSeriesType"
                                        },
                                        {"$ref": "#/$defs/NumpyNdarrayType"
                                        }
                                    ], "title": "Element Type"
                                }
                            }, "required": ["type", "element_type"
                            ], "title": "SetType", "type": "object", "additionalProperties": false
                        }, "StrType": {"properties": {"type": {"const": "str", "title": "Type", "type": "string"
                                }
                            }, "required": ["type"
                            ], "title": "StrType", "type": "object", "additionalProperties": false
                        }, "TupleType": {"properties": {"type": {"const": "Tuple", "title": "Type", "type": "string"
                                }, "elements": {"items": {"anyOf": [
                                            {"$ref": "#/$defs/IntType"
                                            },
                                            {"$ref": "#/$defs/BoolType"
                                            },
                                            {"$ref": "#/$defs/StrType"
                                            },
                                            {"$ref": "#/$defs/AnyType"
                                            },
                                            {"$ref": "#/$defs/NoneType"
                                            },
                                            {"$ref": "#/$defs/FloatType"
                                            },
                                            {"$ref": "#/$defs/OptionalType"
                                            },
                                            {"$ref": "#/$defs/ListType"
                                            },
                                            {"$ref": "#/$defs/TypedDictType"
                                            },
                                            {"$ref": "#/$defs/TupleType"
                                            },
                                            {"$ref": "#/$defs/SetType"
                                            },
                                            {"$ref": "#/$defs/PDDataFrameType"
                                            },
                                            {"$ref": "#/$defs/PDSeriesType"
                                            },
                                            {"$ref": "#/$defs/NumpyNdarrayType"
                                            }
                                        ]
                                    }, "title": "Elements", "type": "array"
                                }
                            }, "required": ["type", "elements"
                            ], "title": "TupleType", "type": "object", "additionalProperties": false
                        }, "TypedDictType": {"properties": {"type": {"const": "TypedDict", "title": "Type", "type": "string"
                                }, "name": {"description": "A unique name for the dictionary type. This is used to generate a unique TypedDict name.", "title": "Name", "type": "string"
                                }, "items": {"description": "A list of key-value pairs where the key is the key name and the value is the type of the key.", "items": {"$ref": "#/$defs/KeyType"
                                    }, "title": "Items", "type": "array"
                                }
                            }, "required": ["type", "name", "items"
                            ], "title": "TypedDictType", "type": "object", "additionalProperties": false
                        }
                    }, "properties": {"id": {"description": "The id of the node to modify.", "title": "Id", "type": "string"
                        }, "label": {"description": "The new label of the node.  Keep in sync with the requirements, algorithm, and code.", "title": "Label", "type": "string"
                        }, "requirements": {"description": "A list of requirements that must be true of the return value for the function.  Describe the representation of the return value as well.", "items": {"type": "string"
                            }, "title": "Requirements", "type": "array"
                        }, "function_return_type": {"description": "The return type of the node.", "properties": {"the_type": {"anyOf": [
                                        {"$ref": "#/$defs/IntType"
                                        },
                                        {"$ref": "#/$defs/BoolType"
                                        },
                                        {"$ref": "#/$defs/StrType"
                                        },
                                        {"$ref": "#/$defs/AnyType"
                                        },
                                        {"$ref": "#/$defs/NoneType"
                                        },
                                        {"$ref": "#/$defs/FloatType"
                                        },
                                        {"$ref": "#/$defs/OptionalType"
                                        },
                                        {"$ref": "#/$defs/ListType"
                                        },
                                        {"$ref": "#/$defs/TypedDictType"
                                        },
                                        {"$ref": "#/$defs/TupleType"
                                        },
                                        {"$ref": "#/$defs/SetType"
                                        },
                                        {"$ref": "#/$defs/PDDataFrameType"
                                        },
                                        {"$ref": "#/$defs/PDSeriesType"
                                        },
                                        {"$ref": "#/$defs/NumpyNdarrayType"
                                        }
                                    ], "title": "The Type"
                                }, "description": {"description": "A description of what this type represents. Indicate how to interpret each component of the type.", "title": "Description", "type": "string"
                                }
                            }, "required": ["the_type", "description"
                            ], "title": "ExtendedType", "type": "object", "additionalProperties": false
                        }, "algorithm": {"description": "The algorithm of the node.", "items": {"type": "string"
                            }, "title": "Algorithm", "type": "array"
                        }, "code": {"description": "The code for the node.  Only modify if there is already an code.  The code should be a list of strings, one for each line of code.  The signature must match the original version, except for the return type", "items": {"type": "string"
                            }, "title": "Code", "type": "array"
                        }
                    }, "required": ["id", "label", "requirements", "function_return_type", "algorithm", "code"
                    ], "title": "update_node", "type": "object", "additionalProperties": false
                }
            }
            """

            # """
            # {
            #     "name": "update_node",
            #     "description": "Modify a node in the diagram.  You must preserve consistency between the pill, label, requirements, algorithm, and code.  Use null for node parts you don't want to modify.  If you wish to modify the return type for the node, leave the algorithm and code blank.",
            #     "parameters": {
            #         "type": "object",
            #         "properties": {
            #             "id": {
            #                 "type": "string",
            #                 "description": "The id of the node to modify"
            #             },
            #             "pill": {
            #                 "type": "string",
            #                 "description": "The pill of the node.  Two words, hyphenated and title-case.  Keep in sync with the label, requirements, algorithm, and code."
            #             },
            #             "label": {
            #                 "type": "string",
            #                 "description": "The new label of the node.  Keep in sync with the requirements, algorithm, and code."
            #             },
            #             "requirements": {
            #                 "type": "array",
            #                 "items": {
            #                     "type": "string"
            #                 },
            #                 "description": "A list of requirements that must be true of the return value for the function.  Describe the representation of the return value as well."
            #             },
            #             "algorithm": {
            #                 "type": "array",
            #                 "items": {
            #                     "type": "string"
            #                 },
            #                 "description": "The algorithm for the node.  Only modify if there is already an algorithm."
            #             },
            #             "code": {
            #                 "type": "array",
            #                 "items": {
            #                     "type": "string"
            #                 },
            #                 "description": "The code for the node.  Only modify if there is already an code.  The code should be a list of strings, one for each line of code.  The signature must match the original version."
            #             }
            #         },
            #         "required": ["id" ]
            #     }
            # }
            # """
            log(f"update_node: {id}, {requirements}, {algorithm}, {code}")
            dfg = self.page.dfg

            if dfg[id] is None:
                return (f"Node {id} does not exist", None)

            node = dfg[id]
            mods = []

            if code and code != node.code:
                log(f"Updating code to {code}")
                node = node.update(code=code, phase=Phase.algorithm)
                mods.append("code")
            if algorithm and algorithm != node.algorithm:
                log(f"Updating algorithm to {algorithm}")
                node = node.update(algorithm=algorithm, phase=Phase.requirements)
                mods.append("algorithm")
            if requirements and requirements != node.requirements:
                log(f"Updating requirements to {requirements}")
                node = node.update(
                    requirements=requirements,
                    phase=Phase.clean,
                )
                mods.append("requirements")

            if function_return_type:
                function_return_type = ExtendedType.model_validate(function_return_type)
                if function_return_type != node.function_return_type:
                    log(
                        f"Updating function_return_type from {node.function_return_type} to {function_return_type}"
                    )
                    node = node.update(
                        function_return_type=function_return_type,
                        phase=Phase.clean,
                    )
                mods.append("return-type")
            if label and label != node.label:
                log(f"Updating label to {label}")
                node = node.update(label=label, phase=Phase.clean)
                mods.append("label")
            if pill and pill != node.pill:
                log(f"Updating pill to {pill}")
                node = node.update(pill=pill, phase=Phase.clean)
                mods.append("pill")

            if node.phase == Phase.clean:
                dfg = dfg.lower_phase_with_successors(node.id, Phase.clean)

            if config.x_trust_ama:
                if "requirements" in mods or "return-type" in mods:
                    node = node.update(
                        cache=node.cache.update(Phase.requirements, node)
                    )
                    node = node.update(phase=Phase.requirements)
                if "algorithm" in mods:
                    node = node.update(cache=node.cache.update(Phase.algorithm, node))
                    node = node.update(phase=Phase.algorithm)
                if "code" in mods:
                    node = node.update(cache=node.cache.update(Phase.code, node))
                    node = node.update(phase=Phase.code)

            dfg = dfg.with_node(node)
            self.page.update_dfg(dfg)

            mod_str = ", ".join(reversed(mods))

            return (
                f"Updated {mod_str} for {node.pill}",
                node.model_dump_json(indent=2),
            )

    else:

        def update_node(
            self,
            id: str,
            pill: str | None = None,
            label: str | None = None,
            requirements: List[str] | None = None,
            function_return_type: ExtendedType | None = None,
            code: List[str] | None = None,
        ) -> ReturnType:
            """\
            {"name": "update_node",
            "strict": true,
            "parameters": {"$defs": {"AnyType": {"properties": {"type": {"const": "Any", "title": "Type", "type": "string"
                                }
                            }, "required": ["type"
                            ], "title": "AnyType", "type": "object", "additionalProperties": false
                        }, "BoolType": {"properties": {"type": {"const": "bool", "title": "Type", "type": "string"
                                }
                            }, "required": ["type"
                            ], "title": "BoolType", "type": "object", "additionalProperties": false
                        }, "ExtendedType": {"properties": {"the_type": {"anyOf": [
                                        {"$ref": "#/$defs/IntType"
                                        },
                                        {"$ref": "#/$defs/BoolType"
                                        },
                                        {"$ref": "#/$defs/StrType"
                                        },
                                        {"$ref": "#/$defs/AnyType"
                                        },
                                        {"$ref": "#/$defs/NoneType"
                                        },
                                        {"$ref": "#/$defs/FloatType"
                                        },
                                        {"$ref": "#/$defs/OptionalType"
                                        },
                                        {"$ref": "#/$defs/ListType"
                                        },
                                        {"$ref": "#/$defs/TypedDictType"
                                        },
                                        {"$ref": "#/$defs/TupleType"
                                        },
                                        {"$ref": "#/$defs/SetType"
                                        },
                                        {"$ref": "#/$defs/PDDataFrameType"
                                        },
                                        {"$ref": "#/$defs/PDSeriesType"
                                        },
                                        {"$ref": "#/$defs/NumpyNdarrayType"
                                        }
                                    ], "title": "The Type"
                                }, "description": {"description": "A description of what this type represents. Indicate how to interpret each component of the type.", "title": "Description", "type": "string"
                                }
                            }, "required": ["the_type", "description"
                            ], "title": "ExtendedType", "type": "object", "additionalProperties": false
                        }, "FloatType": {"properties": {"type": {"const": "float", "title": "Type", "type": "string"
                                }
                            }, "required": ["type"
                            ], "title": "FloatType", "type": "object", "additionalProperties": false
                        }, "IntType": {"properties": {"type": {"const": "int", "title": "Type", "type": "string"
                                }
                            }, "required": ["type"
                            ], "title": "IntType", "type": "object", "additionalProperties": false
                        }, "KeyType": {"properties": {"key": {"title": "Key", "type": "string"
                                }, "type": {"anyOf": [
                                        {"$ref": "#/$defs/IntType"
                                        },
                                        {"$ref": "#/$defs/BoolType"
                                        },
                                        {"$ref": "#/$defs/StrType"
                                        },
                                        {"$ref": "#/$defs/AnyType"
                                        },
                                        {"$ref": "#/$defs/NoneType"
                                        },
                                        {"$ref": "#/$defs/FloatType"
                                        },
                                        {"$ref": "#/$defs/OptionalType"
                                        },
                                        {"$ref": "#/$defs/ListType"
                                        },
                                        {"$ref": "#/$defs/TypedDictType"
                                        },
                                        {"$ref": "#/$defs/TupleType"
                                        },
                                        {"$ref": "#/$defs/SetType"
                                        },
                                        {"$ref": "#/$defs/PDDataFrameType"
                                        },
                                        {"$ref": "#/$defs/PDSeriesType"
                                        },
                                        {"$ref": "#/$defs/NumpyNdarrayType"
                                        }
                                    ], "title": "Type"
                                }, "description": {"description": "What this key represents.", "title": "Description", "type": "string"
                                }
                            }, "required": ["key", "type", "description"
                            ], "title": "KeyType", "type": "object", "additionalProperties": false
                        }, "ListType": {"properties": {"type": {"const": "List", "title": "Type", "type": "string"
                                }, "element_type": {"anyOf": [
                                        {"$ref": "#/$defs/IntType"
                                        },
                                        {"$ref": "#/$defs/BoolType"
                                        },
                                        {"$ref": "#/$defs/StrType"
                                        },
                                        {"$ref": "#/$defs/AnyType"
                                        },
                                        {"$ref": "#/$defs/NoneType"
                                        },
                                        {"$ref": "#/$defs/FloatType"
                                        },
                                        {"$ref": "#/$defs/OptionalType"
                                        },
                                        {"$ref": "#/$defs/ListType"
                                        },
                                        {"$ref": "#/$defs/TypedDictType"
                                        },
                                        {"$ref": "#/$defs/TupleType"
                                        },
                                        {"$ref": "#/$defs/SetType"
                                        },
                                        {"$ref": "#/$defs/PDDataFrameType"
                                        },
                                        {"$ref": "#/$defs/PDSeriesType"
                                        },
                                        {"$ref": "#/$defs/NumpyNdarrayType"
                                        }
                                    ], "title": "Element Type"
                                }, "length": {"anyOf": [
                                        {"type": "integer"
                                        },
                                        {"type": "null"
                                        }
                                    ], "description": "The expected length of the list. If None, the length can be arbitrary.", "title": "Length"
                                }
                            }, "required": ["type", "element_type", "length"
                            ], "title": "ListType", "type": "object", "additionalProperties": false
                        }, "NoneType": {"properties": {"type": {"const": "None", "title": "Type", "type": "string"
                                }
                            }, "required": ["type"
                            ], "title": "NoneType", "type": "object", "additionalProperties": false
                        }, "NumpyNdarrayType": {"properties": {"type": {"const": "np.ndarray", "title": "Type", "type": "string"
                                }, "element_type": {"anyOf": [
                                        {"$ref": "#/$defs/IntType"
                                        },
                                        {"$ref": "#/$defs/BoolType"
                                        },
                                        {"$ref": "#/$defs/StrType"
                                        },
                                        {"$ref": "#/$defs/AnyType"
                                        },
                                        {"$ref": "#/$defs/NoneType"
                                        },
                                        {"$ref": "#/$defs/FloatType"
                                        },
                                        {"$ref": "#/$defs/OptionalType"
                                        },
                                        {"$ref": "#/$defs/ListType"
                                        },
                                        {"$ref": "#/$defs/TypedDictType"
                                        },
                                        {"$ref": "#/$defs/TupleType"
                                        },
                                        {"$ref": "#/$defs/SetType"
                                        },
                                        {"$ref": "#/$defs/PDDataFrameType"
                                        },
                                        {"$ref": "#/$defs/PDSeriesType"
                                        },
                                        {"$ref": "#/$defs/NumpyNdarrayType"
                                        }
                                    ], "title": "Element Type"
                                }, "length": {"anyOf": [
                                        {"type": "integer"
                                        },
                                        {"type": "null"
                                        }
                                    ], "description": "The expected length of the list. If None, the length can be arbitrary.", "title": "Length"
                                }
                            }, "required": ["type", "element_type", "length"
                            ], "title": "NumpyNdarrayType", "type": "object", "additionalProperties": false
                        }, "OptionalType": {"properties": {"type": {"const": "Optional", "title": "Type", "type": "string"
                                }, "wrapped_type": {"anyOf": [
                                        {"$ref": "#/$defs/IntType"
                                        },
                                        {"$ref": "#/$defs/BoolType"
                                        },
                                        {"$ref": "#/$defs/StrType"
                                        },
                                        {"$ref": "#/$defs/AnyType"
                                        },
                                        {"$ref": "#/$defs/NoneType"
                                        },
                                        {"$ref": "#/$defs/FloatType"
                                        },
                                        {"$ref": "#/$defs/OptionalType"
                                        },
                                        {"$ref": "#/$defs/ListType"
                                        },
                                        {"$ref": "#/$defs/TypedDictType"
                                        },
                                        {"$ref": "#/$defs/TupleType"
                                        },
                                        {"$ref": "#/$defs/SetType"
                                        },
                                        {"$ref": "#/$defs/PDDataFrameType"
                                        },
                                        {"$ref": "#/$defs/PDSeriesType"
                                        },
                                        {"$ref": "#/$defs/NumpyNdarrayType"
                                        }
                                    ], "title": "Wrapped Type"
                                }
                            }, "required": ["type", "wrapped_type"
                            ], "title": "OptionalType", "type": "object", "additionalProperties": false
                        }, "PDDataFrameType": {"properties": {"type": {"const": "pd.DataFrame", "title": "Type", "type": "string"
                                }, "columns": {"description": "A list of key-value pairs where the key is the column name and the value is the type of the column.", "items": {"$ref": "#/$defs/KeyType"
                                    }, "title": "Columns", "type": "array"
                                }
                            }, "required": ["type", "columns"
                            ], "title": "PDDataFrameType", "type": "object", "additionalProperties": false
                        }, "PDSeriesType": {"properties": {"type": {"const": "pd.Series", "title": "Type", "type": "string"
                                }, "element_type": {"anyOf": [
                                        {"$ref": "#/$defs/IntType"
                                        },
                                        {"$ref": "#/$defs/BoolType"
                                        },
                                        {"$ref": "#/$defs/StrType"
                                        },
                                        {"$ref": "#/$defs/AnyType"
                                        },
                                        {"$ref": "#/$defs/NoneType"
                                        },
                                        {"$ref": "#/$defs/FloatType"
                                        },
                                        {"$ref": "#/$defs/OptionalType"
                                        },
                                        {"$ref": "#/$defs/ListType"
                                        },
                                        {"$ref": "#/$defs/TypedDictType"
                                        },
                                        {"$ref": "#/$defs/TupleType"
                                        },
                                        {"$ref": "#/$defs/SetType"
                                        },
                                        {"$ref": "#/$defs/PDDataFrameType"
                                        },
                                        {"$ref": "#/$defs/PDSeriesType"
                                        },
                                        {"$ref": "#/$defs/NumpyNdarrayType"
                                        }
                                    ], "title": "Element Type"
                                }
                            }, "required": ["type", "element_type"
                            ], "title": "PDSeriesType", "type": "object", "additionalProperties": false
                        }, "SetType": {"properties": {"type": {"const": "Set", "title": "Type", "type": "string"
                                }, "element_type": {"anyOf": [
                                        {"$ref": "#/$defs/IntType"
                                        },
                                        {"$ref": "#/$defs/BoolType"
                                        },
                                        {"$ref": "#/$defs/StrType"
                                        },
                                        {"$ref": "#/$defs/AnyType"
                                        },
                                        {"$ref": "#/$defs/NoneType"
                                        },
                                        {"$ref": "#/$defs/FloatType"
                                        },
                                        {"$ref": "#/$defs/OptionalType"
                                        },
                                        {"$ref": "#/$defs/ListType"
                                        },
                                        {"$ref": "#/$defs/TypedDictType"
                                        },
                                        {"$ref": "#/$defs/TupleType"
                                        },
                                        {"$ref": "#/$defs/SetType"
                                        },
                                        {"$ref": "#/$defs/PDDataFrameType"
                                        },
                                        {"$ref": "#/$defs/PDSeriesType"
                                        },
                                        {"$ref": "#/$defs/NumpyNdarrayType"
                                        }
                                    ], "title": "Element Type"
                                }
                            }, "required": ["type", "element_type"
                            ], "title": "SetType", "type": "object", "additionalProperties": false
                        }, "StrType": {"properties": {"type": {"const": "str", "title": "Type", "type": "string"
                                }
                            }, "required": ["type"
                            ], "title": "StrType", "type": "object", "additionalProperties": false
                        }, "TupleType": {"properties": {"type": {"const": "Tuple", "title": "Type", "type": "string"
                                }, "elements": {"items": {"anyOf": [
                                            {"$ref": "#/$defs/IntType"
                                            },
                                            {"$ref": "#/$defs/BoolType"
                                            },
                                            {"$ref": "#/$defs/StrType"
                                            },
                                            {"$ref": "#/$defs/AnyType"
                                            },
                                            {"$ref": "#/$defs/NoneType"
                                            },
                                            {"$ref": "#/$defs/FloatType"
                                            },
                                            {"$ref": "#/$defs/OptionalType"
                                            },
                                            {"$ref": "#/$defs/ListType"
                                            },
                                            {"$ref": "#/$defs/TypedDictType"
                                            },
                                            {"$ref": "#/$defs/TupleType"
                                            },
                                            {"$ref": "#/$defs/SetType"
                                            },
                                            {"$ref": "#/$defs/PDDataFrameType"
                                            },
                                            {"$ref": "#/$defs/PDSeriesType"
                                            },
                                            {"$ref": "#/$defs/NumpyNdarrayType"
                                            }
                                        ]
                                    }, "title": "Elements", "type": "array"
                                }
                            }, "required": ["type", "elements"
                            ], "title": "TupleType", "type": "object", "additionalProperties": false
                        }, "TypedDictType": {"properties": {"type": {"const": "TypedDict", "title": "Type", "type": "string"
                                }, "name": {"description": "A unique name for the dictionary type. This is used to generate a unique TypedDict name.", "title": "Name", "type": "string"
                                }, "items": {"description": "A list of key-value pairs where the key is the key name and the value is the type of the key.", "items": {"$ref": "#/$defs/KeyType"
                                    }, "title": "Items", "type": "array"
                                }
                            }, "required": ["type", "name", "items"
                            ], "title": "TypedDictType", "type": "object", "additionalProperties": false
                        }
                    }, "properties": {"id": {"description": "The id of the node to modify.", "title": "Id", "type": "string"
                        }, "label": {"description": "The new label of the node.  Keep in sync with the requirements and code.", "title": "Label", "type": "string"
                        }, "requirements": {"description": "A list of requirements that must be true of the return value for the function.  Describe the representation of the return value as well.", "items": {"type": "string"
                            }, "title": "Requirements", "type": "array"
                        }, "function_return_type": {"description": "The return type of the node.  Only change if necessary.", "properties": {"the_type": {"anyOf": [
                                        {"$ref": "#/$defs/IntType"
                                        },
                                        {"$ref": "#/$defs/BoolType"
                                        },
                                        {"$ref": "#/$defs/StrType"
                                        },
                                        {"$ref": "#/$defs/AnyType"
                                        },
                                        {"$ref": "#/$defs/NoneType"
                                        },
                                        {"$ref": "#/$defs/FloatType"
                                        },
                                        {"$ref": "#/$defs/OptionalType"
                                        },
                                        {"$ref": "#/$defs/ListType"
                                        },
                                        {"$ref": "#/$defs/TypedDictType"
                                        },
                                        {"$ref": "#/$defs/TupleType"
                                        },
                                        {"$ref": "#/$defs/SetType"
                                        },
                                        {"$ref": "#/$defs/PDDataFrameType"
                                        },
                                        {"$ref": "#/$defs/PDSeriesType"
                                        },
                                        {"$ref": "#/$defs/NumpyNdarrayType"
                                        }
                                    ], "title": "The Type"
                                }, "description": {"description": "A description of what this type represents. Indicate how to interpret each component of the type.", "title": "Description", "type": "string"
                                }
                            }, "required": ["the_type", "description"
                            ], "title": "ExtendedType", "type": "object", "additionalProperties": false
                        },"code": {"description": "The code for the node.  Only modify if there is already an code.  The code should be a list of strings, one for each line of code.  The signature must match the original version, except for the return type", "items": {"type": "string"
                            }, "title": "Code", "type": "array"
                        }
                    }, "required": ["id", "label", "requirements", "function_return_type", "code"],
                      "title": "update_node", "type": "object", "additionalProperties": false
                }
            }
            """
            log(f"update_node: {id}, {requirements},  {code}")
            dfg = self.page.dfg

            if dfg[id] is None:
                return (f"Node {id} does not exist", None)

            node = dfg[id]
            mods = []

            if code and code != node.code:
                log(f"Updating code to {code}")
                node = node.update(code=code, phase=Phase.algorithm)
                mods.append("code")
            if requirements and requirements != node.requirements:
                log(f"Updating requirements to {requirements}")
                node = node.update(
                    requirements=requirements,
                    phase=Phase.clean,
                )
                mods.append("requirements")

            if function_return_type:
                function_return_type = ExtendedType.model_validate(function_return_type)
                if function_return_type != node.function_return_type:
                    log(
                        f"Updating function_return_type from {node.function_return_type.to_markdown(True)} to {function_return_type.to_markdown(True)}"
                    )
                    node = node.update(
                        function_return_type=function_return_type,
                        phase=Phase.clean,
                    )
                mods.append("return-type")
            if label and label != node.label:
                log(f"Updating label to {label}")
                node = node.update(label=label, phase=Phase.clean)
                mods.append("label")
            if pill and pill != node.pill:
                log(f"Updating pill to {pill}")
                node = node.update(pill=pill, phase=Phase.clean)
                mods.append("pill")

            if node.phase == Phase.clean:
                dfg = dfg.lower_phase_with_successors(node.id, Phase.clean)

            if config.x_trust_ama:
                if "requirements" in mods or "return-type" in mods:
                    node = node.update(
                        cache=node.cache.update(Phase.requirements, node)
                    )
                    node = node.update(phase=Phase.requirements)
                if "code" in mods:
                    node = node.update(cache=node.cache.update(Phase.code, node))
                    node = node.update(phase=Phase.code)

            dfg = dfg.with_node(node)
            self.page.update_dfg(dfg)

            mod_str = ", ".join(reversed(mods))

            return (
                f"Updated {mod_str} for {node.pill}",
                node.model_dump_json(indent=2),
            )

    def add_node(
        self,
        id: str,
        predecessors: List[str],
        label: str,
        requirements: List[str],
    ) -> ReturnType:
        """
        {
            "name": "add_node",
            "description": "Add a node and its requirements to the diagram.  Do not provide an algorithm or code.  Nodes should represent one small step in a pipeline. Eg: one transformation, one statistical test, one visualization, one output, ...  Provide a list of nodes that should point to the new node.  Provide a unique id for the node, a list of predecessor nodes, a label, and a list of requirements that must be true of the return value for the function.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "A unique id for the new node.  No spaces or special characters."
                    },
                    "predecessors": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "The ids of the predecessor nodes"
                    },
                    "label": {
                        "type": "string",
                        "description": "The label of the new node"
                    },
                    "requirements": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "A list of requirements that must be true of the return value for the function.  Describe the representation of the return value as well."
                    }
                },
                "required": ["id", "predecessors", "label", "requirements"],
                "additionalProperties": false
            }
        }
        """
        log(f"add_node: {id}, {predecessors}, {label}, {requirements}")
        dfg = self.page.dfg

        pill = "tmp-pill"

        if dfg.node_for_pill(pill) is not None:
            return (f"Node with pill {pill} already exists", None)

        for pred in predecessors:
            if dfg[pred] is None:
                return (f"predecessor {pred} does not exist", None)

        pill = dfg.make_pill(label)
        geometry = Geometry(x=0, y=0, width=0, height=0)
        output_geometry = geometry.translate(geometry.width + 100, 0).resize(200, 150)
        node_updates = {
            x.id: DiagramNodeUpdate(
                id=x.id,
                pill=x.pill,
                label=x.label,
                geometry=x.geometry,
                output_geometry=x.output_geometry,
            )
            for x in dfg.nodes
        } | {
            id: DiagramNodeUpdate(
                id=id,
                pill=pill,
                label=label,
                geometry=geometry,
                output_geometry=output_geometry,
            )
        }
        edge_updates = {
            x.id: DiagramEdgeUpdate(id=x.id, src=x.src, dst=x.dst) for x in dfg.edges
        } | {
            f"{src}-{id}": DiagramEdgeUpdate(id=f"{src}-{id}", src=src, dst=id)
            for src in predecessors
        }

        dfg = update_dataflow_graph(
            dfg,
            mxDiagramUpdate(
                version=dfg.version, nodes=node_updates, edges=edge_updates
            ),
        )

        node = dfg[id]
        node = node.update(requirements=requirements)
        dfg = dfg.with_node(node)

        self.page.update_dfg(dfg)

        src_pills = ", ".join(predecessors)
        return (
            f"Added new node {node.pill}.  Connected it to {src_pills}",
            node.model_dump_json(indent=2),
        )

    def add_edge(self, src_id: str, dst_id: str) -> ReturnType:
        """
        {
            "name": "add_edge",
            "description": "Add an edge to the diagram",
            "parameters": {
                "type": "object",
                "properties": {
                    "src_id": {
                        "type": "string",
                        "description": "The id of the source node"
                    },
                    "dst_id": {
                        "type": "string",
                        "description": "The id of the destination node"
                    }
                },
                "required": ["src_id", "dst_id"]
            }
        }
        """
        log(f"add_edge: {src_id}, {dst_id}")
        dfg = self.page.dfg

        for id in [src_id, dst_id]:
            if dfg[id] is None:
                return (f"Node {id} does not exist", None)

        dfg = dfg.with_new_edge(src_id, dst_id)
        self.page.update_dfg(dfg)

        # find id for that edge
        edge = dfg.edge_for_nodes(src_id, dst_id)

        return (
            f"Added new edge from {src_id} to {dst_id}",
            edge.model_dump_json(indent=2),
        )

    def remove_node(self, id: str) -> ReturnType:
        """
        {
            "name": "remove_node",
            "description": "Remove a node from the diagram",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The id of the node to remove"
                    }
                },
                "required": ["id"]
            }
        }
        """
        log(f"remove_node: {id}")
        dfg = self.page.dfg

        if dfg[id] is None:
            return (f"Node {id} does not exist", None)

        node = dfg[id]
        dfg_update = mxDiagramUpdate(
            version=dfg.version,
            nodes={
                x.id: DiagramNodeUpdate(
                    id=x.id,
                    pill=x.pill,
                    label=x.label,
                    geometry=x.geometry,
                    output_geometry=x.output_geometry,
                )
                for x in dfg.nodes
                if x.id != id
            },
            edges={
                x.id: DiagramEdgeUpdate(id=x.id, src=x.src, dst=x.dst)
                for x in dfg.edges
                if not (x.src == id or x.dst == id)
            },
        )

        dfg = update_dataflow_graph(dfg, dfg_update)
        self.page.update_dfg(dfg)
        return (f"Removed node {node.pill}", None)

    def remove_edge(self, id: str) -> ReturnType:
        """
        {
            "name": "remove_edge",
            "description": "Remove an edge from the diagram",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The id of the edge to remove"
                    }
                },
                "required": ["id"]
            }
        }
        """
        log(f"remove_edge: {id}")
        dfg = self.page.dfg

        edge_to_remove = dfg.get_edge(id)

        if edge_to_remove is None:
            return (
                "Edge has already been removed",
                f"{{ success: false, message: 'Edge {id} does not exist' }}",
            )

        dfg_update = mxDiagramUpdate(
            version=dfg.version,
            nodes={
                x.id: DiagramNodeUpdate(
                    id=x.id,
                    pill=x.pill,
                    label=x.label,
                    geometry=x.geometry,
                    output_geometry=x.output_geometry,
                )
                for x in dfg.nodes
            },
            edges={
                x.id: DiagramEdgeUpdate(id=x.id, src=x.src, dst=x.dst)
                for x in dfg.edges
                if x.id != id
            },
        )

        dfg = update_dataflow_graph(dfg, dfg_update)
        self.page.update_dfg(dfg)
        return (f"Removed edge from {edge_to_remove.src} to {edge_to_remove.dst}", None)

    def classify_question(self, question: str) -> str:
        assistant: OpenAIAssistant = OpenAIAssistant(
            model="gpt-4o-mini",
            interactive=False,
            system_prompt_key="classify_ama_prompt",
        )
        for message in self.visible_messages[-4:]:
            assistant.add_message(message.role, message.content)
        assistant.add_message("user", f"Classify this prompt:\n```\n{question}\n```\n")
        return str(assistant.completion(QuestionKind, "gpt-4o-mini").kind)

    def complete(self, prompt: str, selected_node: str | None = None) -> Iterable[str]:
        try:
            kind = self.classify_question(prompt)

            with logger(f"AMA: {kind}"):
                if kind == "Explain":
                    yield from self._complete(
                        "ama_explain",
                        [self.python_eval, self.inspect],
                        prompt,
                        selected_node,
                    )

                elif kind == "Modify":
                    original_dfg = self.page.dfg
                    yield from self._complete(
                        "ama_modify",
                        [
                            self.python_eval,
                            self.inspect,
                            self.add_node,
                            self.add_edge,
                            self.remove_node,
                            self.remove_edge,
                            self.update_node,
                        ],
                        prompt,
                        selected_node,
                    )
                else:
                    raise ValueError(f"Unknown kind: {kind}")
        except Exception as e:
            error(f"Error: {e}")
            raise FlowcoError(f"Error: {e}")

    def _complete(
        self, system_prompt, functions, prompt: str, selected_node: str | None = None
    ) -> Iterable[str]:

        markdown = ""

        self.assistant.set_functions(functions)

        if (
            self.completion_dfg is None
            or self.completion_dfg.nodes != self.page.dfg.nodes
            or self.completion_dfg.edges != self.page.dfg.edges
        ):
            log("Recomputing completion DFG prompts")
            self.completion_dfg = self.page.dfg
            locals = "The following variables are already defined.  You may use them in any code you run via a function call.\n"

            for node in self.page.dfg.nodes:
                result = node.result
                if (
                    result is not None
                    and result.result is not None
                    and node.function_return_type is not None
                    and not node.function_return_type.is_None_type()
                ):
                    # type_description = node.function_return_type.type_description()
                    locals += f"`{node.function_result_var} : {node.function_return_type.to_python_type()}` is {node.function_return_type.description}\n\n"

            locals += "\nYou have access to these files:\n" + str(
                self.page.tables.as_preconditions()
            )

            self.assistant.add_message("user", self.page.dfg.to_image_prompt_messages())

            node_fields = [
                "id",
                "pill",
                "label",
                "requirements",
                "code",
            ]

            if config.x_algorithm_phase:
                node_fields.append("algorithm")

            self.assistant.add_message(
                "user",
                messages_for_graph(
                    self.page.dfg, graph_fields=["edges"], node_fields=node_fields
                ),
            )

            self.assistant.add_message("user", locals)

        if selected_node is not None:
            self.assistant.add_message(
                "user",
                f"The currently selected node in the diagram is: `{selected_node}`",
            )

        self.visible_messages += [VisibleMessage(role="user", content=prompt)]
        self.assistant.add_message("system", config.get_prompt(system_prompt))
        self.assistant.add_message("user", prompt)

        for x in self.assistant.str_completion():
            markdown += x
            yield x

        with_embedded_images = self.assistant.replace_placeholders_with_base64_images(
            markdown
        )
        with_embedded_images = self.page.dfg.replace_placeholders_with_base64_images(
            with_embedded_images
        )

        self.visible_messages += [
            VisibleMessage(role="assistant", content=with_embedded_images)
        ]

    def last_message(self) -> VisibleMessage:
        return self.visible_messages[-1]

    def __len__(self) -> int:
        return len(self.visible_messages)

    def messages(self) -> Iterable[VisibleMessage]:
        for message in self.visible_messages:
            yield message

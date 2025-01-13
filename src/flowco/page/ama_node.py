from __future__ import annotations

from typing import Callable, Iterable, List, Literal, Tuple
from flowco.assistant.openai import OpenAIAssistant
from flowco.assistant.stream import StreamingAssistantWithFunctionCalls
from flowco.builder.graph_completions import messages_for_graph, messages_for_node
from flowco.dataflow.dfg import DataFlowGraph, Geometry, Node
from flowco.dataflow.extended_type import ExtendedType
from flowco.dataflow.phase import Phase
from flowco.page.page import Page
from flowco.pythonshell.shells import PythonShells
from flowco.session.session import session
from flowco.util.config import config
from flowco.util.errors import FlowcoError
from flowco.util.output import error, log, logger
from pydantic import BaseModel


class VisibleMessage(BaseModel):
    role: str
    content: str


ReturnType = Tuple[str, str | None]


class AskMeAnythingNode:

    def __init__(self, dfg: DataFlowGraph, show_code: bool):
        self.dfg = dfg
        self.show_code = show_code

        if show_code:
            prompt = "ama_node_editor"
            functions = [
                self.update_node,
            ]
        else:
            prompt = "ama_node_editor_no_code"
            functions = [self.update_node_requirements]

        self.assistant = StreamingAssistantWithFunctionCalls(
            functions,
            ["system-prompt", prompt],
            imports="",
        )
        self.completion_node = None
        self.visible_messages = []

    def update_node(
        self,
        label: str | None = None,
        requirements: List[str] | None = None,
        function_return_type: ExtendedType | None = None,
        code: List[str] | None = None,
    ) -> ReturnType:
        """\
        {
        "name": "update_node",
        "strict": true,
        "parameters": {
            "$defs": {
            "AnyType": {
                "properties": {
                "type": {
                    "const": "Any",
                    "title": "Type",
                    "type": "string"
                }
                },
                "required": [
                "type"
                ],
                "title": "AnyType",
                "type": "object",
                "additionalProperties": false
            },
            "BoolType": {
                "properties": {
                "type": {
                    "const": "bool",
                    "title": "Type",
                    "type": "string"
                }
                },
                "required": [
                "type"
                ],
                "title": "BoolType",
                "type": "object",
                "additionalProperties": false
            },
            "ExtendedType": {
                "properties": {
                "the_type": {
                    "anyOf": [
                    {
                        "$ref": "#/$defs/IntType"
                    },
                    {
                        "$ref": "#/$defs/BoolType"
                    },
                    {
                        "$ref": "#/$defs/StrType"
                    },
                    {
                        "$ref": "#/$defs/AnyType"
                    },
                    {
                        "$ref": "#/$defs/NoneType"
                    },
                    {
                        "$ref": "#/$defs/FloatType"
                    },
                    {
                        "$ref": "#/$defs/OptionalType"
                    },
                    {
                        "$ref": "#/$defs/ListType"
                    },
                    {
                        "$ref": "#/$defs/TypedDictType"
                    },
                    {
                        "$ref": "#/$defs/TupleType"
                    },
                    {
                        "$ref": "#/$defs/SetType"
                    },
                    {
                        "$ref": "#/$defs/PDDataFrameType"
                    },
                    {
                        "$ref": "#/$defs/PDSeriesType"
                    },
                    {
                        "$ref": "#/$defs/NumpyNdarrayType"
                    }
                    ],
                    "title": "The Type"
                },
                "description": {
                    "description": "A description of what this type represents. Indicate how to interpret each component of the type.",
                    "title": "Description",
                    "type": "string"
                }
                },
                "required": [
                "the_type",
                "description"
                ],
                "title": "ExtendedType",
                "type": "object",
                "additionalProperties": false
            },
            "FloatType": {
                "properties": {
                "type": {
                    "const": "float",
                    "title": "Type",
                    "type": "string"
                }
                },
                "required": [
                "type"
                ],
                "title": "FloatType",
                "type": "object",
                "additionalProperties": false
            },
            "IntType": {
                "properties": {
                "type": {
                    "const": "int",
                    "title": "Type",
                    "type": "string"
                }
                },
                "required": [
                "type"
                ],
                "title": "IntType",
                "type": "object",
                "additionalProperties": false
            },
            "KeyType": {
                "properties": {
                "key": {
                    "title": "Key",
                    "type": "string"
                },
                "type": {
                    "anyOf": [
                    {
                        "$ref": "#/$defs/IntType"
                    },
                    {
                        "$ref": "#/$defs/BoolType"
                    },
                    {
                        "$ref": "#/$defs/StrType"
                    },
                    {
                        "$ref": "#/$defs/AnyType"
                    },
                    {
                        "$ref": "#/$defs/NoneType"
                    },
                    {
                        "$ref": "#/$defs/FloatType"
                    },
                    {
                        "$ref": "#/$defs/OptionalType"
                    },
                    {
                        "$ref": "#/$defs/ListType"
                    },
                    {
                        "$ref": "#/$defs/TypedDictType"
                    },
                    {
                        "$ref": "#/$defs/TupleType"
                    },
                    {
                        "$ref": "#/$defs/SetType"
                    },
                    {
                        "$ref": "#/$defs/PDDataFrameType"
                    },
                    {
                        "$ref": "#/$defs/PDSeriesType"
                    },
                    {
                        "$ref": "#/$defs/NumpyNdarrayType"
                    }
                    ],
                    "title": "Type"
                },
                "description": {
                    "description": "What this key represents.",
                    "title": "Description",
                    "type": "string"
                }
                },
                "required": [
                "key",
                "type",
                "description"
                ],
                "title": "KeyType",
                "type": "object",
                "additionalProperties": false
            },
            "ListType": {
                "properties": {
                "type": {
                    "const": "List",
                    "title": "Type",
                    "type": "string"
                },
                "element_type": {
                    "anyOf": [
                    {
                        "$ref": "#/$defs/IntType"
                    },
                    {
                        "$ref": "#/$defs/BoolType"
                    },
                    {
                        "$ref": "#/$defs/StrType"
                    },
                    {
                        "$ref": "#/$defs/AnyType"
                    },
                    {
                        "$ref": "#/$defs/NoneType"
                    },
                    {
                        "$ref": "#/$defs/FloatType"
                    },
                    {
                        "$ref": "#/$defs/OptionalType"
                    },
                    {
                        "$ref": "#/$defs/ListType"
                    },
                    {
                        "$ref": "#/$defs/TypedDictType"
                    },
                    {
                        "$ref": "#/$defs/TupleType"
                    },
                    {
                        "$ref": "#/$defs/SetType"
                    },
                    {
                        "$ref": "#/$defs/PDDataFrameType"
                    },
                    {
                        "$ref": "#/$defs/PDSeriesType"
                    },
                    {
                        "$ref": "#/$defs/NumpyNdarrayType"
                    }
                    ],
                    "title": "Element Type"
                },
                "length": {
                    "anyOf": [
                    {
                        "type": "integer"
                    },
                    {
                        "type": "null"
                    }
                    ],
                    "description": "The expected length of the list. If None, the length can be arbitrary.",
                    "title": "Length"
                }
                },
                "required": [
                "type",
                "element_type",
                "length"
                ],
                "title": "ListType",
                "type": "object",
                "additionalProperties": false
            },
            "NoneType": {
                "properties": {
                "type": {
                    "const": "None",
                    "title": "Type",
                    "type": "string"
                }
                },
                "required": [
                "type"
                ],
                "title": "NoneType",
                "type": "object",
                "additionalProperties": false
            },
            "NumpyNdarrayType": {
                "properties": {
                "type": {
                    "const": "np.ndarray",
                    "title": "Type",
                    "type": "string"
                },
                "element_type": {
                    "anyOf": [
                    {
                        "$ref": "#/$defs/IntType"
                    },
                    {
                        "$ref": "#/$defs/BoolType"
                    },
                    {
                        "$ref": "#/$defs/StrType"
                    },
                    {
                        "$ref": "#/$defs/AnyType"
                    },
                    {
                        "$ref": "#/$defs/NoneType"
                    },
                    {
                        "$ref": "#/$defs/FloatType"
                    },
                    {
                        "$ref": "#/$defs/OptionalType"
                    },
                    {
                        "$ref": "#/$defs/ListType"
                    },
                    {
                        "$ref": "#/$defs/TypedDictType"
                    },
                    {
                        "$ref": "#/$defs/TupleType"
                    },
                    {
                        "$ref": "#/$defs/SetType"
                    },
                    {
                        "$ref": "#/$defs/PDDataFrameType"
                    },
                    {
                        "$ref": "#/$defs/PDSeriesType"
                    },
                    {
                        "$ref": "#/$defs/NumpyNdarrayType"
                    }
                    ],
                    "title": "Element Type"
                },
                "length": {
                    "anyOf": [
                    {
                        "type": "integer"
                    },
                    {
                        "type": "null"
                    }
                    ],
                    "description": "The expected length of the list. If None, the length can be arbitrary.",
                    "title": "Length"
                }
                },
                "required": [
                "type",
                "element_type",
                "length"
                ],
                "title": "NumpyNdarrayType",
                "type": "object",
                "additionalProperties": false
            },
            "OptionalType": {
                "properties": {
                "type": {
                    "const": "Optional",
                    "title": "Type",
                    "type": "string"
                },
                "wrapped_type": {
                    "anyOf": [
                    {
                        "$ref": "#/$defs/IntType"
                    },
                    {
                        "$ref": "#/$defs/BoolType"
                    },
                    {
                        "$ref": "#/$defs/StrType"
                    },
                    {
                        "$ref": "#/$defs/AnyType"
                    },
                    {
                        "$ref": "#/$defs/NoneType"
                    },
                    {
                        "$ref": "#/$defs/FloatType"
                    },
                    {
                        "$ref": "#/$defs/OptionalType"
                    },
                    {
                        "$ref": "#/$defs/ListType"
                    },
                    {
                        "$ref": "#/$defs/TypedDictType"
                    },
                    {
                        "$ref": "#/$defs/TupleType"
                    },
                    {
                        "$ref": "#/$defs/SetType"
                    },
                    {
                        "$ref": "#/$defs/PDDataFrameType"
                    },
                    {
                        "$ref": "#/$defs/PDSeriesType"
                    },
                    {
                        "$ref": "#/$defs/NumpyNdarrayType"
                    }
                    ],
                    "title": "Wrapped Type"
                }
                },
                "required": [
                "type",
                "wrapped_type"
                ],
                "title": "OptionalType",
                "type": "object",
                "additionalProperties": false
            },
            "PDDataFrameType": {
                "properties": {
                "type": {
                    "const": "pd.DataFrame",
                    "title": "Type",
                    "type": "string"
                },
                "columns": {
                    "description": "A list of key-value pairs where the key is the column name and the value is the type of the column.",
                    "items": {
                    "$ref": "#/$defs/KeyType"
                    },
                    "title": "Columns",
                    "type": "array"
                }
                },
                "required": [
                "type",
                "columns"
                ],
                "title": "PDDataFrameType",
                "type": "object",
                "additionalProperties": false
            },
            "PDSeriesType": {
                "properties": {
                "type": {
                    "const": "pd.Series",
                    "title": "Type",
                    "type": "string"
                },
                "element_type": {
                    "anyOf": [
                    {
                        "$ref": "#/$defs/IntType"
                    },
                    {
                        "$ref": "#/$defs/BoolType"
                    },
                    {
                        "$ref": "#/$defs/StrType"
                    },
                    {
                        "$ref": "#/$defs/AnyType"
                    },
                    {
                        "$ref": "#/$defs/NoneType"
                    },
                    {
                        "$ref": "#/$defs/FloatType"
                    },
                    {
                        "$ref": "#/$defs/OptionalType"
                    },
                    {
                        "$ref": "#/$defs/ListType"
                    },
                    {
                        "$ref": "#/$defs/TypedDictType"
                    },
                    {
                        "$ref": "#/$defs/TupleType"
                    },
                    {
                        "$ref": "#/$defs/SetType"
                    },
                    {
                        "$ref": "#/$defs/PDDataFrameType"
                    },
                    {
                        "$ref": "#/$defs/PDSeriesType"
                    },
                    {
                        "$ref": "#/$defs/NumpyNdarrayType"
                    }
                    ],
                    "title": "Element Type"
                }
                },
                "required": [
                "type",
                "element_type"
                ],
                "title": "PDSeriesType",
                "type": "object",
                "additionalProperties": false
            },
            "SetType": {
                "properties": {
                "type": {
                    "const": "Set",
                    "title": "Type",
                    "type": "string"
                },
                "element_type": {
                    "anyOf": [
                    {
                        "$ref": "#/$defs/IntType"
                    },
                    {
                        "$ref": "#/$defs/BoolType"
                    },
                    {
                        "$ref": "#/$defs/StrType"
                    },
                    {
                        "$ref": "#/$defs/AnyType"
                    },
                    {
                        "$ref": "#/$defs/NoneType"
                    },
                    {
                        "$ref": "#/$defs/FloatType"
                    },
                    {
                        "$ref": "#/$defs/OptionalType"
                    },
                    {
                        "$ref": "#/$defs/ListType"
                    },
                    {
                        "$ref": "#/$defs/TypedDictType"
                    },
                    {
                        "$ref": "#/$defs/TupleType"
                    },
                    {
                        "$ref": "#/$defs/SetType"
                    },
                    {
                        "$ref": "#/$defs/PDDataFrameType"
                    },
                    {
                        "$ref": "#/$defs/PDSeriesType"
                    },
                    {
                        "$ref": "#/$defs/NumpyNdarrayType"
                    }
                    ],
                    "title": "Element Type"
                }
                },
                "required": [
                "type",
                "element_type"
                ],
                "title": "SetType",
                "type": "object",
                "additionalProperties": false
            },
            "StrType": {
                "properties": {
                "type": {
                    "const": "str",
                    "title": "Type",
                    "type": "string"
                }
                },
                "required": [
                "type"
                ],
                "title": "StrType",
                "type": "object",
                "additionalProperties": false
            },
            "TupleType": {
                "properties": {
                "type": {
                    "const": "Tuple",
                    "title": "Type",
                    "type": "string"
                },
                "elements": {
                    "items": {
                    "anyOf": [
                        {
                        "$ref": "#/$defs/IntType"
                        },
                        {
                        "$ref": "#/$defs/BoolType"
                        },
                        {
                        "$ref": "#/$defs/StrType"
                        },
                        {
                        "$ref": "#/$defs/AnyType"
                        },
                        {
                        "$ref": "#/$defs/NoneType"
                        },
                        {
                        "$ref": "#/$defs/FloatType"
                        },
                        {
                        "$ref": "#/$defs/OptionalType"
                        },
                        {
                        "$ref": "#/$defs/ListType"
                        },
                        {
                        "$ref": "#/$defs/TypedDictType"
                        },
                        {
                        "$ref": "#/$defs/TupleType"
                        },
                        {
                        "$ref": "#/$defs/SetType"
                        },
                        {
                        "$ref": "#/$defs/PDDataFrameType"
                        },
                        {
                        "$ref": "#/$defs/PDSeriesType"
                        },
                        {
                        "$ref": "#/$defs/NumpyNdarrayType"
                        }
                    ]
                    },
                    "title": "Elements",
                    "type": "array"
                }
                },
                "required": [
                "type",
                "elements"
                ],
                "title": "TupleType",
                "type": "object",
                "additionalProperties": false
            },
            "TypedDictType": {
                "properties": {
                "type": {
                    "const": "TypedDict",
                    "title": "Type",
                    "type": "string"
                },
                "name": {
                    "description": "A unique name for the dictionary type. This is used to generate a unique TypedDict name.",
                    "title": "Name",
                    "type": "string"
                },
                "items": {
                    "description": "A list of key-value pairs where the key is the key name and the value is the type of the key.",
                    "items": {
                    "$ref": "#/$defs/KeyType"
                    },
                    "title": "Items",
                    "type": "array"
                }
                },
                "required": [
                "type",
                "name",
                "items"
                ],
                "title": "TypedDictType",
                "type": "object",
                "additionalProperties": false
            }
            },
            "properties": {
            "label": {
                "description": "The new label of the node.  Keep in sync with the requirements, algorithm, and code.", 
                "title": "Label", 
                "type": "string"
            },             
            "requirements": {
                "description": "A list of requirements that must be true of the return value for the function.  Describe the representation of the return value as well.",
                "items": {
                "type": "string"
                },
                "title": "Requirements",
                "type": "array"
            },
            "function_return_type": {
                "description": "The return type of the node.",
                "properties": {
                "the_type": {
                    "anyOf": [
                    {
                        "$ref": "#/$defs/IntType"
                    },
                    {
                        "$ref": "#/$defs/BoolType"
                    },
                    {
                        "$ref": "#/$defs/StrType"
                    },
                    {
                        "$ref": "#/$defs/AnyType"
                    },
                    {
                        "$ref": "#/$defs/NoneType"
                    },
                    {
                        "$ref": "#/$defs/FloatType"
                    },
                    {
                        "$ref": "#/$defs/OptionalType"
                    },
                    {
                        "$ref": "#/$defs/ListType"
                    },
                    {
                        "$ref": "#/$defs/TypedDictType"
                    },
                    {
                        "$ref": "#/$defs/TupleType"
                    },
                    {
                        "$ref": "#/$defs/SetType"
                    },
                    {
                        "$ref": "#/$defs/PDDataFrameType"
                    },
                    {
                        "$ref": "#/$defs/PDSeriesType"
                    },
                    {
                        "$ref": "#/$defs/NumpyNdarrayType"
                    }
                    ],
                    "title": "The Type"
                },
                "description": {
                    "description": "A description of what this type represents. Indicate how to interpret each component of the type.",
                    "title": "Description",
                    "type": "string"
                }
                },
                "required": [
                "the_type",
                "description"
                ],
                "title": "ExtendedType",
                "type": "object",
                "additionalProperties": false
            },
            "code": {
                "description": "The code for the node.  Only modify if there is already an code.  The code should be a list of strings, one for each line of code.  The signature must match the original version, except for the return type",
                "items": {
                "type": "string"
                },
                "title": "Code",
                "type": "array"
            }
            },
            "required": [
            "label",
            "requirements",
            "function_return_type",
            "code"
            ],
            "title": "update_node",
            "type": "object",
            "additionalProperties": false
        }
        }
        """
        node = self.completion_node
        assert node is not None, "Node must be set before calling update_node"
        mods = []
        log(f"update_node: {label}, {requirements}, {function_return_type}, {code}")
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
                    f"Updating function_return_type from {node.function_return_type} to {function_return_type}"
                )
                node = node.update(
                    function_return_type=function_return_type,
                    phase=Phase.clean,
                )
            if "requirements" not in mods:
                mods.append("requirements")
        if label and label != node.label:
            log(f"Updating label to {label}")
            node = node.update(label=label, phase=Phase.clean)
            mods.append("label")

        if config.x_trust_ama:
            if "requirements" in mods:
                node = node.update(cache=node.cache.update(Phase.requirements, node))
                node = node.update(phase=Phase.requirements)
            if "algorithm" in mods:
                node = node.update(cache=node.cache.update(Phase.algorithm, node))
                node = node.update(phase=Phase.algorithm)
            if "code" in mods:
                node = node.update(cache=node.cache.update(Phase.code, node))
                node = node.update(phase=Phase.code)

        mod_str = ", ".join(reversed(mods))

        self.completion_node = node

        return (
            f"Updated {mod_str} for {node.pill}",
            node.model_dump_json(indent=2),
        )

    def update_node_requirements(
        self,
        label: str | None = None,
        requirements: List[str] | None = None,
        function_return_type: ExtendedType | None = None,
    ) -> ReturnType:
        """\
        {
        "name": "update_node_requirements",
        "strict": true,
        "parameters": {
            "$defs": {
            "AnyType": {
                "properties": {
                "type": {
                    "const": "Any",
                    "title": "Type",
                    "type": "string"
                }
                },
                "required": [
                "type"
                ],
                "title": "AnyType",
                "type": "object",
                "additionalProperties": false
            },
            "BoolType": {
                "properties": {
                "type": {
                    "const": "bool",
                    "title": "Type",
                    "type": "string"
                }
                },
                "required": [
                "type"
                ],
                "title": "BoolType",
                "type": "object",
                "additionalProperties": false
            },
            "ExtendedType": {
                "properties": {
                "the_type": {
                    "anyOf": [
                    {
                        "$ref": "#/$defs/IntType"
                    },
                    {
                        "$ref": "#/$defs/BoolType"
                    },
                    {
                        "$ref": "#/$defs/StrType"
                    },
                    {
                        "$ref": "#/$defs/AnyType"
                    },
                    {
                        "$ref": "#/$defs/NoneType"
                    },
                    {
                        "$ref": "#/$defs/FloatType"
                    },
                    {
                        "$ref": "#/$defs/OptionalType"
                    },
                    {
                        "$ref": "#/$defs/ListType"
                    },
                    {
                        "$ref": "#/$defs/TypedDictType"
                    },
                    {
                        "$ref": "#/$defs/TupleType"
                    },
                    {
                        "$ref": "#/$defs/SetType"
                    },
                    {
                        "$ref": "#/$defs/PDDataFrameType"
                    },
                    {
                        "$ref": "#/$defs/PDSeriesType"
                    },
                    {
                        "$ref": "#/$defs/NumpyNdarrayType"
                    }
                    ],
                    "title": "The Type"
                },
                "description": {
                    "description": "A description of what this type represents. Indicate how to interpret each component of the type.",
                    "title": "Description",
                    "type": "string"
                }
                },
                "required": [
                "the_type",
                "description"
                ],
                "title": "ExtendedType",
                "type": "object",
                "additionalProperties": false
            },
            "FloatType": {
                "properties": {
                "type": {
                    "const": "float",
                    "title": "Type",
                    "type": "string"
                }
                },
                "required": [
                "type"
                ],
                "title": "FloatType",
                "type": "object",
                "additionalProperties": false
            },
            "IntType": {
                "properties": {
                "type": {
                    "const": "int",
                    "title": "Type",
                    "type": "string"
                }
                },
                "required": [
                "type"
                ],
                "title": "IntType",
                "type": "object",
                "additionalProperties": false
            },
            "KeyType": {
                "properties": {
                "key": {
                    "title": "Key",
                    "type": "string"
                },
                "type": {
                    "anyOf": [
                    {
                        "$ref": "#/$defs/IntType"
                    },
                    {
                        "$ref": "#/$defs/BoolType"
                    },
                    {
                        "$ref": "#/$defs/StrType"
                    },
                    {
                        "$ref": "#/$defs/AnyType"
                    },
                    {
                        "$ref": "#/$defs/NoneType"
                    },
                    {
                        "$ref": "#/$defs/FloatType"
                    },
                    {
                        "$ref": "#/$defs/OptionalType"
                    },
                    {
                        "$ref": "#/$defs/ListType"
                    },
                    {
                        "$ref": "#/$defs/TypedDictType"
                    },
                    {
                        "$ref": "#/$defs/TupleType"
                    },
                    {
                        "$ref": "#/$defs/SetType"
                    },
                    {
                        "$ref": "#/$defs/PDDataFrameType"
                    },
                    {
                        "$ref": "#/$defs/PDSeriesType"
                    },
                    {
                        "$ref": "#/$defs/NumpyNdarrayType"
                    }
                    ],
                    "title": "Type"
                },
                "description": {
                    "description": "What this key represents.",
                    "title": "Description",
                    "type": "string"
                }
                },
                "required": [
                "key",
                "type",
                "description"
                ],
                "title": "KeyType",
                "type": "object",
                "additionalProperties": false
            },
            "ListType": {
                "properties": {
                "type": {
                    "const": "List",
                    "title": "Type",
                    "type": "string"
                },
                "element_type": {
                    "anyOf": [
                    {
                        "$ref": "#/$defs/IntType"
                    },
                    {
                        "$ref": "#/$defs/BoolType"
                    },
                    {
                        "$ref": "#/$defs/StrType"
                    },
                    {
                        "$ref": "#/$defs/AnyType"
                    },
                    {
                        "$ref": "#/$defs/NoneType"
                    },
                    {
                        "$ref": "#/$defs/FloatType"
                    },
                    {
                        "$ref": "#/$defs/OptionalType"
                    },
                    {
                        "$ref": "#/$defs/ListType"
                    },
                    {
                        "$ref": "#/$defs/TypedDictType"
                    },
                    {
                        "$ref": "#/$defs/TupleType"
                    },
                    {
                        "$ref": "#/$defs/SetType"
                    },
                    {
                        "$ref": "#/$defs/PDDataFrameType"
                    },
                    {
                        "$ref": "#/$defs/PDSeriesType"
                    },
                    {
                        "$ref": "#/$defs/NumpyNdarrayType"
                    }
                    ],
                    "title": "Element Type"
                },
                "length": {
                    "anyOf": [
                    {
                        "type": "integer"
                    },
                    {
                        "type": "null"
                    }
                    ],
                    "description": "The expected length of the list. If None, the length can be arbitrary.",
                    "title": "Length"
                }
                },
                "required": [
                "type",
                "element_type",
                "length"
                ],
                "title": "ListType",
                "type": "object",
                "additionalProperties": false
            },
            "NoneType": {
                "properties": {
                "type": {
                    "const": "None",
                    "title": "Type",
                    "type": "string"
                }
                },
                "required": [
                "type"
                ],
                "title": "NoneType",
                "type": "object",
                "additionalProperties": false
            },
            "NumpyNdarrayType": {
                "properties": {
                "type": {
                    "const": "np.ndarray",
                    "title": "Type",
                    "type": "string"
                },
                "element_type": {
                    "anyOf": [
                    {
                        "$ref": "#/$defs/IntType"
                    },
                    {
                        "$ref": "#/$defs/BoolType"
                    },
                    {
                        "$ref": "#/$defs/StrType"
                    },
                    {
                        "$ref": "#/$defs/AnyType"
                    },
                    {
                        "$ref": "#/$defs/NoneType"
                    },
                    {
                        "$ref": "#/$defs/FloatType"
                    },
                    {
                        "$ref": "#/$defs/OptionalType"
                    },
                    {
                        "$ref": "#/$defs/ListType"
                    },
                    {
                        "$ref": "#/$defs/TypedDictType"
                    },
                    {
                        "$ref": "#/$defs/TupleType"
                    },
                    {
                        "$ref": "#/$defs/SetType"
                    },
                    {
                        "$ref": "#/$defs/PDDataFrameType"
                    },
                    {
                        "$ref": "#/$defs/PDSeriesType"
                    },
                    {
                        "$ref": "#/$defs/NumpyNdarrayType"
                    }
                    ],
                    "title": "Element Type"
                },
                "length": {
                    "anyOf": [
                    {
                        "type": "integer"
                    },
                    {
                        "type": "null"
                    }
                    ],
                    "description": "The expected length of the list. If None, the length can be arbitrary.",
                    "title": "Length"
                }
                },
                "required": [
                "type",
                "element_type",
                "length"
                ],
                "title": "NumpyNdarrayType",
                "type": "object",
                "additionalProperties": false
            },
            "OptionalType": {
                "properties": {
                "type": {
                    "const": "Optional",
                    "title": "Type",
                    "type": "string"
                },
                "wrapped_type": {
                    "anyOf": [
                    {
                        "$ref": "#/$defs/IntType"
                    },
                    {
                        "$ref": "#/$defs/BoolType"
                    },
                    {
                        "$ref": "#/$defs/StrType"
                    },
                    {
                        "$ref": "#/$defs/AnyType"
                    },
                    {
                        "$ref": "#/$defs/NoneType"
                    },
                    {
                        "$ref": "#/$defs/FloatType"
                    },
                    {
                        "$ref": "#/$defs/OptionalType"
                    },
                    {
                        "$ref": "#/$defs/ListType"
                    },
                    {
                        "$ref": "#/$defs/TypedDictType"
                    },
                    {
                        "$ref": "#/$defs/TupleType"
                    },
                    {
                        "$ref": "#/$defs/SetType"
                    },
                    {
                        "$ref": "#/$defs/PDDataFrameType"
                    },
                    {
                        "$ref": "#/$defs/PDSeriesType"
                    },
                    {
                        "$ref": "#/$defs/NumpyNdarrayType"
                    }
                    ],
                    "title": "Wrapped Type"
                }
                },
                "required": [
                "type",
                "wrapped_type"
                ],
                "title": "OptionalType",
                "type": "object",
                "additionalProperties": false
            },
            "PDDataFrameType": {
                "properties": {
                "type": {
                    "const": "pd.DataFrame",
                    "title": "Type",
                    "type": "string"
                },
                "columns": {
                    "description": "A list of key-value pairs where the key is the column name and the value is the type of the column.",
                    "items": {
                    "$ref": "#/$defs/KeyType"
                    },
                    "title": "Columns",
                    "type": "array"
                }
                },
                "required": [
                "type",
                "columns"
                ],
                "title": "PDDataFrameType",
                "type": "object",
                "additionalProperties": false
            },
            "PDSeriesType": {
                "properties": {
                "type": {
                    "const": "pd.Series",
                    "title": "Type",
                    "type": "string"
                },
                "element_type": {
                    "anyOf": [
                    {
                        "$ref": "#/$defs/IntType"
                    },
                    {
                        "$ref": "#/$defs/BoolType"
                    },
                    {
                        "$ref": "#/$defs/StrType"
                    },
                    {
                        "$ref": "#/$defs/AnyType"
                    },
                    {
                        "$ref": "#/$defs/NoneType"
                    },
                    {
                        "$ref": "#/$defs/FloatType"
                    },
                    {
                        "$ref": "#/$defs/OptionalType"
                    },
                    {
                        "$ref": "#/$defs/ListType"
                    },
                    {
                        "$ref": "#/$defs/TypedDictType"
                    },
                    {
                        "$ref": "#/$defs/TupleType"
                    },
                    {
                        "$ref": "#/$defs/SetType"
                    },
                    {
                        "$ref": "#/$defs/PDDataFrameType"
                    },
                    {
                        "$ref": "#/$defs/PDSeriesType"
                    },
                    {
                        "$ref": "#/$defs/NumpyNdarrayType"
                    }
                    ],
                    "title": "Element Type"
                }
                },
                "required": [
                "type",
                "element_type"
                ],
                "title": "PDSeriesType",
                "type": "object",
                "additionalProperties": false
            },
            "SetType": {
                "properties": {
                "type": {
                    "const": "Set",
                    "title": "Type",
                    "type": "string"
                },
                "element_type": {
                    "anyOf": [
                    {
                        "$ref": "#/$defs/IntType"
                    },
                    {
                        "$ref": "#/$defs/BoolType"
                    },
                    {
                        "$ref": "#/$defs/StrType"
                    },
                    {
                        "$ref": "#/$defs/AnyType"
                    },
                    {
                        "$ref": "#/$defs/NoneType"
                    },
                    {
                        "$ref": "#/$defs/FloatType"
                    },
                    {
                        "$ref": "#/$defs/OptionalType"
                    },
                    {
                        "$ref": "#/$defs/ListType"
                    },
                    {
                        "$ref": "#/$defs/TypedDictType"
                    },
                    {
                        "$ref": "#/$defs/TupleType"
                    },
                    {
                        "$ref": "#/$defs/SetType"
                    },
                    {
                        "$ref": "#/$defs/PDDataFrameType"
                    },
                    {
                        "$ref": "#/$defs/PDSeriesType"
                    },
                    {
                        "$ref": "#/$defs/NumpyNdarrayType"
                    }
                    ],
                    "title": "Element Type"
                }
                },
                "required": [
                "type",
                "element_type"
                ],
                "title": "SetType",
                "type": "object",
                "additionalProperties": false
            },
            "StrType": {
                "properties": {
                "type": {
                    "const": "str",
                    "title": "Type",
                    "type": "string"
                }
                },
                "required": [
                "type"
                ],
                "title": "StrType",
                "type": "object",
                "additionalProperties": false
            },
            "TupleType": {
                "properties": {
                "type": {
                    "const": "Tuple",
                    "title": "Type",
                    "type": "string"
                },
                "elements": {
                    "items": {
                    "anyOf": [
                        {
                        "$ref": "#/$defs/IntType"
                        },
                        {
                        "$ref": "#/$defs/BoolType"
                        },
                        {
                        "$ref": "#/$defs/StrType"
                        },
                        {
                        "$ref": "#/$defs/AnyType"
                        },
                        {
                        "$ref": "#/$defs/NoneType"
                        },
                        {
                        "$ref": "#/$defs/FloatType"
                        },
                        {
                        "$ref": "#/$defs/OptionalType"
                        },
                        {
                        "$ref": "#/$defs/ListType"
                        },
                        {
                        "$ref": "#/$defs/TypedDictType"
                        },
                        {
                        "$ref": "#/$defs/TupleType"
                        },
                        {
                        "$ref": "#/$defs/SetType"
                        },
                        {
                        "$ref": "#/$defs/PDDataFrameType"
                        },
                        {
                        "$ref": "#/$defs/PDSeriesType"
                        },
                        {
                        "$ref": "#/$defs/NumpyNdarrayType"
                        }
                    ]
                    },
                    "title": "Elements",
                    "type": "array"
                }
                },
                "required": [
                "type",
                "elements"
                ],
                "title": "TupleType",
                "type": "object",
                "additionalProperties": false
            },
            "TypedDictType": {
                "properties": {
                "type": {
                    "const": "TypedDict",
                    "title": "Type",
                    "type": "string"
                },
                "name": {
                    "description": "A unique name for the dictionary type. This is used to generate a unique TypedDict name.",
                    "title": "Name",
                    "type": "string"
                },
                "items": {
                    "description": "A list of key-value pairs where the key is the key name and the value is the type of the key.",
                    "items": {
                    "$ref": "#/$defs/KeyType"
                    },
                    "title": "Items",
                    "type": "array"
                }
                },
                "required": [
                "type",
                "name",
                "items"
                ],
                "title": "TypedDictType",
                "type": "object",
                "additionalProperties": false
            }
            },
            "properties": {
            "label": {
                "description": "The new label of the node.  Keep in sync with the requirements, algorithm, and code.", 
                "title": "Label", 
                "type": "string"
            },             
            "requirements": {
                "description": "A list of requirements that must be true of the return value for the function.  Describe the representation of the return value as well.",
                "items": {
                "type": "string"
                },
                "title": "Requirements",
                "type": "array"
            },
            "function_return_type": {
                "description": "The return type of the node.",
                "properties": {
                "the_type": {
                    "anyOf": [
                    {
                        "$ref": "#/$defs/IntType"
                    },
                    {
                        "$ref": "#/$defs/BoolType"
                    },
                    {
                        "$ref": "#/$defs/StrType"
                    },
                    {
                        "$ref": "#/$defs/AnyType"
                    },
                    {
                        "$ref": "#/$defs/NoneType"
                    },
                    {
                        "$ref": "#/$defs/FloatType"
                    },
                    {
                        "$ref": "#/$defs/OptionalType"
                    },
                    {
                        "$ref": "#/$defs/ListType"
                    },
                    {
                        "$ref": "#/$defs/TypedDictType"
                    },
                    {
                        "$ref": "#/$defs/TupleType"
                    },
                    {
                        "$ref": "#/$defs/SetType"
                    },
                    {
                        "$ref": "#/$defs/PDDataFrameType"
                    },
                    {
                        "$ref": "#/$defs/PDSeriesType"
                    },
                    {
                        "$ref": "#/$defs/NumpyNdarrayType"
                    }
                    ],
                    "title": "The Type"
                },
                "description": {
                    "description": "A description of what this type represents. Indicate how to interpret each component of the type.",
                    "title": "Description",
                    "type": "string"
                }
                },
                "required": [
                "the_type",
                "description"
                ],
                "title": "ExtendedType",
                "type": "object",
                "additionalProperties": false
            }
            },
            "required": [
            "label",
            "requirements",
            "function_return_type"
            ],
            "title": "update_node",
            "type": "object",
            "additionalProperties": false
        }
        }
        """
        node = self.completion_node
        assert node is not None, "Node must be set before calling update_node"
        mods = []
        log(
            f"update_node_requirements: {label}, {requirements}, {function_return_type}"
        )
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
            if "requirements" not in mods:
                mods.append("requirements")
        if label and label != node.label:
            log(f"Updating label to {label}")
            node = node.update(label=label, phase=Phase.clean)
            mods.append("label")

        if config.x_trust_ama:
            if "requirements" in mods:
                node = node.update(cache=node.cache.update(Phase.requirements, node))
                node = node.update(phase=Phase.requirements)
            if "algorithm" in mods:
                node = node.update(cache=node.cache.update(Phase.algorithm, node))
                node = node.update(phase=Phase.algorithm)

        mod_str = ", ".join(reversed(mods))

        self.completion_node = node

        return (
            f"Updated {mod_str} for {node.pill}",
            node.model_dump_json(indent=2),
        )

    def complete(
        self, prompt: str, node: Node, show_prompt: bool = True
    ) -> Iterable[str]:
        # try:
        yield from self._complete(prompt, node, show_prompt)

    # except Exception as e:
    #     error(f"Error: {e}")
    #     raise FlowcoError(f"Error: {e}")

    def _complete(
        self,
        prompt: str,
        node: Node,
        show_prompt: bool,
    ) -> Iterable[str]:

        markdown = ""

        if self.completion_node is None:
            self.assistant.add_message(
                "user",
                messages_for_graph(
                    graph=self.dfg,
                    graph_fields=["edges", "description"],
                    node_fields=["requirements"],
                ),
            )

            image = self.dfg.to_image_prompt_messages()
            self.assistant.add_message("user", image)

        self.completion_node = node
        self.assistant.add_message(
            "user",
            messages_for_node(
                node=self.completion_node,
                node_fields=(
                    [
                        "id",
                        "pill",
                        "label",
                        "predecessors",
                        "preconditions",
                        "requirements",
                        "function_return_type",
                    ]
                    + (["code"] if self.show_code else [])
                ),
            ),
        )

        if show_prompt:
            self.visible_messages += [VisibleMessage(role="user", content=prompt)]
        else:
            yield "Working...\n\n"

        self.assistant.add_message("user", prompt)

        for x in self.assistant.str_completion():
            markdown += x
            yield x

        self.visible_messages += [VisibleMessage(role="assistant", content=markdown)]

    def updated_node(self) -> Node | None:
        return self.completion_node

    def last_message(self) -> VisibleMessage:
        return self.visible_messages[-1]

    def __len__(self) -> int:
        return len(self.visible_messages)

    def messages(self) -> Iterable[VisibleMessage]:
        for message in self.visible_messages:
            yield message

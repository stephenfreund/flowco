from typing import Any, Iterable, Set, Tuple, Union, Dict, List, Literal, TypedDict
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from abc import abstractmethod


# Base class for all type representations with required methods
class BaseType(BaseModel):
    def to_python_type(self) -> str:
        raise NotImplementedError("to_python_type method not implemented.")

    def to_markdown(self, indent: int = 0) -> List[str]:
        raise NotImplementedError("to_markdown method not implemented.")

    def __str__(self) -> str:
        raise NotImplementedError("__str__ method not implemented.")

    @abstractmethod
    def validate(self, value: Any) -> bool:
        """Validate whether the given value conforms to the type."""
        pass


class IntType(BaseType):
    type: Literal["int"] #= "int"

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "int"
        super().__init__(**data)

    def to_python_type(self) -> str:
        return "int"

    def to_markdown(self, indent: int = 0) -> List[str]:
        spaces = "  " * indent
        return [f"{spaces}- **Integer**"]

    def __str__(self) -> str:
        return "int"

    def validate(self, value: Any) -> bool:
        return isinstance(value, int) and not isinstance(
            value, bool
        )  # bool is subclass of int


class BoolType(BaseType):
    type: Literal["bool"] #= "bool"

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "bool"
        super().__init__(**data)

    def to_python_type(self) -> str:
        return "bool"

    def to_markdown(self, indent: int = 0) -> List[str]:
        spaces = "  " * indent
        return [f"{spaces}- **Boolean**"]

    def __str__(self) -> str:
        return "bool"

    def validate(self, value: Any) -> bool:
        return isinstance(value, bool)


class StrType(BaseType):
    type: Literal["str"] #= "str"

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "str"
        super().__init__(**data)

    def to_python_type(self) -> str:
        return "str"

    def to_markdown(self, indent: int = 0) -> List[str]:
        spaces = "  " * indent
        return [f"{spaces}- **String**"]

    def __str__(self) -> str:
        return "str"

    def validate(self, value: Any) -> bool:
        return isinstance(value, str)


class AnyType(BaseType):
    type: Literal["Any"] #= "Any"

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "Any"
        super().__init__(**data)

    def to_python_type(self) -> str:
        return "Any"

    def to_markdown(self, indent: int = 0) -> List[str]:
        spaces = "  " * indent
        return [f"{spaces}- **Any**"]

    def __str__(self) -> str:
        return "Any"

    def validate(self, value: Any) -> bool:
        return True


class NoneType(BaseType):
    type: Literal["None"] #= "None"

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "None"
        super().__init__(**data)

    def to_python_type(self) -> str:
        return "None"

    def to_markdown(self, indent: int = 0) -> List[str]:
        spaces = "  " * indent
        return [f"{spaces}- **None**"]

    def __str__(self) -> str:
        return "None"

    def validate(self, value: Any) -> bool:
        return value is None


class FloatType(BaseType):
    type: Literal["float"] #= "float"

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "float"
        super().__init__(**data)

    def to_python_type(self) -> str:
        return "float"

    def to_markdown(self, indent: int = 0) -> List[str]:
        spaces = "  " * indent
        return [f"{spaces}- **Float**"]

    def __str__(self) -> str:
        return "float"

    def validate(self, value: Any) -> bool:
        return isinstance(value, float)


class OptionalType(BaseType):
    type: Literal["Optional"] #= "Optional"
    wrapped_type: "TypeRepresentation"

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "Optional"
        super().__init__(**data)

    def to_python_type(self) -> str:
        return f"Optional[{self.wrapped_type.to_python_type()}]"

    def to_markdown(self, indent: int = 0) -> List[str]:
        spaces = "  " * indent
        wrapped_markdown = self.wrapped_type.to_markdown(indent + 1)
        if len(wrapped_markdown) == 1:
            return [f"{spaces}- **Optional** of {wrapped_markdown[0].strip('- ')}"]
        return [f"{spaces}- **Optional**:"] + wrapped_markdown

    def __str__(self) -> str:
        return f"Optional[{self.wrapped_type}]"

    def validate(self, value: Any) -> bool:
        if value is None:
            return True
        return self.wrapped_type.validate(value)


class KeyType(BaseModel):
    key: str
    type: "TypeRepresentation"
    description: str = Field(description="What this key represents.")

    def to_python_type(self) -> str:
        return self.type.to_python_type()

    def to_markdown(self, indent: int = 0) -> List[str]:
        spaces = "  " * indent
        type_markdown = self.type.to_markdown(indent + 1)
        if len(type_markdown) == 1:
            return [
                f"{spaces}- `{self.key}`: {self.description} ({type_markdown[0].strip('- ')})"
            ]
        return [f"{spaces}- `{self.key}`: {self.description}"] + type_markdown

    def __str__(self) -> str:
        return f"{self.key}: {self.type}"

    def validate(self, value: Any) -> bool:
        return self.type.validate(value)


class ListType(BaseType):
    type: Literal["List"] #= "List"
    element_type: "TypeRepresentation"
    length: int | None = Field(
        description="The expected length of the list. If None, the length can be arbitrary."
    )

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "List"
        super().__init__(**data)

    def to_python_type(self) -> str:
        return f"List[{self.element_type.to_python_type()}]"

    def to_markdown(self, indent: int = 0) -> List[str]:
        spaces = "  " * indent
        element_markdown = self.element_type.to_markdown(indent + 1)
        if len(element_markdown) == 1:
            return [f"{spaces}- **List** of {element_markdown[0].strip('- ')}"]
        return [f"{spaces}- **List**:"] + element_markdown

    def __str__(self) -> str:
        return f"List[{self.element_type}]"

    def validate(self, value: Any) -> bool:
        if not isinstance(value, list):
            return False
        if self.length is not None and len(value) != self.length:
            return False
        return all(self.element_type.validate(elem) for elem in value)


class TypedDictType(BaseType):
    type: Literal["TypedDict"] #= "TypedDict"
    name: str = Field(
        description="A unique name for the dictionary type. This is used to generate a unique TypedDict name."
    )
    items: List[KeyType] = Field(
        description="A list of key-value pairs where the key is the key name and the value is the type of the key."
    )

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "TypedDict"
        super().__init__(**data)

    def to_python_type(self) -> str:
        elems = [f"'{item.key}': {item.type.to_python_type()}" for item in self.items]
        map = ", ".join(elems)
        return f"TypedDict('{self.name}', {{{map}}})"

    def to_markdown(self, indent: int = 0) -> List[str]:
        if not self.items:
            return ["- **Dictionary** (empty)"]
        spaces = "  " * indent
        desc = [f"{spaces}- **Dictionary** with keys:"]
        for item in self.items:
            desc.extend(item.to_markdown(indent + 1))
        return desc

    def __str__(self) -> str:
        elems = [f"'{item.key}': {item.type.to_python_type()}" for item in self.items]
        map = ", ".join(elems)
        return f"TypedDict('{self.name}', {{{map}}})"

    def validate(self, value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        for item in self.items:
            if item.key not in value:
                return False
            if not item.type.validate(value[item.key]):
                return False
        return True


class TupleType(BaseType):
    type: Literal["Tuple"] #= "Tuple"
    elements: List["TypeRepresentation"]

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "Tuple"
        super().__init__(**data)

    def to_python_type(self) -> str:
        elements_str = ", ".join([elem.to_python_type() for elem in self.elements])
        return f"Tuple[{elements_str}]"

    def to_markdown(self, indent: int = 0) -> List[str]:
        spaces = "  " * indent
        desc = [f"{spaces}- **Tuple** containing:"]
        for elem in self.elements:
            desc.extend(elem.to_markdown(indent + 1))
        return desc

    def __str__(self) -> str:
        elements_str = ", ".join([str(elem) for elem in self.elements])
        return f"Tuple[{elements_str}]"

    def validate(self, value: Any) -> bool:
        if not isinstance(value, tuple):
            return False
        if len(value) != len(self.elements):
            return False
        return all(elem_type.validate(v) for elem_type, v in zip(self.elements, value))


class SetType(BaseType):
    type: Literal["Set"] #= "Set"
    element_type: "TypeRepresentation"

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "Set"
        super().__init__(**data)

    def to_python_type(self) -> str:
        return f"Set[{self.element_type.to_python_type()}]"

    def to_markdown(self, indent: int = 0) -> List[str]:
        spaces = "  " * indent
        element_markdown = self.element_type.to_markdown(indent + 1)
        if len(element_markdown) == 1:
            return [f"{spaces}- **Set** of {element_markdown[0].strip('- ')}"]
        return [f"{spaces}- **Set**:"] + element_markdown

    def __str__(self) -> str:
        return f"Set[{self.element_type}]"

    def validate(self, value: Any) -> bool:
        if not isinstance(value, set):
            return False
        return all(self.element_type.validate(elem) for elem in value)


class PDDataFrameType(BaseType):
    type: Literal["pd.DataFrame"] #= "pd.DataFrame"
    columns: List[KeyType] = Field(
        description="A list of key-value pairs where the key is the column name and the value is the type of the column."
    )

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "pd.DataFrame"
        super().__init__(**data)

    def to_python_type(self) -> str:
        return "pd.DataFrame"

    def to_markdown(self, indent: int = 0) -> List[str]:
        spaces = "  " * indent
        desc = [f"{spaces}- **Pandas DataFrame** with columns:"]
        for col in self.columns:
            desc.extend(col.to_markdown(indent + 1))
        return desc

    def __str__(self) -> str:
        column_types = ", ".join([f"{col}" for col in self.columns])
        return f"pd.DataFrame[{column_types}]"

    def validate(self, value: Any) -> bool:
        if not isinstance(value, pd.DataFrame):
            return False
        for col in self.columns:
            if col.key not in value.columns:
                return False
            # Check each element in the column
            for item in value[col.key]:
                if not col.type.validate(item):
                    return False
        return True


class PDSeriesType(BaseType):
    type: Literal["pd.Series"] #= "pd.Series"
    element_type: "TypeRepresentation"

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "pd.Series"
        super().__init__(**data)

    def to_python_type(self) -> str:
        return "pd.Series"

    def to_markdown(self, indent: int = 0) -> List[str]:
        spaces = "  " * indent
        element_markdown = self.element_type.to_markdown(indent + 1)
        if len(element_markdown) == 1:
            return [f"{spaces}- **Pandas Series** of {element_markdown[0].strip('- ')}"]
        return [f"{spaces}- **Pandas Series**:"] + element_markdown

    def __str__(self) -> str:
        return f"pd.Series[{self.element_type}]"

    def validate(self, value: Any) -> bool:
        if not isinstance(value, pd.Series):
            return False
        for item in value:
            if not self.element_type.validate(item):
                return False
        return True


class NumpyNdarrayType(BaseType):
    type: Literal["np.ndarray"] # = "np.ndarray"
    element_type: "TypeRepresentation"
    length: int | None = Field(
        description="The expected length of the list. If None, the length can be arbitrary."
    )

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "np.ndarray"
        super().__init__(**data)

    def to_python_type(self) -> str:
        return "np.ndarray"

    def to_markdown(self, indent: int = 0) -> List[str]:
        spaces = "  " * indent
        element_markdown = self.element_type.to_markdown(indent + 1)
        if len(element_markdown) == 1:
            return [f"{spaces}- **NumPy ndarray** of {element_markdown[0].strip('- ')}"]
        return [f"{spaces}- **NumPy ndarray**:"] + element_markdown

    def __str__(self) -> str:
        return f"np.ndarray[{self.element_type}]"

    def validate(self, value: Any) -> bool:
        if not isinstance(value, np.ndarray):
            return False
        if self.length is not None and value.size != self.length:
            return False
        return all(self.element_type.validate(elem) for elem in value.flat)


# Update TypeRepresentation to use the updated classes
TypeRepresentation = Union[
    IntType,
    BoolType,
    StrType,
    AnyType,
    NoneType,
    FloatType,
    OptionalType,
    ListType,
    TypedDictType,
    TupleType,
    SetType,
    PDDataFrameType,
    PDSeriesType,
    NumpyNdarrayType,
]


class ExtendedType(BaseModel):
    the_type: TypeRepresentation
    description: str = Field(
        description="A description of what this type represents. Indicate how to interpret each component of the type."
    )

    def to_python_type(self) -> str:
        """
        Converts the custom TypeRepresentation into a string representation of the Python type.
        """
        return self.the_type.to_python_type()

    def to_markdown(self) -> str:
        """
        Generate a human-readable Markdown description for the type.
        """
        lines = [self.description] + self.the_type.to_markdown()
        return "\n".join(lines)

    def __str__(self) -> str:
        """
        Returns a string combining the Python type with extra composite type information.
        """
        return str(self.the_type)

    def is_None_type(self) -> bool:
        return isinstance(self.the_type, NoneType)

    @classmethod
    def from_value(cls, value) -> "ExtendedType":

        def infer_type_of_elements(elements: Iterable) -> TypeRepresentation:
            element_types = [infer_type(item) for item in elements]
            unique_types = {et.model_dump_json() for et in element_types}
            if len(unique_types) == 1:
                return element_types[0]
            else:
                return AnyType()

        def infer_type(val) -> TypeRepresentation:
            if val is None:
                return NoneType()
            elif isinstance(val, bool):  # do before int
                return BoolType()
            elif isinstance(val, int):
                return IntType()
            elif isinstance(val, float):
                return FloatType()
            elif isinstance(val, str):
                return StrType()
            elif isinstance(val, list):
                try:
                    element_type = infer_type_of_elements(val)
                    return ListType(element_type=element_type,length=len(val))
                except AnyType:
                    return AnyType()
            elif isinstance(val, set):
                try:
                    element_type = infer_type_of_elements(val)
                    return SetType(element_type=element_type)
                except AnyType:
                    return AnyType()
            elif isinstance(val, tuple):
                return TupleType(elements=[infer_type(item) for item in val])
            elif isinstance(val, dict):
                items = [
                    KeyType(key=k, type=infer_type(v), description="")
                    for k, v in val.items()
                ]
                return TypedDictType(name="AutoGeneratedTypedDict", items=items)
            elif isinstance(val, pd.DataFrame):
                if val.empty:
                    return PDDataFrameType(columns=[])
                columns = [
                    KeyType(key=col, type=infer_type(val[col].iloc[0]), description="")
                    for col in val.columns
                ]
                return PDDataFrameType(columns=columns)
            elif isinstance(val, pd.Series):
                if val.empty:
                    element = AnyType()
                else:
                    element = infer_type(val.iloc[0])
                return PDSeriesType(element_type=element)
            elif isinstance(val, np.ndarray):
                if val.size == 0:
                    element = AnyType()
                else:
                    element = infer_type(val.flat[0])
                return NumpyNdarrayType(element_type=element, length=val.size)
            return AnyType()

        return cls(
            the_type=infer_type(value), description="Automatically generated type"
        )

    def matches_value(self, value: Any) -> bool:
        """
        Determines whether the given value conforms to the current type.
        """
        return self.the_type.validate(value)


# -----------------------
# Update Forward References
# -----------------------

TypeRepresentation = Union[
    IntType,
    BoolType,
    StrType,
    AnyType,
    NoneType,
    FloatType,
    OptionalType,
    ListType,
    TypedDictType,
    TupleType,
    SetType,
    PDDataFrameType,
    PDSeriesType,
    NumpyNdarrayType,
]

# Update forward references in all classes
IntType.model_rebuild()
BoolType.model_rebuild()
StrType.model_rebuild()
AnyType.model_rebuild()
NoneType.model_rebuild()
FloatType.model_rebuild()
OptionalType.model_rebuild()
KeyType.model_rebuild()
ListType.model_rebuild()
TypedDictType.model_rebuild()
TupleType.model_rebuild()
SetType.model_rebuild()
PDDataFrameType.model_rebuild()
PDSeriesType.model_rebuild()
NumpyNdarrayType.model_rebuild()
ExtendedType.model_rebuild()

if __name__ == "__main__":
    from openai import OpenAI

    client = OpenAI()

    # Example usage
    example_value = {
        "name": "John",
        "age": 30,
        "tags": ["developer", "python"],
        "active": True,
        "projects": {
            "project1": {"name": "Project A", "completed": False},
            "project2": {"name": "Project B", "completed": True},
        },
    }

    type_representation = ExtendedType.from_value(example_value)
    print(type_representation.to_markdown())

    # print(json.dumps(TypeDeclarationModel.model_json_schema(), indent=2))

    prompts = [
        "Give me a type for a data frame with three float columns 'A', 'B', and 'Cow'.",
        # "Give me a dictionary with keys 'name' and 'age' and values of type str and int respectively.",
        # "Give me a list of dictionaries with keys 'name' and 'age' and values of type str and int respectively.",
        # "Give me a tuple of two integers.",
        # "Give me a set of strings.",
        # "Give me a numpy ndarray of floats.",
        "Give me a pandas series of integers.",
        "Give me a dictionary for the results of a scipy LinearRegression model.",
        """
        Give me a dataframe type for this data file:
            species,"Beak length, mm","Beak depth, mm"
            fortis,9.4,8.0
            fortis,9.2,8.3 
            scandens,13.9,8.4
        """,
    ]

    for p in prompts:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-11-20",
            response_format=ExtendedType,
            messages=[{"role": "system", "content": p}],
        )

        print(completion.choices[0].message.parsed.to_markdown())

# from typing import Any, Type, Callable, Union, Optional, Tuple
# import numpy as np
# import pandas as pd
# import re
# from pydantic import BaseModel, Field
# from flowco.builder.type_ops import str_to_type


# class ExtendedType(BaseModel):
#     type: str = Field(
#         description="""\
#         The type of the return value. Should be one of 'int', 'bool', 'str', 'None', 'float', 'Tuple', 'Dict', 'Set', 'pd.DataFrame', 'np.ndarray', 'List',
#         along with the necessary type parameters:
#         * For List include the extended type of the elements in the list.
#         * For Dict include the specific string keys with their extended types.
#         * For Tuple include the extended type of each element.
#         * For pd.DataFrame include the extended types of the columns as a type parameter to the DataFrame type that is a map from column names to types.
#         * For pd.Series include the extended type of the elements in the series.
#         * For np.array include the extended type of the elements in the array.
#         * For Set include the extended type of the elements in the set.
#         These are the only options.  User-defined types are not allowed.  All nested types must be one of these types as well.
#         Examples:
#             "int",
#             "List[int]",
#             "Dict['key1': int, 'key2': str]",
#             "Tuple[int, str]",
#             "pd.DataFrame['name': str, 'value': float, 'count': int]",
#             "pd.Series[int]",
#             "np.ndarray[int]",
#             "Set[str]",
#         """,
#     )

# #            "Callable[[int, str], bool]",
# #        * For Callable include the argument types and the return type.

#     def remove_nonstandard_params(self) -> "ExtendedType":
#         rep = self.type

#         prefixes = ["pd.DataFrame[", "np.ndarray[", "pd.Series[", "Dict[" ]
#         for prefix in prefixes:
#             while prefix in rep:
#                 rep = self.remove_type_parameters_helper(rep, prefix)
#         return ExtendedType(type=rep)

#     def remove_type_parameters_helper(self, rep: str, prefix: str) -> str:
#         start_index = rep.find(prefix)
#         if start_index == -1:
#             return rep
#         end_index = start_index + len(prefix)
#         bracket_count = 1
#         while bracket_count > 0 and end_index < len(rep):
#             if rep[end_index] == "[":
#                 bracket_count += 1
#             elif rep[end_index] == "]":
#                 bracket_count -= 1
#             end_index += 1
#         # Remove the parameters inside the brackets
#         rep = rep[:start_index] + prefix[:-1] + rep[end_index:]
#         return rep

#     def to_python_type(self) -> str:
#         try:
#             return self.remove_nonstandard_params().type
#         except Exception:
#             return "Any"

#     def is_not_NoneType(self) -> bool:
#         return self.type != None

#     def type_description(self) -> str:
#         """
#         Return a human-readable description of the type.

#         TODO: Make robust for []s inside other brackets...
#         """
#         type_str = self.type

#         # Handle Callable
#         if type_str.startswith("Callable["):
#             match = re.match(r"Callable\[\s*\[(.*?)\s*\],\s*(\w+)\s*\]", type_str)
#             if match:
#                 args_str, return_type = match.groups()
#                 args = [arg.strip() for arg in args_str.split(",")] if args_str else []
#                 args_formatted = ", ".join(args) if args else "No arguments"
#                 return f"a callable that takes arguments of types ({args_formatted}) and returns {return_type}."
#             else:
#                 return "a callable with unspecified signature."

#         # Handle pd.DataFrame
#         elif type_str.startswith("pd.DataFrame["):
#             columns_str = type_str[
#                 len("pd.DataFrame[") : -1
#             ]  # Remove prefix and trailing ']'
#             if not columns_str:
#                 return "a pandas DataFrame with unspecified columns."
#             columns = [col.strip() for col in columns_str.split(",")]
#             columns_descriptions = []
#             for col in columns:
#                 # Each column is in the format "'name': type"
#                 col_match = re.match(r"'(.*?)'\s*:\s*(\w+)", col)
#                 if col_match:
#                     col_name, col_type = col_match.groups()
#                     columns_descriptions.append(f"'{col_name}' of type {col_type}")
#                 else:
#                     columns_descriptions.append(col)  # Fallback
#             columns_formatted = ", ".join(columns_descriptions)
#             return (
#                 f"a pandas DataFrame with the following columns: {columns_formatted}."
#             )

#         # Handle pd.Series
#         elif type_str.startswith("pd.Series["):
#             element_type = type_str[len("pd.Series[") : -1]
#             return f"a pandas Series with elements of type {element_type}."

#         # Handle np.ndarray
#         elif type_str.startswith("np.ndarray["):
#             element_type = type_str[len("np.ndarray[") : -1]
#             return f"a numpy array with elements of type {element_type}."

#         # Handle List
#         elif type_str.startswith("List["):
#             element_type = type_str[len("List[") : -1]
#             return f"a list with elements of type {element_type}."

#         # Handle Tuple
#         elif type_str.startswith("Tuple["):
#             elements = type_str[len("Tuple[") : -1]
#             if not elements:
#                 return "an empty tuple."
#             elements_list = [elem.strip() for elem in elements.split(",")]
#             if len(elements_list) == 1:
#                 return f"a tuple with a single element of type {elements_list[0]}."
#             else:
#                 elements_formatted = ", ".join(elements_list)
#                 return f"a tuple with elements of types ({elements_formatted})."

#         # Handle Dict
#         elif type_str.startswith("Dict["):
#             # Check if it's a Dict with specific string keys
#             if type_str.startswith("Dict['"):
#                 columns_str = type_str[len("Dict[") : -1]  # Remove prefix and trailing ']'
#                 if not columns_str:
#                     return "a dictionary with no keys specified."
#                 columns = [col.strip() for col in columns_str.split(",")]
#                 columns_descriptions = []
#                 for col in columns:
#                     # Each column is in the format "'key': type"
#                     col_match = re.match(r"'(.*?)'\s*:\s*(\w+)", col)
#                     if col_match:
#                         key_name, key_type = col_match.groups()
#                         columns_descriptions.append(f"'{key_name}' of type {key_type}")
#                     else:
#                         columns_descriptions.append(col)  # Fallback
#                 columns_formatted = ", ".join(columns_descriptions)
#                 return f"a dictionary with the following keys and types: {columns_formatted}."
#             else:
#                 # Generic Dict[key, value]
#                 key_value = type_str[len("Dict[") : -1]
#                 key, value = [part.strip() for part in key_value.split(",", 1)]
#                 return f"a dictionary with keys of type {key} and values of type {value}."

#         # Handle Set
#         elif type_str.startswith("Set["):
#             element_type = type_str[len("Set[") : -1]
#             return f"a set with elements of type {element_type}."

#         # Handle Union
#         elif type_str.startswith("Union["):
#             union_types = type_str[len("Union[") : -1]
#             types_list = [t.strip() for t in union_types.split(",")]
#             types_formatted = ", ".join(types_list)
#             return f"one of the following types: {types_formatted}."

#         # Handle basic types
#         else:
#             basic_descriptions = {
#                 "int": "an integer.",
#                 "float": "a float.",
#                 "bool": "a boolean.",
#                 "str": "a string.",
#                 "None": "None.",
#                 "Any": "any type.",
#             }
#             return basic_descriptions.get(type_str, f"a {type_str}.")

#     @classmethod
#     def from_value(cls, o: Any) -> 'ExtendedType':
#         """
#         Analyze the object `o` and return an ExtendedType instance describing its type.

#         Supported types:
#         - Basic Types: int, float, bool, str, None
#         - Collections: list, tuple, dict, set
#         - NumPy Arrays: np.ndarray
#         - Pandas Objects: pd.DataFrame, pd.Series
#         - Callables: functions, methods
#         """

#         def get_type_str(obj: Any) -> str:
#             # **IMPORTANT**: Check for bool before int because bool is a subclass of int in Python
#             if isinstance(obj, bool) or isinstance(obj, np.bool_):
#                 return "bool"
#             elif isinstance(obj, (int, np.integer)):
#                 return "int"
#             elif isinstance(obj, (float, np.floating)):
#                 return "float"
#             elif isinstance(obj, (str, np.str_)):
#                 return "str"
#             elif obj is None:
#                 return "None"
#             elif isinstance(obj, list):
#                 if not obj:
#                     return "List[Any]"
#                 # Determine types of all elements
#                 element_types = set(get_type_str(item) for item in obj)
#                 if len(element_types) == 1:
#                     element_type = element_types.pop()
#                 else:
#                     element_type = "Union[" + ", ".join(sorted(element_types)) + "]"
#                 return f"List[{element_type}]"
#             elif isinstance(obj, tuple):
#                 if not obj:
#                     return "Tuple[]"
#                 # Determine types of each element
#                 element_types = [get_type_str(item) for item in obj]
#                 return f"Tuple[{', '.join(element_types)}]"
#             elif isinstance(obj, dict):
#                 if not obj:
#                     return "Dict[]"
#                 if all(isinstance(k, str) for k in obj.keys()):
#                     # Use specific key type representations
#                     key_type_map = {}
#                     for k, v in obj.items():
#                         key_type_map[k] = get_type_str(v)
#                     # To make it deterministic, sort keys
#                     sorted_keys = sorted(key_type_map.keys())
#                     items = [f"'{k}': {key_type_map[k]}" for k in sorted_keys]
#                     return f"Dict[{', '.join(items)}]"
#                 else:
#                     # General dict representation
#                     key_types = set(get_type_str(k) for k in obj.keys())
#                     value_types = set(get_type_str(v) for v in obj.values())
#                     if len(key_types) == 1:
#                         key_type = key_types.pop()
#                     else:
#                         key_type = "Union[" + ", ".join(sorted(key_types)) + "]"
#                     if len(value_types) == 1:
#                         value_type = value_types.pop()
#                     else:
#                         value_type = "Union[" + ", ".join(sorted(value_types)) + "]"
#                     return f"Dict[{key_type}, {value_type}]"
#             elif isinstance(obj, set):
#                 if not obj:
#                     return "Set[Any]"
#                 # Determine types of all elements
#                 element_types = set(get_type_str(item) for item in obj)
#                 if len(element_types) == 1:
#                     element_type = element_types.pop()
#                 else:
#                     element_type = "Union[" + ", ".join(sorted(element_types)) + "]"
#                 return f"Set[{element_type}]"
#             elif isinstance(obj, np.ndarray):
#                 if obj.size == 0:
#                     return "np.ndarray[Any]"
#                 # Assume homogeneous types; get type of first element
#                 first_elem = obj.flat[0]
#                 element_type = get_type_str(first_elem)
#                 return f"np.ndarray[{element_type}]"
#             elif isinstance(obj, pd.Series):
#                 if obj.empty:
#                     return "pd.Series[Any]"
#                 # Assume homogeneous types; get type of first element
#                 first_elem = obj.iloc[0]
#                 element_type = get_type_str(first_elem)
#                 return f"pd.Series[{element_type}]"
#             elif isinstance(obj, pd.DataFrame):
#                 if obj.empty:
#                     return "pd.DataFrame[]"
#                 column_type_map = {}
#                 for col in obj.columns:
#                     dtype = obj[col].dtype
#                     # Map pandas/numpy dtype to Python type
#                     if pd.api.types.is_integer_dtype(dtype):
#                         column_type_map[col] = "int"
#                     elif pd.api.types.is_float_dtype(dtype):
#                         column_type_map[col] = "float"
#                     elif pd.api.types.is_bool_dtype(dtype):
#                         column_type_map[col] = "bool"
#                     elif pd.api.types.is_datetime64_any_dtype(dtype):
#                         column_type_map[col] = "datetime"
#                     else:
#                         column_type_map[col] = "str"
#                 # Create type string for DataFrame
#                 columns_types = [f"'{col}': {typ}" for col, typ in column_type_map.items()]
#                 return f"pd.DataFrame[{', '.join(columns_types)}]"
#             elif isinstance(obj, Callable):
#                 # Extract signature if possible
#                 import inspect

#                 try:
#                     sig = inspect.signature(obj)
#                     arg_types = []
#                     for param in sig.parameters.values():
#                         if param.annotation != inspect.Parameter.empty:
#                             # Handle complex annotations (like typing.Optional[int], etc.)
#                             if hasattr(param.annotation, '__name__'):
#                                 arg_types.append(param.annotation.__name__)
#                             else:
#                                 arg_types.append("Any")
#                         else:
#                             arg_types.append("Any")
#                     if sig.return_annotation != inspect.Signature.empty:
#                         if hasattr(sig.return_annotation, '__name__'):
#                             return_type = sig.return_annotation.__name__
#                         else:
#                             return_type = "Any"
#                     else:
#                         return_type = "Any"
#                     return f'Callable[[{", ".join(arg_types)}], {return_type}]'
#                 except Exception:
#                     return "Callable[..., Any]"
#             else:
#                 raise TypeError(f"Type {type(obj)} not supported.")

#         type_str = get_type_str(o)
#         return ExtendedType(type=type_str)


# def run_tests():
#     # Example Usage:

#     # Basic types
#     assert from_value(42).type == "int", "Failed on int"
#     assert from_value(3.14).type == "float", "Failed on float"
#     assert from_value("Hello").type == "str", "Failed on str"
#     assert from_value(True).type == "bool", "Failed on bool"
#     assert from_value(False).type == "bool", "Failed on bool"
#     assert from_value(None).type == "None", "Failed on None"

#     # Collections
#     assert from_value([1, 2, 3]).type == "List[int]", "Failed on List[int]"
#     assert (
#         from_value([1, "a", True]).type == "List[Union[bool, int, str]]"
#     ), "Failed on List[Union[bool, int, str]]"
#     assert (
#         from_value((1, "a")).type == "Tuple[int, str]"
#     ), "Failed on Tuple[int, str]"
#     assert from_value((1,)).type == "Tuple[int]", "Failed on Tuple[int]"
#     # Modified dict tests with specific string keys
#     assert from_value({"a": 1, "b": 2}).type == "Dict['a': int, 'b': int]", "Failed on Dict with string keys"
#     assert from_value({"a": 1, "b": "two"}).type == "Dict['a': int, 'b': str]", "Failed on Dict with string keys with different value types"
#     assert from_value({"a": 1, "b": {"c": 3}}).type == "Dict['a': int, 'b': Dict['c': int]]", "Failed on nested Dict with string keys"
#     assert from_value({1, 2, 3}).type == "Set[int]", "Failed on Set[int]"
#     assert from_value(set()).type == "Set[Any]", "Failed on Set[Any]"

#     # NumPy array
#     arr = np.array([1, 2, 3], dtype=np.int32)
#     assert from_value(arr).type == "np.ndarray[int]", "Failed on np.ndarray[int]"

#     empty_arr = np.array([], dtype=np.float64)
#     assert (
#         from_value(empty_arr).type == "np.ndarray[Any]"
#     ), "Failed on empty np.ndarray"

#     # Pandas Series
#     series = pd.Series(
#         [10, 20, 30], index=["x", "y", "z"], name="numbers", dtype="int64"
#     )
#     assert from_value(series).type == "pd.Series[int]", "Failed on pd.Series[int]"

#     empty_series = pd.Series([], dtype="float64")
#     assert (
#         from_value(empty_series).type == "pd.Series[Any]"
#     ), "Failed on empty pd.Series"

#     # Pandas DataFrame
#     df = pd.DataFrame(
#         {"A": [1, 2, 3], "B": ["x", "y", "z"]}, index=["row1", "row2", "row3"]
#     )
#     assert (
#         from_value(df).type == "pd.DataFrame['A': int, 'B': str]"
#     ), "Failed on pd.DataFrame['A': int, 'B': str]"

#     empty_df = pd.DataFrame(columns=["A", "B"])
#     assert (
#         from_value(empty_df).type == "pd.DataFrame[]"
#     ), "Failed on empty pd.DataFrame"

#     # Callable (Function)
#     def sample_function(a: int, b: str) -> bool:
#         return str(a) == b

#     assert (
#         from_value(sample_function).type == "Callable[[int, str], bool]"
#     ), "Failed on Callable[[int, str], bool]"

#     # Callable without annotations
#     def unannotated_function(a, b):
#         return a + b

#     assert (
#         from_value(unannotated_function).type == "Callable[[Any, Any], Any]"
#     ), "Failed on Callable[[Any, Any], Any]"

#     # Nested Structures
#     nested_obj = {
#         "numbers": [1, 2, 3],
#         "matrix": np.array([[1, 2], [3, 4]], dtype=np.int32),
#         "series": pd.Series([10, 20], index=["a", "b"], name="scores", dtype="int64"),
#         "details": {"name": "Test", "active": True, "tags": ("example", "test")},
#         "df": pd.DataFrame({"X": [5, 6], "Y": ["foo", "bar"]}, index=["row1", "row2"]),
#         "func": sample_function,
#     }

#     expected_nested_obj_type = (
#         "Dict['details': Dict['active': bool, 'name': str, 'tags': Tuple[str, str]], 'df': pd.DataFrame['X': int, 'Y': str], 'func': Callable[[int, str], bool], "
#         "'matrix': np.ndarray[int], 'numbers': List[int], 'series': pd.Series[int]]"
#     )
#     # Ensure type_from_value sorts the keys alphabetically
#     # So, 'df', 'func', 'details', 'matrix', 'numbers', 'series'
#     assert (
#         from_value(nested_obj).type == expected_nested_obj_type
#     ), f"type_from_value(nested_obj).type={from_value(nested_obj).type}, expected={expected_nested_obj_type}"

#     # Another Nested Example
#     nested_list = [
#         pd.DataFrame(
#             {"name": ["Alice", "Bob"], "value": [10.5, 20.3], "count": [1, 2]}
#         ),
#         pd.DataFrame(
#             {"name": ["Charlie", "David"], "value": [30.7, 40.2], "count": [3, 4]}
#         ),
#     ]

#     expected_nested_list_type = (
#         "List[pd.DataFrame['name': str, 'value': float, 'count': int]]"
#     )
#     assert (
#         from_value(nested_list).type == expected_nested_list_type
#     ), f"type_from_value(nested_list).type={from_value(nested_list).type}, expected={expected_nested_list_type}"

#     # Tests for extended_repr
#     # Since extended_repr now returns a tuple (representation_str, clipped_flag),
#     # we need to unpack the results and adjust the assertions accordingly.

#     # Basic types without clipping
#     repr_str, clipped = extended_repr(42)
#     assert repr_str == "42", "Failed on extended_repr(int)"
#     assert not clipped, "Clipping occurred on extended_repr(int)"

#     repr_str, clipped = extended_repr(3.14)
#     assert repr_str == "3.14", "Failed on extended_repr(float)"
#     assert not clipped, "Clipping occurred on extended_repr(float)"

#     repr_str, clipped = extended_repr("Hello")
#     assert repr_str == "'Hello'", "Failed on extended_repr(str)"
#     assert not clipped, "Clipping occurred on extended_repr(str)"

#     repr_str, clipped = extended_repr(True)
#     assert repr_str == "True", "Failed on extended_repr(bool)"
#     assert not clipped, "Clipping occurred on extended_repr(bool)"

#     repr_str, clipped = extended_repr(False)
#     assert repr_str == "False", "Failed on extended_repr(bool)"
#     assert not clipped, "Clipping occurred on extended_repr(bool)"

#     repr_str, clipped = extended_repr(None)
#     assert repr_str == "None", "Failed on extended_repr(None)"
#     assert not clipped, "Clipping occurred on extended_repr(None)"

#     # Collections without clipping
#     repr_str, clipped = extended_repr([1, 2, 3])
#     assert repr_str == "[1, 2, 3]", "Failed on extended_repr(list)"
#     assert not clipped, "Clipping occurred on extended_repr(list)"

#     repr_str, clipped = extended_repr([1, "a", True])
#     assert repr_str == "[1, 'a', True]", "Failed on extended_repr(heterogeneous list)"
#     assert not clipped, "Clipping occurred on extended_repr(heterogeneous list)"

#     repr_str, clipped = extended_repr((1, 2, 3))
#     assert repr_str == "(1, 2, 3)", "Failed on extended_repr(tuple)"
#     assert not clipped, "Clipping occurred on extended_repr(tuple)"

#     repr_str, clipped = extended_repr((1,))
#     assert repr_str == "(1,)", "Failed on extended_repr(single-element tuple)" + repr_str
#     assert not clipped, "Clipping occurred on extended_repr(single-element tuple)"

#     # Modified dict tests with specific string keys
#     repr_str, clipped = extended_repr({"a": 1, "b": 2})
#     expected_repr = "{'a': 1, 'b': 2}"
#     assert repr_str == expected_repr, "Failed on extended_repr(dict)"
#     assert not clipped, "Clipping occurred on extended_repr(dict)"

#     repr_str, clipped = extended_repr({"a": 1, "b": "two"})
#     expected_repr = "{'a': 1, 'b': 'two'}"
#     assert repr_str == expected_repr, "Failed on extended_repr(dict with Union)"
#     assert not clipped, "Clipping occurred on extended_repr(dict with Union)"

#     repr_str, clipped = extended_repr({"a": 1, "b": {"c": 3}})
#     expected_repr = "{'a': 1, 'b': {'c': 3}}"
#     assert repr_str == expected_repr, "Failed on extended_repr(nested dict)"
#     assert not clipped, "Clipping occurred on extended_repr(nested dict)"

#     repr_str, clipped = extended_repr({1, 2, 3})
#     # Since sets are sorted based on their repr, the order should be consistent
#     # Depending on the sorting key, it could be [1, 2, 3]
#     assert repr_str == "set([1, 2, 3])", "Failed on extended_repr(set)"
#     assert not clipped, "Clipping occurred on extended_repr(set)"

#     repr_str, clipped = extended_repr(set())
#     assert repr_str == "set([])", "Failed on extended_repr(empty set)"
#     assert not clipped, "Clipping occurred on extended_repr(empty set)"

#     # NumPy array without clipping
#     arr = np.array([1, 2, 3], dtype=np.int32)
#     arr_repr = "np.array([1, 2, 3], dtype='int32')"
#     repr_str, clipped = extended_repr(arr)
#     assert repr_str == arr_repr, "Failed on extended_repr(np.ndarray)"
#     assert not clipped, "Clipping occurred on extended_repr(np.ndarray)"

#     empty_arr = np.array([], dtype=np.float64)
#     empty_arr_repr = "np.array([], dtype='float64')"
#     repr_str, clipped = extended_repr(empty_arr)
#     assert repr_str == empty_arr_repr, "Failed on extended_repr(empty np.ndarray)"
#     assert not clipped, "Clipping occurred on extended_repr(empty np.ndarray)"

#     # Pandas Series without clipping
#     series = pd.Series(
#         [10, 20, 30], index=["x", "y", "z"], name="numbers", dtype="int64"
#     )
#     series_repr = (
#         "pd.Series([10, 20, 30], index=['x', 'y', 'z'], name='numbers', dtype='int64')"
#     )
#     repr_str, clipped = extended_repr(series)
#     assert repr_str == series_repr, "Failed on extended_repr(pd.Series)"
#     assert not clipped, "Clipping occurred on extended_repr(pd.Series)"

#     empty_series = pd.Series([], dtype="float64")
#     empty_series_repr = "pd.Series([], index=[], name=None, dtype='float64')"
#     repr_str, clipped = extended_repr(empty_series)
#     assert repr_str == empty_series_repr, "Failed on extended_repr(empty pd.Series)"
#     assert not clipped, "Clipping occurred on extended_repr(empty pd.Series)"

#     # Pandas DataFrame without clipping
#     df = pd.DataFrame(
#         {"A": [1, 2, 3], "B": ["x", "y", "z"]}, index=["row1", "row2", "row3"]
#     )
#     df_repr = "pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']}, index=['row1', 'row2', 'row3'], columns=['A', 'B'])"
#     repr_str, clipped = extended_repr(df)
#     assert repr_str == df_repr, "Failed on extended_repr(pd.DataFrame)"
#     assert not clipped, "Clipping occurred on extended_repr(pd.DataFrame)"

#     empty_df = pd.DataFrame(columns=["A", "B"])
#     empty_df_repr = "pd.DataFrame({'A': [], 'B': []}, index=[], columns=['A', 'B'])"
#     repr_str, clipped = extended_repr(empty_df)
#     assert repr_str == empty_df_repr, "Failed on extended_repr(empty pd.DataFrame)"
#     assert not clipped, "Clipping occurred on extended_repr(empty pd.DataFrame)"

#     # Nested Structures without clipping
#     nested_obj = {
#         "numbers": [1, 2, 3],
#         "matrix": np.array([[1, 2], [3, 4]], dtype=np.int32),
#         "series": pd.Series([10, 20], index=["a", "b"], name="scores", dtype="int64"),
#         "details": {"name": "Test", "active": True, "tags": ("example", "test")},
#         "df": pd.DataFrame({"X": [5, 6], "Y": ["foo", "bar"]}, index=["row1", "row2"]),
#         "func": sample_function,
#     }

#     expected_nested_obj_type = (
#         "Dict['details': Dict['active': bool, 'name': str, 'tags': Tuple[str, str]], 'df': pd.DataFrame['X': int, 'Y': str], 'func': Callable[[int, str], bool], "
#         "'matrix': np.ndarray[int], 'numbers': List[int], 'series': pd.Series[int]]"
#     )
#     # Ensure type_from_value sorts the keys alphabetically
#     # So, 'df', 'func', 'details', 'matrix', 'numbers', 'series'
#     assert (
#         from_value(nested_obj).type == expected_nested_obj_type
#     ), f"type_from_value(nested_obj).type={from_value(nested_obj).type}, expected={expected_nested_obj_type}"

#     # Another Nested Example
#     nested_list = [
#         pd.DataFrame(
#             {"name": ["Alice", "Bob"], "value": [10.5, 20.3], "count": [1, 2]}
#         ),
#         pd.DataFrame(
#             {"name": ["Charlie", "David"], "value": [30.7, 40.2], "count": [3, 4]}
#         ),
#     ]

#     expected_nested_list_type = (
#         "List[pd.DataFrame['name': str, 'value': float, 'count': int]]"
#     )
#     assert (
#         from_value(nested_list).type == expected_nested_list_type
#     ), f"type_from_value(nested_list).type={from_value(nested_list).type}, expected={expected_nested_list_type}"

#     print("All assertions passed successfully!")


# if __name__ == "__main__":
#     run_tests()

#     # Additional Example Usage:

#     # Creating ExtendedType instances and testing remove_nonstandard_params()
#     e1 = ExtendedType(type="pd.DataFrame['name': str, 'value': float, 'count': int]")
#     # Assuming remove_nonstandard_params() processes the type string in some way
#     # Replace print statements with assertions as needed
#     # For demonstration, we'll assert the method returns the expected string
#     assert e1.remove_nonstandard_params().type == "pd.DataFrame"

#     e2 = ExtendedType(type="pd.DataFrame['mean_beak_length': float]")
#     assert e2.remove_nonstandard_params().type == "pd.DataFrame"

#     e3 = ExtendedType(type="Callable[[int, str], bool]")
#     assert e3.remove_nonstandard_params().type == "Callable[[int, str], bool]"

#     print("ExtendedType remove_nonstandard_params assertions passed successfully!")

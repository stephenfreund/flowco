from __future__ import annotations
import importlib
import inspect
import pprint
import textwrap
from typing import Any, Iterable, Set, Tuple, Union, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from abc import abstractmethod
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from flowco.builder import type_ops
from flowco.util.output import error

# Define TypeRepresentation before using it in classes
TypeRepresentation = Union[
    "IntType",
    "BoolType",
    "StrType",
    "NoneType",
    "FloatType",
    "OptionalType",
    "ListType",
    "RecordType",
    "DictType",
    # "TupleType",
    "SklearnClassType",
    "SetType",
    "PDDataFrameType",
    "PDSeriesType",
    "NumpyNdarrayType",
]


# Base class for all type representations with required methods
class BaseType(BaseModel):
    def to_python_type(self) -> str:
        raise NotImplementedError("to_python_type method not implemented.")

    def to_markdown(self, indent: int = 0) -> List[str]:
        raise NotImplementedError("to_markdown method not implemented.")

    def __str__(self) -> str:
        raise NotImplementedError("__str__ method not implemented.")

    @abstractmethod
    def check_value(self, value: Any) -> None:
        """Validate whether the given value conforms to the type.
        Raises:
            ValueError: If the value does not conform to the type.
        """
        pass

    @abstractmethod
    def type_schema(self) -> Dict[str, Any]:
        """Generate a JSON schema representation of the type."""
        pass


# Implementing each type class without default descriptions and with __init__ methods


class IntType(BaseType):
    type: Literal["int"]
    description: str = Field(
        ..., description="A human-readable description of the type."  # Required field
    )

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "int"
        super().__init__(**data)

    def to_python_type(self) -> str:
        return "int"

    def to_markdown(self, indent: int = 0) -> List[str]:
        spaces = "  " * indent
        return [f"{spaces}- **Integer**: {self.description}"]

    def __str__(self) -> str:
        return "int"

    def check_value(self, value: Any) -> None:
        if (isinstance(value, int) or isinstance(value, np.integer)) and not isinstance(
            value, bool
        ):
            return
        raise ValueError(f"Expected int, got {type(value).__name__}")

    def type_schema(self) -> Dict[str, Any]:
        return {"type": "int", "description": self.description}


class BoolType(BaseType):
    type: Literal["bool"]
    description: str = Field(
        ..., description="A human-readable description of the type."  # Required field
    )

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "bool"
        super().__init__(**data)

    def to_python_type(self) -> str:
        return "bool"

    def to_markdown(self, indent: int = 0) -> List[str]:
        spaces = "  " * indent
        return [f"{spaces}- **Boolean**: {self.description}"]

    def __str__(self) -> str:
        return "bool"

    def check_value(self, value: Any) -> None:
        if isinstance(value, bool):
            return
        raise ValueError(f"Expected bool, got {type(value).__name__}")

    def type_schema(self) -> Dict[str, Any]:
        return {"type": "bool", "description": self.description}


class StrType(BaseType):
    type: Literal["str"]
    description: str = Field(
        ..., description="A human-readable description of the type."  # Required field
    )

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "str"
        super().__init__(**data)

    def to_python_type(self) -> str:
        return "str"

    def to_markdown(self, indent: int = 0) -> List[str]:
        spaces = "  " * indent
        return [f"{spaces}- **String**: {self.description}"]

    def __str__(self) -> str:
        return "str"

    def check_value(self, value: Any) -> None:
        if isinstance(value, str):
            return
        raise ValueError(f"Expected str, got {type(value).__name__}")

    def type_schema(self) -> Dict[str, Any]:
        return {"type": "str", "description": self.description}


class AnyType(BaseType):
    type: Literal["Any"]
    description: str = Field(
        ..., description="A human-readable description of the type."  # Required field
    )

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "Any"
        super().__init__(**data)

    def to_python_type(self) -> str:
        return "Any"

    def to_markdown(self, indent: int = 0) -> List[str]:
        spaces = "  " * indent
        return [f"{spaces}- **Any**: {self.description}"]

    def __str__(self) -> str:
        return "Any"

    def check_value(self, value: Any) -> None:
        return  # Always matches

    def type_schema(self) -> Dict[str, Any]:
        return {
            "type": "Any",
            "description": self.description,
            # No type constraint
        }


class NoneType(BaseType):
    type: Literal["None"]
    description: str = Field(
        ..., description="A human-readable description of the type."  # Required field
    )

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "None"
        super().__init__(**data)

    def to_python_type(self) -> str:
        return "None"

    def to_markdown(self, indent: int = 0) -> List[str]:
        spaces = "  " * indent
        return [f"{spaces}- **None**: {self.description}"]

    def __str__(self) -> str:
        return "None"

    def check_value(self, value: Any) -> None:
        if value is None:
            return
        raise ValueError(f"Expected None, got {type(value).__name__}")

    def type_schema(self) -> Dict[str, Any]:
        return {"type": "None", "description": self.description}


class FloatType(BaseType):
    type: Literal["float"]
    description: str = Field(
        ..., description="A human-readable description of the type."  # Required field
    )

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "float"
        super().__init__(**data)

    def to_python_type(self) -> str:
        return "float"

    def to_markdown(self, indent: int = 0) -> List[str]:
        spaces = "  " * indent
        return [f"{spaces}- **Float**: {self.description}"]

    def __str__(self) -> str:
        return "float"

    def check_value(self, value: Any) -> None:
        if (
            isinstance(value, float)
            or isinstance(value, np.floating)
            or isinstance(value, int)
        ):
            return
        raise ValueError(f"Expected float, got {type(value).__name__}")

    def type_schema(self) -> Dict[str, Any]:
        return {"type": "float", "description": self.description}


class OptionalType(BaseType):
    type: Literal["Optional"]
    wrapped_type: TypeRepresentation
    description: str = Field(
        ..., description="A human-readable description of the type."  # Required field
    )

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
            return [f"{spaces}- **Optional**: {self.description}"]
        return [f"{spaces}- **Optional**: {self.description}"] + wrapped_markdown

    def __str__(self) -> str:
        return f"Optional[{self.wrapped_type}]"

    def check_value(self, value: Any) -> None:
        if value is None:
            return
        self.wrapped_type.check_value(value)

    def type_schema(self) -> Dict[str, Any]:
        return {
            "type": "optional",
            "wrapped": self.wrapped_type.type_schema(),
            "description": self.description,
        }


class KeyType(BaseModel):
    key: str
    type: TypeRepresentation
    description: str = Field(
        ..., description="A description of what this key represents."  # Required field
    )

    def to_python_type(self) -> str:
        return self.type.to_python_type()

    def __init__(self, **data):
        super().__init__(**data)

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

    def check_value(self, value: Any) -> None:
        self.type.check_value(value)

    def type_schema(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "type": self.type.type_schema(),
            "description": self.description,
        }


class ListType(BaseType):
    type: Literal["List"]
    element_type: TypeRepresentation
    length: Optional[int] = Field(
        None,  # No default value; can be omitted to indicate arbitrary length
        description="The expected length of the list. If None, the length can be arbitrary.",
    )
    description: str = Field(
        ..., description="A human-readable description of the type."  # Required field
    )

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "List"
        super().__init__(**data)

    def to_python_type(self) -> str:
        return f"List[{self.element_type.to_python_type()}]"

    def to_markdown(self, indent: int = 0) -> List[str]:
        spaces = "  " * indent
        if self.length is not None:
            desc = f"{self.description} Expected length: {self.length}."
        else:
            desc = self.description
        element_markdown = self.element_type.to_markdown(indent + 1)
        if len(element_markdown) == 1:
            return [f"{spaces}- **List**: {desc}"]
        return [f"{spaces}- **List**: {desc}"] + element_markdown

    def __str__(self) -> str:
        return f"List[{self.element_type}]"

    def check_value(self, value: Any) -> None:
        if not isinstance(value, list):
            raise ValueError(f"Expected list, got {type(value).__name__}")
        if self.length is not None and len(value) != self.length:
            raise ValueError(
                f"Expected list of length {self.length}, got length {len(value)}"
            )
        for index, elem in enumerate(value):
            try:
                self.element_type.check_value(elem)
            except ValueError as ve:
                raise ValueError(f"List element at index {index}: {ve}") from ve

    def type_schema(self) -> Dict[str, Any]:
        schema = {
            "type": "list",
            "items": self.element_type.type_schema(),
            "description": self.description,
        }
        if self.length is not None:
            schema["length"] = self.length
        return schema


class RecordType(BaseType):
    """
    An object record with a fixed set of fields and types, encoded as a TypedDict.
    Use this when the set of field names is known in advance and fixed.
    Do not use when the set of field names is dynamic or not known in advance.
    """

    type: Literal["record"]
    name: str = Field(
        ...,  # Required field
        description="A unique name for the record type. This is used to generate a unique record name.",
    )
    items: List[KeyType] = Field(
        ...,  # Required field
        description="A list of key-value pairs where the key is the key name and the value is the type of the key.",
    )
    description: str = Field(
        ..., description="A human-readable description of the type."  # Required field
    )

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "record"
        super().__init__(**data)

    def to_python_type(self) -> str:
        elems = [f"'{item.key}': {item.type.to_python_type()}" for item in self.items]
        map_str = ", ".join(elems)
        return f"TypedDict('{self.name}', {{{map_str}}})"

    def to_markdown(self, indent: int = 0) -> List[str]:
        if not self.items:
            return ["- **Dictionary** (empty)"]
        spaces = "  " * indent
        desc = [f"{spaces}- **Dictionary** with keys: {self.description}"]
        for item in self.items:
            desc.extend(item.to_markdown(indent + 1))
        return desc

    def __str__(self) -> str:
        elems = [f"'{item.key}': {item.type.to_python_type()}" for item in self.items]
        map_str = ", ".join(elems)
        return f"TypedDict('{self.name}', {{{map_str}}})"

    def check_value(self, value: Any) -> None:
        if not isinstance(value, dict):
            raise ValueError(f"Expected dict, got {type(value).__name__}")
        for item in self.items:
            if item.key not in value:
                raise ValueError(
                    f"Missing key '{item.key}' in dictionary with keys {value.keys()}"
                )
            try:
                item.type.check_value(value[item.key])
            except ValueError as ve:
                raise ValueError(f"Key '{item.key}': {ve}") from ve

    def type_schema(self) -> Dict[str, Any]:
        properties = {item.key: item.type.type_schema() for item in self.items}
        return {
            "type": "record",
            "properties": properties,
            "description": self.description,
        }


class DictType(BaseType):
    """
    A general-purpose dictionary with string keys and values of some pre-determined type.
    Use this when the set of strings used as keys is dynamic and not known in advance.
    """

    type: Literal["Dict"]
    # key_type: TypeRepresentation = Field(
    #     ..., description="The type of the dictionary keys."  # Required field
    # )
    value_type: TypeRepresentation = Field(
        ..., description="The type of the dictionary values."  # Required field
    )
    description: str = Field(
        ..., description="A human-readable description of the type."  # Required field
    )

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "Dict"
        super().__init__(**data)

    def to_python_type(self) -> str:
        # return f"Dict[{self.key_type.to_python_type()}, {self.value_type.to_python_type()}]"
        return f"Dict[str, {self.value_type.to_python_type()}]"

    def to_markdown(self, indent: int = 0) -> List[str]:
        spaces = "  " * indent
        desc = f"{self.description}"
        return [f"{spaces}- **Dict**: {desc}"]

    def __str__(self) -> str:
        # return f"Dict[{self.key_type}, {self.value_type}]"
        return f"Dict[str, {self.value_type}]"

    def check_value(self, value: Any) -> None:
        if not isinstance(value, dict):
            raise ValueError(f"Expected dict, got {type(value).__name__}")
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError(f"Dictionary key '{k}' has type {type(k)}, not string")
            try:
                self.value_type.check_value(v)
            except ValueError as ve:
                raise ValueError(f"Dictionary value for key '{k}': {ve}") from ve

    def type_schema(self) -> Dict[str, Any]:
        schema = {
            "type": "dict",
            "values": self.value_type.type_schema(),
            "description": self.description,
        }
        return schema


# class TupleType(BaseType):
#     type: Literal["Tuple"]
#     elements: List[TypeRepresentation]
#     description: str = Field(
#         ..., description="A human-readable description of the type."  # Required field
#     )

#     def __init__(self, **data):
#         if "type" not in data:
#             data["type"] = "Tuple"
#         super().__init__(**data)

#     def to_python_type(self) -> str:
#         elements_str = ", ".join([elem.to_python_type() for elem in self.elements])
#         return f"Tuple[{elements_str}]"

#     def to_markdown(self, indent: int = 0) -> List[str]:
#         spaces = "  " * indent
#         desc = f"{self.description}"
#         desc += f" Elements count: {len(self.elements)}."
#         desc_list = [f"{spaces}- **Tuple**: {desc}"]
#         for elem in self.elements:
#             desc_list.extend(elem.to_markdown(indent + 1))
#         return desc_list

#     def __str__(self) -> str:
#         elements_str = ", ".join([str(elem) for elem in self.elements])
#         return f"Tuple[{elements_str}]"

#     def check_value(self, value: Any) -> None:
#         if not isinstance(value, tuple):
#             raise ValueError(f"Expected tuple, got {type(value).__name__}")
#         if len(value) != len(self.elements):
#             raise ValueError(
#                 f"Expected tuple of length {len(self.elements)}, got length {len(value)}"
#             )
#         for index, (elem_type, elem_value) in enumerate(zip(self.elements, value)):
#             try:
#                 elem_type.check_value(elem_value)
#             except ValueError as ve:
#                 raise ValueError(f"Tuple element at index {index}: {ve}") from ve

#     def type_schema(self) -> Dict[str, Any]:
#         return {
#             "type": "array",
#             "items": [elem.type_schema() for elem in self.elements],
#             "minItems": len(self.elements),
#             "maxItems": len(self.elements),
#             "description": self.description,
#         }


class SetType(BaseType):
    type: Literal["Set"]
    element_type: TypeRepresentation
    description: str = Field(
        ..., description="A human-readable description of the type."  # Required field
    )

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "Set"
        super().__init__(**data)

    def to_python_type(self) -> str:
        return f"Set[{self.element_type.to_python_type()}]"

    def to_markdown(self, indent: int = 0) -> List[str]:
        spaces = "  " * indent
        desc = f"{self.description}"
        element_markdown = self.element_type.to_markdown(indent + 1)
        if len(element_markdown) == 1:
            return [f"{spaces}- **Set**: {desc}"]
        return [f"{spaces}- **Set**: {desc}"] + element_markdown

    def __str__(self) -> str:
        return f"Set[{self.element_type}]"

    def check_value(self, value: Any) -> None:
        if not isinstance(value, set):
            raise ValueError(f"Expected set, got {type(value).__name__}")
        for elem in value:
            try:
                self.element_type.check_value(elem)
            except ValueError as ve:
                raise ValueError(f"Set element '{elem}': {ve}") from ve

    def type_schema(self) -> Dict[str, Any]:
        return {
            "type": "set",
            "items": self.element_type.type_schema(),
            "description": self.description,
        }


class PDDataFrameType(BaseType):
    """
    Use this to define a Pandas DataFrame with specific columns and types.
    """

    type: Literal["pd.DataFrame"]
    columns: List[KeyType] = Field(
        ...,  # Required field
        description="A list of key-value pairs where the key is the column name and the value is the type of the elements in each column.",
    )
    description: str = Field(
        ..., description="A human-readable description of the type."  # Required field
    )

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "pd.DataFrame"
        super().__init__(**data)

    def to_python_type(self) -> str:
        return "pd.DataFrame"

    def to_markdown(self, indent: int = 0) -> List[str]:
        if not self.columns:
            return [f"- **Pandas DataFrame** (empty): {self.description}"]
        spaces = "  " * indent
        desc = [f"{spaces}- **Pandas DataFrame**: {self.description}"]
        for col in self.columns:
            desc.extend(col.to_markdown(indent + 1))
        return desc

    def __str__(self) -> str:
        column_types = ", ".join([f"{col}" for col in self.columns])
        return f"pd.DataFrame[{column_types}]"

    def check_value(self, value: Any) -> None:
        if not isinstance(value, pd.DataFrame):
            raise ValueError(f"Expected pd.DataFrame, got {type(value).__name__}")
        df_columns = value.columns.to_list()
        # print(value)
        if not value.empty:
            for col in self.columns:
                if col.key not in df_columns:
                    raise ValueError(
                        f"Missing column '{col.key}' in DataFrame with columns {value.columns}"
                    )
                for index, item in enumerate(value[col.key]):
                    try:
                        col.type.check_value(item)
                    except ValueError as ve:
                        raise ValueError(
                            f"DataFrame column '{col.key}', row {index} has value {item}: {ve}"
                        ) from ve

    def type_schema(self) -> Dict[str, Any]:
        properties = {col.key: col.type.type_schema() for col in self.columns}
        required = [col.key for col in self.columns]
        return {
            "type": "dataframe",
            "properties": properties,
            "description": self.description,
        }


class PDSeriesType(BaseType):
    type: Literal["pd.Series"]
    element_type: TypeRepresentation
    description: str = Field(
        ..., description="A human-readable description of the type."  # Required field
    )

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "pd.Series"
        super().__init__(**data)

    def to_python_type(self) -> str:
        return "pd.Series"

    def to_markdown(self, indent: int = 0) -> List[str]:
        spaces = "  " * indent
        return [f"{spaces}- **Pandas Series**: {self.description}"]

    def __str__(self) -> str:
        return f"pd.Series[{self.element_type}]"

    def check_value(self, value: Any) -> None:
        if not isinstance(value, pd.Series):
            raise ValueError(f"Expected pd.Series, got {type(value).__name__}")
        for index, item in value.items():
            try:
                self.element_type.check_value(item)
            except ValueError as ve:
                raise ValueError(f"Pandas Series at index {index}: {ve}") from ve

    def type_schema(self) -> Dict[str, Any]:
        return {
            "type": "series",
            "items": self.element_type.type_schema(),
            "description": self.description,
        }


class NumpyNdarrayType(BaseType):
    type: Literal["np.ndarray"]
    element_type: TypeRepresentation
    length: Optional[int] = Field(
        None,  # No default value; can be omitted to indicate arbitrary length
        description="The expected length of the array. If None, the length can be arbitrary.",
    )
    description: str = Field(
        ..., description="A human-readable description of the type."  # Required field
    )

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "np.ndarray"
        super().__init__(**data)

    def to_python_type(self) -> str:
        return "np.ndarray"

    def to_markdown(self, indent: int = 0) -> List[str]:
        spaces = "  " * indent
        desc = f"{self.description}"
        if self.length is not None:
            desc += f" Expected length: {self.length}."
        element_markdown = self.element_type.to_markdown(indent + 1)
        return [f"{spaces}- **NumPy ndarray**: {desc}"] + element_markdown

    def __str__(self) -> str:
        return f"np.ndarray[{self.element_type}]"

    def check_value(self, value: Any) -> None:
        if not isinstance(value, np.ndarray):
            raise ValueError(f"Expected np.ndarray, got {type(value).__name__}")
        if self.length is not None and value.size != self.length:
            raise ValueError(
                f"Expected ndarray of size {self.length}, got size {value.size}"
            )
        for index, elem in enumerate(value.flat):
            try:
                self.element_type.check_value(elem)
            except ValueError as ve:
                raise ValueError(f"ndarray element at flat index {index}: {ve}") from ve

    def type_schema(self) -> Dict[str, Any]:
        schema = {
            "type": "array",
            "items": self.element_type.type_schema(),
            "description": self.description,
        }
        if self.length is not None:
            schema["length"] = self.length
        return schema


class SklearnClassType(BaseType):
    type: Literal["class"]
    name: str = Field(
        ...,
        description="The fully qualified name of any class in the sklearn library.",
    )  # Required field
    description: str = Field(
        ..., description="A human-readable description of the type."  # Required field
    )

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "class"
        super().__init__(**data)

    def to_python_type(self) -> str:
        return self.name

    def to_markdown(self, indent: int = 0) -> List[str]:
        spaces = "  " * indent
        return [f"{spaces}- **Class** {self.name}: {self.description}"]

    def __str__(self) -> str:
        return self.name

    def import_class_safe(self, fully_qualified_name):
        """
        Safely imports a class from its fully qualified name with error handling.

        Args:
            fully_qualified_name (str): The fully qualified name of the class.

        Returns:
            type: The class type if successful.
            None: If import fails.
        """
        try:
            module_path, class_name = fully_qualified_name.rsplit(".", 1)
        except ValueError:
            error(
                f"Error: '{fully_qualified_name}' is not a valid fully qualified name."
            )
            return None

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            error(f"Error: Could not import module '{module_path}'.\n{e}")
            return None

        try:
            cls = getattr(module, class_name)
        except AttributeError:
            error(
                f"Error: Module '{module_path}' does not have a class named '{class_name}'."
            )
            return None

        if not isinstance(cls, type):
            error(f"Error: '{class_name}' in module '{module_path}' is not a class.")
            return None

        return cls

    def check_value(self, value: Any) -> None:
        # Assuming 'name' is the class name, you might need a mapping to actual classes
        # For demonstration, we'll check if the value is an instance of any class
        t = type(value)
        fully_qualified_name = f"{t.__module__}.{t.__name__}"
        self_t = self.import_class_safe(fully_qualified_name)
        print(t, self_t)
        if t == self_t:
            return

        raise ValueError(f"Expected type '{self.name}', got {t.__name__}")

    def type_schema(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "type": "class",
            "class_name": self.name,
        }


class ExtendedType(BaseModel):
    the_type: TypeRepresentation
    description: str = Field(
        ...,  # Required field
        description="A description of what this type represents. Indicate how to interpret each component of the type.",
    )

    def to_python_type(self) -> str:
        """
        Converts the custom TypeRepresentation into a string representation of the Python type.
        """
        return self.the_type.to_python_type()

    def to_markdown(self, include_description=False) -> str:
        """
        Generate a human-readable Markdown description for the type.
        """
        lines = [self.description] if include_description else []
        lines += self.the_type.to_markdown()
        return "\n".join(lines)

    def to_description(self) -> str:
        return self.description + ".  " + str(self.the_type)

    def __str__(self) -> str:
        """
        Returns a string combining the Python type with extra composite type information.
        """
        return str(self.the_type)

    def is_None_type(self) -> bool:
        return isinstance(self.the_type, NoneType)

    @classmethod
    def from_value(cls, value) -> "ExtendedType":

        def infer_type_of_elements(elements: Iterable) -> "TypeRepresentation":
            element_types = [infer_type(item) for item in elements]
            unique_types = {et.model_dump_json() for et in element_types}
            if len(unique_types) == 1:
                return element_types[0]
            else:
                return AnyType(
                    description="Automatically inferred as Any due to mixed types."
                )

        def infer_type(val) -> "TypeRepresentation":
            if val is None:
                return NoneType(description="Automatically inferred as NoneType.")
            elif isinstance(val, bool):  # do before int
                return BoolType(description="Automatically inferred as BoolType.")
            elif isinstance(val, int) or isinstance(val, np.integer):
                return IntType(description="Automatically inferred as IntType.")
            elif isinstance(val, float) or isinstance(val, np.float64):
                return FloatType(description="Automatically inferred as FloatType.")
            elif isinstance(val, str) or isinstance(val, np.str_):
                return StrType(description="Automatically inferred as StrType.")
            elif isinstance(val, list):
                try:
                    element_type = infer_type_of_elements(val)
                    return ListType(
                        element_type=element_type,
                        length=len(val),
                        description="Automatically inferred as ListType.",
                    )
                except:
                    return AnyType(
                        description="Automatically inferred as AnyType due to error."
                    )
            elif isinstance(val, set):
                try:
                    element_type = infer_type_of_elements(val)
                    return SetType(
                        element_type=element_type,
                        description="Automatically inferred as SetType.",
                    )
                except:
                    return AnyType(
                        description="Automatically inferred as AnyType due to error."
                    )
            # elif isinstance(val, tuple):
            #     return TupleType(
            #         elements=[infer_type(item) for item in val],
            #         description="Automatically inferred as TupleType.",
            #     )
            elif isinstance(val, dict):
                # Decide between RecordType and DictType
                if all(isinstance(k, str) for k in val.keys()):
                    items = [
                        KeyType(
                            key=k,
                            type=infer_type(v),
                            description=f"Automatically inferred type for key '{k}'.",
                        )
                        for k, v in val.items()
                    ]
                    return RecordType(
                        name="AutoGeneratedRecord",
                        items=items,
                        description="Automatically inferred as RecordType with string keys.",
                    )
                else:
                    # For heterogeneous keys, use DictType with key and value types inferred
                    key_type = infer_type_of_elements(val.keys())
                    value_type = infer_type_of_elements(val.values())
                    return DictType(
                        key_type=key_type,
                        value_type=value_type,
                        description="Automatically inferred as DictType with mixed key types.",
                    )
            elif isinstance(val, pd.DataFrame):
                if val.empty:
                    return PDDataFrameType(
                        columns=[],
                        description="Automatically inferred as empty PDDataFrameType.",
                    )
                columns = [
                    KeyType(
                        key=col,
                        type=infer_type(val[col].iloc[0]),
                        description=f"Automatically inferred type for column '{col}'.",
                    )
                    for col in val.columns
                ]
                return PDDataFrameType(
                    columns=columns,
                    description="Automatically inferred as PDDataFrameType.",
                )
            elif isinstance(val, pd.Series):
                if val.empty:
                    element = AnyType(
                        description="Automatically inferred as AnyType due to empty Series."
                    )
                else:
                    element = infer_type(val.iloc[0])
                return PDSeriesType(
                    element_type=element,
                    description="Automatically inferred as PDSeriesType.",
                )
            elif isinstance(val, np.ndarray):
                if val.size == 0:
                    element = AnyType(
                        description="Automatically inferred as AnyType due to empty ndarray."
                    )
                else:
                    element = infer_type(val.flat[0])
                return NumpyNdarrayType(
                    element_type=element,
                    length=val.size,
                    description="Automatically inferred as NumpyNdarrayType.",
                )
            elif not isinstance(val, type):
                return SklearnClassType(
                    name=str(type(val)),
                    description="Automatically inferred as SklearnClassType.",
                )
            return AnyType(description="Automatically inferred as AnyType.")

        return cls(
            the_type=infer_type(value), description="Automatically generated type"
        )

    def check_value(self, value: Any) -> None:
        """
        Determines whether the given value conforms to the current type.
        Raises:
            ValueError: If the value does not conform to the type.
        """
        try:
            self.the_type.check_value(value)
        except Exception as e:
            raise ValueError(f"The output value does not conform to type: {e}") from e

    def type_schema(self) -> Dict[str, Any]:
        """
        Generates a JSON schema for the ExtendedType.
        """
        return {"description": self.description, "type": self.the_type.type_schema()}


def schema_to_text(schema: Dict[str, Any]) -> str:
    type_width = 60

    def indent_lines(s):
        lines = s.split("\n")
        return "\n".join([" " + line for line in lines])

    def process_schema(sch: Dict[str, Any], pre_comment: str = "") -> str:
        lines = []

        description = sch.get("description", "")

        if sch["type"] == "Any":
            lines.append(f"Any{pre_comment}  # {description}")
        elif sch["type"] == "None":
            lines.append(f"None{pre_comment}  # {description}")
        elif sch["type"] == "float":
            lines.append(f"float{pre_comment}  # {description}")
        elif sch["type"] == "int":
            lines.append(f"int{pre_comment}  # {description}")
        elif sch["type"] == "str":
            lines.append(f"str{pre_comment}  # {description}")
        elif sch["type"] == "bool":
            lines.append(f"bool{pre_comment}  # {description}")
        elif sch["type"] == "optional":
            wrapped = process_schema(sch["wrapped"])
            # lines.append(f"{'Optional[':<{type_width}} # {description}")
            lines.append("Optional[")
            lines.append(f"{indent_lines(wrapped)}")
            lines.append(f"]{pre_comment}")
        elif sch["type"] == "dict":
            # lines.append(f"{'Dict[':<{type_width}} # {description}")
            lines.append(f"Dict[")
            # Process keys
            # key_schema = sch["keys"]
            # key_str = process_schema(key_schema, pre_comment=",")
            key_str = f"str  #{description}"
            lines.append(f"{indent_lines(key_str)}")
            # Process values
            value_schema = sch["values"]
            value_str = process_schema(value_schema)
            lines.append(f"{indent_lines(value_str)}")
            lines.append(f"]{pre_comment}")
        elif sch["type"] == "record":
            # lines.append(f"{'{':<{type_width}} # {description}")
            lines.append("{")
            properties = sch["properties"]
            body = []
            for key, prop_schema in properties.items():
                prop_str = process_schema(prop_schema)
                body.append(f"'{key}': {prop_str}")
            lines.append(indent_lines("\n".join(body)))
            lines.append(f"}}{pre_comment}")
        elif sch["type"] == "dataframe":
            # lines.append(f"{'DataFrame':<{type_width}} # {description}")
            lines.append("DataFrame[")
            properties = sch["properties"]
            body = []
            for key, prop_schema in properties.items():
                prop_str = process_schema(prop_schema)
                body.append(f"'{key}': {prop_str}")
            lines.append(indent_lines("\n".join(body)))
            lines.append(f"]{pre_comment}")
        elif sch["type"] in ["array", "list", "set", "series"]:
            kind = {
                "array": "Array",
                "list": "List",
                "set": "Set",
                "series": "Series",
            }[sch["type"]]
            items_schema = sch["items"]
            items_str = process_schema(items_schema)
            length = sch.get("length")
            if length:
                # lines.append(f"# {description} (length {length})")
                lines.append(f"{kind}[ (length {length})")
            else:
                # lines.append(f"# {description}")
                lines.append(f"{kind}[")
            lines.append(f"{indent_lines(items_str)}")
            lines.append(f"]{pre_comment}")
        elif sch["type"] == "class":
            lines.append(f"{sch['class_name']}{pre_comment}  # {description}")
        else:
            lines.append(f"Any  # Unknown type: {description}")

        return "\n".join(lines)

    return process_schema(schema["type"])


# -----------------------
# Update Forward References
# -----------------------

# Re-defining TypeRepresentation with updated classes
TypeRepresentation = Union[
    IntType,
    BoolType,
    StrType,
    NoneType,
    FloatType,
    OptionalType,
    ListType,
    RecordType,
    DictType,
    # TupleType,
    SklearnClassType,
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
RecordType.model_rebuild()
DictType.model_rebuild()  # Newly added
# TupleType.model_rebuild()
SklearnClassType.model_rebuild()
SetType.model_rebuild()
PDDataFrameType.model_rebuild()
PDSeriesType.model_rebuild()
NumpyNdarrayType.model_rebuild()
ExtendedType.model_rebuild()


# Hack to get schema for ama.py
class update_node(BaseModel):
    id: str = Field(description="The id of the node to modify.")
    label: str = Field(
        description="The new label of the node.  Keep in sync with the requirements, algorithm, and code."
    )
    requirements: List[str] = Field(
        description="A list of requirements that must be true of the return value for the function.  Describe the representation of the return value as well."
    )
    return_type: ExtendedType = Field(description="The return type of the node.")
    algorithm: List[str] = Field(description="The algorithm of the node.")
    code: List[str] = Field(
        description="The code for the node.  Only modify if there is already an code.  The code should be a list of strings, one for each line of code.  The signature must match the original version, except for the return type"
    )


if __name__ == "__main__":

    from openai import OpenAI
    import openai

    value = {
        "Adelie": {
            "coefficients": [
                np.float64(0.13565399039812356),
                np.float64(13.035777373288223),
            ],
            "rsquared": 0.1104059858867179,
            "p_value": np.nan,
        },
        "Chinstrap": {
            "coefficients": [
                np.float64(0.22081333931715394),
                np.float64(5.593376083129094),
            ],
            "rsquared": 0.22241342558500843,
            "p_value": np.nan,
        },
        "Gentoo": {
            "coefficients": [
                np.float64(0.313282219872178),
                np.float64(-20.48788794887121),
            ],
            "rsquared": 0.44116860076801545,
            "p_value": np.nan,
        },
    }

    print(ExtendedType.from_value(value))

    print("---")
    print(openai.pydantic_function_tool(update_node))
    print("---")

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
    print(type_representation)

    print(ExtendedType.from_value(LinearRegression()))
    # print(type_representation.to_markdown())

    # print(json.dumps(TypeDeclarationModel.model_json_schema(), indent=2))

    prompts = [
        "Give me a type for a data frame with three float columns 'A', 'B', and 'Cow'.",
        # "Give me a dictionary with keys 'name' and 'age' and values of type str and int respectively.",
        # "Give me a list of dictionaries with keys 'name' and 'age' and values of type str and int respectively.",
        # "Give me a tuple of two integers.",
        # "Give me a set of strings.",
        # "Give me a numpy ndarray of floats.",
        "Give me a pandas series of integers.",
        "Give me a dictionary for the results of a sklearn LinearRegression model.",
        """
        Give me a dataframe type for this data file:
            species,"Beak length, mm","Beak depth, mm"
            fortis,9.4,8.0
            fortis,9.2,8.3 
            scandens,13.9,8.4
        """,
        "You are going to compute a linear regression for me.  Give me a return type.  Use a Sklearn object.",
        "You are going to compute a logistic regression for me.  Give me a return type.  Use a Sklearn object.",
    ]

    for p in prompts:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-11-20",
            response_format=ExtendedType,
            messages=[{"role": "system", "content": p}],
        )

        print(completion.choices[0].message.parsed)

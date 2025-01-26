from __future__ import annotations
from typing import Any, Iterable, Set, Tuple, Union, Dict, List, Literal, TypedDict
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from abc import abstractmethod

from typing import Any, List, Union, Iterable, Optional, Literal
from pydantic import BaseModel, Field
from abc import abstractmethod
import numpy as np
import pandas as pd


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


class IntType(BaseType):
    type: Literal["int"]  # = "int"

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

    def check_value(self, value: Any) -> None:
        if (isinstance(value, int) or isinstance(value, np.integer)) and not isinstance(
            value, bool
        ):
            return
        raise ValueError(f"Expected int, got {type(value).__name__}")


class BoolType(BaseType):
    type: Literal["bool"]  # = "bool"

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

    def check_value(self, value: Any) -> None:
        if isinstance(value, bool):
            return
        raise ValueError(f"Expected bool, got {type(value).__name__}")


class StrType(BaseType):
    type: Literal["str"]  # = "str"

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

    def check_value(self, value: Any) -> None:
        if isinstance(value, str):
            return
        raise ValueError(f"Expected str, got {type(value).__name__}")


class AnyType(BaseType):
    type: Literal["Any"]  # = "Any"

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

    def check_value(self, value: Any) -> None:
        return  # Always matches


class NoneType(BaseType):
    type: Literal["None"]  # = "None"

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

    def check_value(self, value: Any) -> None:
        if value is None:
            return
        raise ValueError(f"Expected None, got {type(value).__name__}")


class FloatType(BaseType):
    type: Literal["float"]  # = "float"

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

    def check_value(self, value: Any) -> None:
        if isinstance(value, float):
            return
        raise ValueError(f"Expected float, got {type(value).__name__}")


class OptionalType(BaseType):
    type: Literal["Optional"]  # = "Optional"
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

    def check_value(self, value: Any) -> None:
        if value is None:
            return
        self.wrapped_type.check_value(value)


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

    def check_value(self, value: Any) -> None:
        self.type.check_value(value)


class ListType(BaseType):
    type: Literal["List"]  # = "List"
    element_type: "TypeRepresentation"
    length: Optional[int] = Field(
        # default=None,
        description="The expected length of the list. If None, the length can be arbitrary.",
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


class TypedDictType(BaseType):
    type: Literal["TypedDict"]  # = "TypedDict"
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
        map_str = ", ".join(elems)
        return f"TypedDict('{self.name}', {{{map_str}}})"

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
        map_str = ", ".join(elems)
        return f"TypedDict('{self.name}', {{{map_str}}})"

    def check_value(self, value: Any) -> None:
        if not isinstance(value, dict):
            raise ValueError(f"Expected dict, got {type(value).__name__}")
        for item in self.items:
            if item.key not in value:
                raise ValueError(f"Missing key '{item.key}' in dictionary")
            try:
                item.type.check_value(value[item.key])
            except ValueError as ve:
                raise ValueError(f"Key '{item.key}': {ve}") from ve


class TupleType(BaseType):
    type: Literal["Tuple"]  # = "Tuple"
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

    def check_value(self, value: Any) -> None:
        if not isinstance(value, tuple):
            raise ValueError(f"Expected tuple, got {type(value).__name__}")
        if len(value) != len(self.elements):
            raise ValueError(
                f"Expected tuple of length {len(self.elements)}, got length {len(value)}"
            )
        for index, (elem_type, elem_value) in enumerate(zip(self.elements, value)):
            try:
                elem_type.check_value(elem_value)
            except ValueError as ve:
                raise ValueError(f"Tuple element at index {index}: {ve}") from ve


class SetType(BaseType):
    type: Literal["Set"]  # = "Set"
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

    def check_value(self, value: Any) -> None:
        if not isinstance(value, set):
            raise ValueError(f"Expected set, got {type(value).__name__}")
        for elem in value:
            try:
                self.element_type.check_value(elem)
            except ValueError as ve:
                raise ValueError(f"Set element '{elem}': {ve}") from ve


class PDDataFrameType(BaseType):
    type: Literal["pd.DataFrame"]  # = "pd.DataFrame"
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

    def check_value(self, value: Any) -> None:
        if not isinstance(value, pd.DataFrame):
            raise ValueError(f"Expected pd.DataFrame, got {type(value).__name__}")
        for col in self.columns:
            if col.key not in value.columns:
                raise ValueError(f"Missing column '{col.key}' in DataFrame")
            for index, item in value[col.key].items():
                try:
                    col.type.check_value(item)
                except ValueError as ve:
                    raise ValueError(
                        f"DataFrame column '{col.key}', row {index}: {ve}"
                    ) from ve


class PDSeriesType(BaseType):
    type: Literal["pd.Series"]  # = "pd.Series"
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

    def check_value(self, value: Any) -> None:
        if not isinstance(value, pd.Series):
            raise ValueError(f"Expected pd.Series, got {type(value).__name__}")
        for index, item in value.items():
            try:
                self.element_type.check_value(item)
            except ValueError as ve:
                raise ValueError(f"Pandas Series at index {index}: {ve}") from ve


class NumpyNdarrayType(BaseType):
    type: Literal["np.ndarray"]  # = "np.ndarray"
    element_type: "TypeRepresentation"
    length: Optional[int] = Field(
        # default=None,
        description="The expected length of the array. If None, the length can be arbitrary.",
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


class ClassType(BaseType):
    type: Literal["class"]  # = "class"
    name: str = Field(description="The name of the class.")

    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "class"
        super().__init__(**data)

    def to_python_type(self) -> str:
        return self.name

    def to_markdown(self, indent: int = 0) -> List[str]:
        spaces = "  " * indent
        return [f"{spaces}- **Class** {self.name}"]

    def __str__(self) -> str:
        return self.name

    def check_value(self, value: Any) -> None:
        # Assuming 'name' is the class name, you might need a mapping to actual classes
        # For demonstration, we'll check if the value is an instance of any class
        if isinstance(value, type):
            return
        raise ValueError(f"Expected type '{self.name}', got {type(value).__name__}")


class DictType(BaseType):
    type: Literal["Dict"]  # = "Dict"
    key_type: "TypeRepresentation" = Field(
        description="The type of the dictionary keys."
    )
    value_type: "TypeRepresentation" = Field(
        description="The type of the dictionary values."
    )
    key_description: Optional[str] = Field(
        description="A description of what the dictionary keys represent."
    )
    value_description: Optional[str] = Field(
        description="A description of what the dictionary values represent.",
    )

    def __init__(self, **data):
        super().__init__(**data)

    def to_python_type(self) -> str:
        return f"Dict[{self.key_type.to_python_type()}, {self.value_type.to_python_type()}]"

    def to_markdown(self, indent: int = 0) -> List[str]:
        spaces = "  " * indent
        key_markdown = self.key_type.to_markdown(indent + 1)
        value_markdown = self.value_type.to_markdown(indent + 1)
        desc = [f"{spaces}- **Dict** with:"]

        if self.key_description:
            desc.append(f"{spaces}  - **Keys**: {self.key_description}")
        else:
            desc += [f"{spaces}  - **Keys**:"] + key_markdown

        if self.value_description:
            desc.append(f"{spaces}  - **Values**: {self.value_description}")
        else:
            desc += [f"{spaces}  - **Values**:"] + value_markdown

        return desc

    def __str__(self) -> str:
        return f"Dict[{self.key_type}, {self.value_type}]"

    def check_value(self, value: Any) -> None:
        if not isinstance(value, dict):
            raise ValueError(f"Expected dict, got {type(value).__name__}")
        for k, v in value.items():
            try:
                self.key_type.check_value(k)
            except ValueError as ve:
                raise ValueError(f"Dictionary key '{k}': {ve}") from ve
            try:
                self.value_type.check_value(v)
            except ValueError as ve:
                raise ValueError(f"Dictionary value for key '{k}': {ve}") from ve


# Update TypeRepresentation to include DictType
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
    DictType,  # Newly added
    # TupleType,
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
            elif isinstance(val, int) or isinstance(val, np.integer):
                return IntType()
            elif isinstance(val, float) or isinstance(val, np.float64):
                return FloatType()
            elif isinstance(val, str) or isinstance(val, np.str_):
                return StrType()
            elif isinstance(val, list):
                try:
                    element_type = infer_type_of_elements(val)
                    return ListType(element_type=element_type, length=len(val))
                except:
                    return AnyType()
            elif isinstance(val, set):
                try:
                    element_type = infer_type_of_elements(val)
                    return SetType(element_type=element_type)
                except:
                    return AnyType()
            elif isinstance(val, tuple):
                return TupleType(elements=[infer_type(item) for item in val])
            elif isinstance(val, dict):
                # Decide between TypedDictType and DictType
                if all(isinstance(k, str) for k in val.keys()):
                    items = [
                        KeyType(key=k, type=infer_type(v), description="")
                        for k, v in val.items()
                    ]
                    return TypedDictType(name="AutoGeneratedTypedDict", items=items)
                else:
                    # For heterogeneous keys, use DictType with key and value types inferred
                    key_type = infer_type_of_elements(val.keys())
                    value_type = infer_type_of_elements(val.values())
                    return DictType(
                        key_type=key_type,
                        value_type=value_type,
                        key_description="",
                        value_description="",
                    )
            elif isinstance(val, pd.DataFrame):
                # print(val.info())
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

    def check_value(self, value: Any) -> None:
        """
        Determines whether the given value conforms to the current type.
        Raises:
            ValueError: If the value does not conform to the type.
        """
        self.the_type.check_value(value)


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
    DictType,  # Newly added
    # TupleType,
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
DictType.model_rebuild()  # Newly added
TupleType.model_rebuild()
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

    # extended_type = ExtendedType(the_type=TypeRepresentation TypedDict(
    #     "SimpleLinearRegressionResultsDict",
    #     {
    #         "species": str,
    #         "coefficients": List[float],
    #         "rsquared": float,
    #         "p_value": float,
    #     },
    # ),

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

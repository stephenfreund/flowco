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

    def validate(self, value: Any) -> bool:
        return isinstance(value, int) and not isinstance(
            value, bool
        )  # bool is subclass of int


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

    def validate(self, value: Any) -> bool:
        return isinstance(value, bool)


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

    def validate(self, value: Any) -> bool:
        return isinstance(value, str)


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

    def validate(self, value: Any) -> bool:
        return True


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

    def validate(self, value: Any) -> bool:
        return value is None


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

    def validate(self, value: Any) -> bool:
        return isinstance(value, float)


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
    type: Literal["List"]  # = "List"
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

    def validate(self, value: Any) -> bool:
        if not isinstance(value, tuple):
            return False
        if len(value) != len(self.elements):
            return False
        return all(elem_type.validate(v) for elem_type, v in zip(self.elements, value))


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

    def validate(self, value: Any) -> bool:
        if not isinstance(value, set):
            return False
        return all(self.element_type.validate(elem) for elem in value)


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

    def validate(self, value: Any) -> bool:
        if not isinstance(value, pd.Series):
            return False
        for item in value:
            if not self.element_type.validate(item):
                return False
        return True


class NumpyNdarrayType(BaseType):
    type: Literal["np.ndarray"]  # = "np.ndarray"
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

    def to_markdown(self, include_description=False) -> str:
        """
        Generate a human-readable Markdown description for the type.
        """
        lines = [self.description] if include_description else []
        lines += self.the_type.to_markdown()
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
                    return ListType(element_type=element_type, length=len(val))
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

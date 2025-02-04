import pytest
import numpy as np
import pandas as pd
from typing import Any
from flowco.dataflow.extended_type import *


# Assuming all your classes are defined in a module named `type_representation`
# from type_representation import (
#     IntType, BoolType, StrType, FloatType, NoneType, AnyType,
#     OptionalType, ListType, SetType, TupleType, DictType,
#     RecordType, PDDataFrameType, PDSeriesType, NumpyNdarrayType,
#     ClassType, ExtendedType
# )

# For demonstration, we'll assume the classes are available in the current namespace
# If they are in a different module, uncomment the import statement above and adjust accordingly

# -----------------------
# Test Cases for IntType
# -----------------------


def test_int_type_valid():
    int_type = IntType()
    try:
        int_type.check_value(10)
    except ValueError:
        pytest.fail(
            "IntType.check_value() raised ValueError unexpectedly for valid int."
        )


def test_int_type_invalid_str():
    int_type = IntType()
    with pytest.raises(ValueError) as exc_info:
        int_type.check_value("10")
    assert "Expected int, got str" in str(exc_info.value)


def test_int_type_invalid_bool():
    int_type = IntType()
    with pytest.raises(ValueError) as exc_info:
        int_type.check_value(True)
    assert "Expected int, got bool" in str(exc_info.value)


def test_int_type_invalid_float():
    int_type = IntType()
    with pytest.raises(ValueError) as exc_info:
        int_type.check_value(10.5)
    assert "Expected int, got float" in str(exc_info.value)


# -----------------------
# Test Cases for BoolType
# -----------------------


def test_bool_type_valid():
    bool_type = BoolType()
    try:
        bool_type.check_value(True)
        bool_type.check_value(False)
    except ValueError:
        pytest.fail(
            "BoolType.check_value() raised ValueError unexpectedly for valid bool."
        )


def test_bool_type_invalid_int():
    bool_type = BoolType()
    with pytest.raises(ValueError) as exc_info:
        bool_type.check_value(1)
    assert "Expected bool, got int" in str(exc_info.value)


def test_bool_type_invalid_str():
    bool_type = BoolType()
    with pytest.raises(ValueError) as exc_info:
        bool_type.check_value("True")
    assert "Expected bool, got str" in str(exc_info.value)


# -----------------------
# Test Cases for StrType
# -----------------------


def test_str_type_valid():
    str_type = StrType()
    try:
        str_type.check_value("hello")
    except ValueError:
        pytest.fail(
            "StrType.check_value() raised ValueError unexpectedly for valid str."
        )


def test_str_type_invalid_int():
    str_type = StrType()
    with pytest.raises(ValueError) as exc_info:
        str_type.check_value(123)
    assert "Expected str, got int" in str(exc_info.value)


def test_str_type_invalid_none():
    str_type = StrType()
    with pytest.raises(ValueError) as exc_info:
        str_type.check_value(None)
    assert "Expected str, got NoneType" in str(exc_info.value)


# -----------------------
# Test Cases for FloatType
# -----------------------


def test_float_type_valid():
    float_type = FloatType()
    try:
        float_type.check_value(10.5)
        float_type.check_value(np.float64(20.3))
    except ValueError:
        pytest.fail(
            "FloatType.check_value() raised ValueError unexpectedly for valid float."
        )


def test_float_type_invalid_int():
    float_type = FloatType()
    with pytest.raises(ValueError) as exc_info:
        float_type.check_value(10)
    assert "Expected float, got int" in str(exc_info.value)


def test_float_type_invalid_str():
    float_type = FloatType()
    with pytest.raises(ValueError) as exc_info:
        float_type.check_value("10.5")
    assert "Expected float, got str" in str(exc_info.value)


# -----------------------
# Test Cases for NoneType
# -----------------------


def test_none_type_valid():
    none_type = NoneType()
    try:
        none_type.check_value(None)
    except ValueError:
        pytest.fail("NoneType.check_value() raised ValueError unexpectedly for None.")


def test_none_type_invalid():
    none_type = NoneType()
    with pytest.raises(ValueError) as exc_info:
        none_type.check_value(0)
    assert "Expected None, got int" in str(exc_info.value)


# -----------------------
# Test Cases for AnyType
# -----------------------


def test_any_type_valid():
    any_type = AnyType()
    try:
        any_type.check_value(10)
        any_type.check_value("hello")
        any_type.check_value(None)
        any_type.check_value([1, 2, 3])
    except ValueError:
        pytest.fail(
            "AnyType.check_value() raised ValueError unexpectedly for any value."
        )


# -----------------------
# Test Cases for OptionalType
# -----------------------


def test_optional_type_valid_with_value():
    optional_int = OptionalType(wrapped_type=IntType())
    try:
        optional_int.check_value(5)
    except ValueError:
        pytest.fail(
            "OptionalType.check_value() raised ValueError unexpectedly for valid int."
        )


def test_optional_type_valid_with_none():
    optional_int = OptionalType(wrapped_type=IntType())
    try:
        optional_int.check_value(None)
    except ValueError:
        pytest.fail(
            "OptionalType.check_value() raised ValueError unexpectedly for None."
        )


def test_optional_type_invalid():
    optional_int = OptionalType(wrapped_type=IntType())
    with pytest.raises(ValueError) as exc_info:
        optional_int.check_value("not an int")
    assert "Expected int, got str" in str(exc_info.value)


# -----------------------
# Test Cases for ListType
# -----------------------


def test_list_type_valid():
    list_of_int = ListType(element_type=IntType(), length=None)
    try:
        list_of_int.check_value([1, 2, 3])
    except ValueError:
        pytest.fail(
            "ListType.check_value() raised ValueError unexpectedly for valid list of ints."
        )


def test_list_type_valid_with_length():
    list_of_str = ListType(element_type=StrType(), length=3)
    try:
        list_of_str.check_value(["a", "b", "c"])
    except ValueError:
        pytest.fail(
            "ListType.check_value() raised ValueError unexpectedly for valid list of strs with correct length."
        )


def test_list_type_invalid_length():
    list_of_str = ListType(element_type=StrType(), length=2)
    with pytest.raises(ValueError) as exc_info:
        list_of_str.check_value(["a", "b", "c"])
    assert "Expected list of length 2, got length 3" in str(exc_info.value)


def test_list_type_invalid_element():
    list_of_int = ListType(element_type=IntType(), length=None)
    with pytest.raises(ValueError) as exc_info:
        list_of_int.check_value([1, "two", 3])
    assert "List element at index 1: Expected int, got str" in str(exc_info.value)


def test_list_type_empty_list():
    list_of_any = ListType(element_type=AnyType(), length=None)
    try:
        list_of_any.check_value([])
    except ValueError:
        pytest.fail(
            "ListType.check_value() raised ValueError unexpectedly for empty list."
        )


# -----------------------
# Test Cases for SetType
# -----------------------


def test_set_type_valid():
    set_of_bool = SetType(element_type=BoolType())
    try:
        set_of_bool.check_value({True, False})
    except ValueError:
        pytest.fail(
            "SetType.check_value() raised ValueError unexpectedly for valid set of bools."
        )


def test_set_type_invalid_element():
    set_of_int = SetType(element_type=IntType())
    with pytest.raises(ValueError) as exc_info:
        set_of_int.check_value({1, 2, "three"})
    assert "Set element 'three': Expected int, got str" in str(exc_info.value)


def test_set_type_invalid_type():
    set_of_int = SetType(element_type=IntType())
    with pytest.raises(ValueError) as exc_info:
        set_of_int.check_value([1, 2, 3])  # Not a set
    assert "Expected set, got list" in str(exc_info.value)


# -----------------------
# Test Cases for TupleType
# # -----------------------


# def test_tuple_type_valid():
#     tuple_type = TupleType(elements=[IntType(), StrType()])
#     try:
#         tuple_type.check_value((1, "two"))
#     except ValueError:
#         pytest.fail(
#             "TupleType.check_value() raised ValueError unexpectedly for valid tuple."
#         )


# def test_tuple_type_invalid_length():
#     tuple_type = TupleType(elements=[IntType(), StrType()])
#     with pytest.raises(ValueError) as exc_info:
#         tuple_type.check_value((1,))
#     assert "Expected tuple of length 2, got length 1" in str(exc_info.value)


# def test_tuple_type_invalid_element():
#     tuple_type = TupleType(elements=[IntType(), StrType()])
#     with pytest.raises(ValueError) as exc_info:
#         tuple_type.check_value((1, 2))  # Second element should be str
#     assert "Tuple element at index 1: Expected str, got int" in str(exc_info.value)


# def test_tuple_type_nested():
#     nested_tuple = TupleType(
#         elements=[IntType(), TupleType(elements=[StrType(), BoolType()])]
#     )
#     try:
#         nested_tuple.check_value((1, ("two", True)))
#     except ValueError:
#         pytest.fail(
#             "TupleType.check_value() raised ValueError unexpectedly for valid nested tuple."
#         )

#     with pytest.raises(ValueError) as exc_info:
#         nested_tuple.check_value((1, ("two", "not_bool")))
#     assert (
#         "Tuple element at index 1: Tuple element at index 1: Expected bool, got str"
#         in str(exc_info.value)
#     )


# -----------------------
# Test Cases for DictType
# -----------------------


def test_dict_type_valid():
    dict_type = DictType(
        key_type=IntType(),
        value_type=StrType(),
        key_description="Key",
        value_description="Value",
    )
    try:
        dict_type.check_value({1: "one", 2: "two"})
    except ValueError:
        pytest.fail(
            "DictType.check_value() raised ValueError unexpectedly for valid dict."
        )


def test_dict_type_invalid_key():
    dict_type = DictType(
        key_type=IntType(),
        value_type=StrType(),
        key_description="Key",
        value_description="Value",
    )
    with pytest.raises(ValueError) as exc_info:
        dict_type.check_value({"one": "1", 2: "two"})
    assert "Dictionary key 'one': Expected int, got str" in str(exc_info.value)


def test_dict_type_invalid_value():
    dict_type = DictType(
        key_type=IntType(),
        value_type=StrType(),
        key_description="Key",
        value_description="Value",
    )
    with pytest.raises(ValueError) as exc_info:
        dict_type.check_value({1: "one", 2: 2})
    assert "Dictionary value for key '2': Expected str, got int" in str(exc_info.value)


def test_dict_type_invalid_type():
    dict_type = DictType(
        key_type=IntType(),
        value_type=StrType(),
        key_description="Key",
        value_description="Value",
    )
    with pytest.raises(ValueError) as exc_info:
        dict_type.check_value([1, "one"])
    assert "Expected dict, got list" in str(exc_info.value)


# -----------------------
# Test Cases for RecordType
# -----------------------


def test_record_type_valid():
    record_type = RecordType(
        name="Person",
        items=[
            KeyType(key="name", type=StrType(), description="Person's name"),
            KeyType(key="age", type=IntType(), description="Person's age"),
        ],
    )
    try:
        record_type.check_value({"name": "Alice", "age": 30})
    except ValueError:
        pytest.fail(
            "RecordType.check_value() raised ValueError unexpectedly for valid Record."
        )


def test_record_type_missing_key():
    record_type = RecordType(
        name="Person",
        items=[
            KeyType(key="name", type=StrType(), description="Person's name"),
            KeyType(key="age", type=IntType(), description="Person's age"),
        ],
    )
    with pytest.raises(ValueError) as exc_info:
        record_type.check_value({"name": "Alice"})
    assert "Missing key 'age' in dictionary" in str(exc_info.value)


def test_record_type_invalid_value():
    record_type = RecordType(
        name="Person",
        items=[
            KeyType(key="name", type=StrType(), description="Person's name"),
            KeyType(key="age", type=IntType(), description="Person's age"),
        ],
    )
    with pytest.raises(ValueError) as exc_info:
        record_type.check_value({"name": "Alice", "age": "thirty"})
    assert "Key 'age': Expected int, got str" in str(exc_info.value)


def test_record_type_extra_keys():
    # Depending on implementation, extra keys might be allowed
    record_type = RecordType(
        name="Person",
        items=[
            KeyType(key="name", type=StrType(), description="Person's name"),
            KeyType(key="age", type=IntType(), description="Person's age"),
        ],
    )
    try:
        record_type.check_value({"name": "Alice", "age": 30, "gender": "female"})
    except ValueError:
        pytest.fail(
            "RecordType.check_value() raised ValueError unexpectedly for dict with extra keys."
        )


# -----------------------
# Test Cases for PDDataFrameType
# -----------------------


def test_pandas_dataframe_type_valid():
    df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})
    dataframe_type = PDDataFrameType(
        columns=[
            KeyType(key="name", type=StrType(), description="Name of the person"),
            KeyType(key="age", type=IntType(), description="Age of the person"),
        ]
    )
    try:
        dataframe_type.check_value(df)
    except ValueError:
        pytest.fail(
            "PDDataFrameType.check_value() raised ValueError unexpectedly for valid DataFrame."
        )


def test_pandas_dataframe_type_missing_column():
    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob"]
            # "age" column is missing
        }
    )
    dataframe_type = PDDataFrameType(
        columns=[
            KeyType(key="name", type=StrType(), description="Name of the person"),
            KeyType(key="age", type=IntType(), description="Age of the person"),
        ]
    )
    with pytest.raises(ValueError) as exc_info:
        dataframe_type.check_value(df)
    assert "Missing column 'age' in DataFrame" in str(exc_info.value)


def test_pandas_dataframe_type_invalid_value():
    df = pd.DataFrame(
        {"name": ["Alice", "Bob"], "age": [30, "twenty-five"]}  # Invalid age
    )
    dataframe_type = PDDataFrameType(
        columns=[
            KeyType(key="name", type=StrType(), description="Name of the person"),
            KeyType(key="age", type=IntType(), description="Age of the person"),
        ]
    )
    with pytest.raises(ValueError) as exc_info:
        dataframe_type.check_value(df)
    assert "DataFrame column 'age', row 1: Expected int, got str" in str(exc_info.value)


def test_pandas_dataframe_type_empty_dataframe():
    df = pd.DataFrame(columns=["name", "age"])
    dataframe_type = PDDataFrameType(
        columns=[
            KeyType(key="name", type=StrType(), description="Name of the person"),
            KeyType(key="age", type=IntType(), description="Age of the person"),
        ]
    )
    try:
        dataframe_type.check_value(df)
    except ValueError:
        pytest.fail(
            "PDDataFrameType.check_value() raised ValueError unexpectedly for empty DataFrame."
        )


# -----------------------
# Test Cases for PDSeriesType
# -----------------------


def test_pandas_series_type_valid():
    series = pd.Series([1, 2, 3])
    series_type = PDSeriesType(element_type=IntType())
    try:
        series_type.check_value(series)
    except ValueError:
        pytest.fail(
            "PDSeriesType.check_value() raised ValueError unexpectedly for valid Series."
        )


def test_pandas_series_type_invalid_element():
    series = pd.Series([1, "two", 3])
    series_type = PDSeriesType(element_type=IntType())
    with pytest.raises(ValueError) as exc_info:
        series_type.check_value(series)
    assert "Pandas Series at index 1: Expected int, got str" in str(exc_info.value)


def test_pandas_series_type_empty_series():
    series = pd.Series([], dtype=object)
    series_type = PDSeriesType(element_type=AnyType())
    try:
        series_type.check_value(series)
    except ValueError:
        pytest.fail(
            "PDSeriesType.check_value() raised ValueError unexpectedly for empty Series."
        )


# -----------------------
# Test Cases for NumpyNdarrayType
# -----------------------


def test_numpy_ndarray_type_valid():
    ndarray = np.array([1, 2, 3])
    ndarray_type = NumpyNdarrayType(element_type=IntType(), length=3)
    try:
        ndarray_type.check_value(ndarray)
    except ValueError:
        pytest.fail(
            "NumpyNdarrayType.check_value() raised ValueError unexpectedly for valid ndarray."
        )


def test_numpy_ndarray_type_invalid_length():
    ndarray = np.array([1, 2, 3, 4])
    ndarray_type = NumpyNdarrayType(element_type=IntType(), length=3)
    with pytest.raises(ValueError) as exc_info:
        ndarray_type.check_value(ndarray)
    assert "Expected ndarray of size 3, got size 4" in str(exc_info.value)


def test_numpy_ndarray_type_invalid_element():
    ndarray = np.array([1, 2, "three"])
    ndarray_type = NumpyNdarrayType(element_type=IntType(), length=None)
    with pytest.raises(ValueError) as exc_info:
        ndarray_type.check_value(ndarray)
    assert "ndarray element at flat index 0: Expected int, got str_" in str(
        exc_info.value
    )


def test_numpy_ndarray_type_empty_array():
    ndarray = np.array([])
    ndarray_type = NumpyNdarrayType(element_type=AnyType(), length=None)
    try:
        ndarray_type.check_value(ndarray)
    except ValueError:
        pytest.fail(
            "NumpyNdarrayType.check_value() raised ValueError unexpectedly for empty ndarray."
        )


# -----------------------
# Test Cases for ClassType
# -----------------------


# def test_class_type_valid():
#     class MyClass:
#         pass

#     class_type = ClassType(name="MyClass")
#     try:
#         class_type.check_value(MyClass)
#     except ValueError:
#         pytest.fail(
#             "ClassType.check_value() raised ValueError unexpectedly for valid class type."
#         )


# def test_class_type_invalid():
#     class MyClass:
#         pass

#     class_type = ClassType(name="MyClass")
#     with pytest.raises(ValueError) as exc_info:
#         class_type.check_value("NotAClass")
#     assert "Expected type 'MyClass', got str" in str(exc_info.value)


# -----------------------
# Test Cases for ExtendedType
# -----------------------


def test_extended_type_from_value_valid():
    value = {"name": "Alice", "age": 30}
    extended_type = ExtendedType.from_value(value)
    assert (
        extended_type.to_python_type()
        == "Record('AutoGeneratedRecord', {'name': str, 'age': int})"
    )
    try:
        extended_type.check_value(value)
    except ValueError:
        pytest.fail(
            "ExtendedType.check_value() raised ValueError unexpectedly for valid value."
        )


# Okay -- from value, its all good.
# def test_extended_type_from_value_invalid():
#     value = {"name": "Alice", "age": "thirty"}  # age should be int
#     extended_type = ExtendedType.from_value(value)
#     with pytest.raises(ValueError) as exc_info:
#         extended_type.check_value(value)
#     assert "Key 'age': Expected int, got str" in str(exc_info.value)


def test_extended_type_from_value_optional():
    value = None
    extended_type = ExtendedType(
        the_type=OptionalType(wrapped_type=StrType()), description="Optional string"
    )
    try:
        extended_type.check_value(value)
    except ValueError:
        pytest.fail(
            "ExtendedType.check_value() raised ValueError unexpectedly for valid None value."
        )

    extended_type_non_none = ExtendedType(
        the_type=OptionalType(wrapped_type=StrType()), description="Optional string"
    )
    try:
        extended_type_non_none.check_value("Hello")
    except ValueError:
        pytest.fail(
            "ExtendedType.check_value() raised ValueError unexpectedly for valid string value."
        )

    with pytest.raises(ValueError) as exc_info:
        extended_type_non_none.check_value(123)
    assert "Expected str, got int" in str(exc_info.value)


def test_extended_type_from_value_list():
    value = [1, 2, 3]
    extended_type = ExtendedType.from_value(value)
    assert extended_type.to_python_type() == "List[int]"
    try:
        extended_type.check_value(value)
    except ValueError:
        pytest.fail(
            "ExtendedType.check_value() raised ValueError unexpectedly for valid list of ints."
        )

    invalid_value = [1, "two", 3]
    with pytest.raises(ValueError) as exc_info:
        extended_type.check_value(invalid_value)
    assert "List element at index 1: Expected int, got str" in str(exc_info.value)


def test_extended_type_from_value_nested():
    value = {"user": {"name": "Alice", "age": 30}, "active": True}
    extended_type = ExtendedType.from_value(value)
    assert "Record" in extended_type.to_python_type()
    try:
        extended_type.check_value(value)
    except ValueError:
        pytest.fail(
            "ExtendedType.check_value() raised ValueError unexpectedly for valid nested dict."
        )

    invalid_value = {"user": {"name": "Alice", "age": "thirty"}, "active": True}
    with pytest.raises(ValueError) as exc_info:
        extended_type.check_value(invalid_value)
    assert "Key 'user': Key 'age': Expected int, got str" in str(exc_info.value)


# -----------------------
# Test Cases for ExtendedType with DictType
# -----------------------


def test_extended_type_dict_type_valid():
    value = {1: "one", 2: "two"}
    extended_type = ExtendedType.from_value(value)
    assert extended_type.to_python_type() == "Dict[int, str]"
    try:
        extended_type.check_value(value)
    except ValueError:
        pytest.fail(
            "ExtendedType.check_value() raised ValueError unexpectedly for valid Dict[int, str]."
        )


def test_dict_type_valid_with_descriptions():
    dict_type = DictType(
        key_type=IntType(),
        value_type=StrType(),
        key_description="User ID",
        value_description="User Name",
    )
    try:
        dict_type.check_value({1: "Alice", 2: "Bob"})
    except ValueError:
        pytest.fail(
            "DictType.check_value() raised ValueError unexpectedly for valid dict with descriptions."
        )


def test_dict_type_invalid_key_with_descriptions():
    dict_type = DictType(
        key_type=IntType(),
        value_type=StrType(),
        key_description="User ID",
        value_description="User Name",
    )
    with pytest.raises(ValueError) as exc_info:
        dict_type.check_value({"one": "Alice", 2: "Bob"})
    assert "Dictionary key 'one': Expected int, got str" in str(exc_info.value)


def test_dict_type_invalid_value_with_descriptions():
    dict_type = DictType(
        key_type=IntType(),
        value_type=StrType(),
        key_description="User ID",
        value_description="User Name",
    )
    with pytest.raises(ValueError) as exc_info:
        dict_type.check_value({1: "Alice", 2: 2})
    assert "Dictionary value for key '2': Expected str, got int" in str(exc_info.value)


# def test_dict_type_valid_without_descriptions():
#     dict_type = DictType(key_type=StrType(), value_type=FloatType())
#     try:
#         dict_type.check_value({"price": 19.99, "tax": 1.99})
#     except ValueError:
#         pytest.fail(
#             "DictType.check_value() raised ValueError unexpectedly for valid dict without descriptions."
#         )


# def test_dict_type_markdown_with_descriptions():
#     dict_type = DictType(
#         key_type=IntType(),
#         value_type=StrType(),
#         key_description="User ID",
#         value_description="User Name",
#     )
#     markdown = dict_type.to_markdown()
#     expected_markdown = [
#         "- **Dict** with:",
#         "  - **Keys**: User ID",
#         "  - **Values**: User Name",
#     ]
#     assert markdown == expected_markdown


# def test_dict_type_markdown_without_descriptions():
#     dict_type = DictType(key_type=IntType(), value_type=StrType())
#     markdown = dict_type.to_markdown()
#     expected_markdown = [
#         "- **Dict** with:",
#         "  - **Keys**:",
#         "    - **Integer**",
#         "  - **Values**:",
#         "    - **String**",
#     ]
#     assert markdown == expected_markdown


# Okay: key type is Any
# def test_extended_type_dict_type_invalid_key():
#     value = {"one": "1", 2: "two"}
#     extended_type = ExtendedType.from_value(value)
#     with pytest.raises(ValueError) as exc_info:
#         extended_type.check_value(value)
#     assert "Dictionary key 'one': Expected int, got str" in str(exc_info.value)


# Okay: key type is Any
# def test_extended_type_dict_type_invalid_value():
#     value = {1: "one", 2: 2}
#     extended_type = ExtendedType.from_value(value)
#     with pytest.raises(ValueError) as exc_info:
#         extended_type.check_value(value)
#     assert "Dictionary value for key '2': Expected str, got int" in str(exc_info.value)


# -----------------------
# Test Cases for ExtendedType with AnyType
# -----------------------


def test_extended_type_any_type():
    value = {"anything": [1, "two", 3.0], "another": {"nested": True}}
    extended_type = ExtendedType.from_value(value)
    assert (
        extended_type.to_python_type()
        == "Record('AutoGeneratedRecord', {'anything': List[Any], 'another': Record('AutoGeneratedRecord', {'nested': bool})})"
    )
    try:
        extended_type.check_value(value)
    except ValueError:
        pytest.fail(
            "ExtendedType.check_value() raised ValueError unexpectedly for AnyType."
        )


# -----------------------
# Test Cases for Empty Types
# -----------------------


def test_empty_list_type():
    list_type = ListType(element_type=AnyType(), length=0)
    try:
        list_type.check_value([])
    except ValueError:
        pytest.fail(
            "ListType.check_value() raised ValueError unexpectedly for empty list."
        )


def test_empty_set_type():
    set_type = SetType(element_type=AnyType())
    try:
        set_type.check_value(set())
    except ValueError:
        pytest.fail(
            "SetType.check_value() raised ValueError unexpectedly for empty set."
        )


# def test_empty_tuple_type():
#     tuple_type = TupleType(elements=[])
#     try:
#         tuple_type.check_value(())
#     except ValueError:
#         pytest.fail(
#             "TupleType.check_value() raised ValueError unexpectedly for empty tuple."
#         )


def test_empty_dict_type():
    dict_type = DictType(
        key_type=AnyType(),
        value_type=AnyType(),
        key_description=None,
        value_description=None,
    )
    try:
        dict_type.check_value({})
    except ValueError:
        pytest.fail(
            "DictType.check_value() raised ValueError unexpectedly for empty dict."
        )


def test_empty_record_type():
    record_type = RecordType(name="EmptyDict", items=[])
    try:
        record_type.check_value({})
    except ValueError:
        pytest.fail(
            "RecordType.check_value() raised ValueError unexpectedly for empty Record."
        )


def test_empty_pandas_dataframe_type():
    df = pd.DataFrame(columns=["name", "age"])
    dataframe_type = PDDataFrameType(
        columns=[
            KeyType(key="name", type=StrType(), description="Name"),
            KeyType(key="age", type=IntType(), description="Age"),
        ]
    )
    try:
        dataframe_type.check_value(df)
    except ValueError:
        pytest.fail(
            "PDDataFrameType.check_value() raised ValueError unexpectedly for empty DataFrame with defined columns."
        )


def test_empty_pandas_series_type():
    series = pd.Series([], dtype=object)
    series_type = PDSeriesType(element_type=AnyType())
    try:
        series_type.check_value(series)
    except ValueError:
        pytest.fail(
            "PDSeriesType.check_value() raised ValueError unexpectedly for empty Series."
        )


def test_empty_numpy_ndarray_type():
    ndarray = np.array([])
    ndarray_type = NumpyNdarrayType(element_type=AnyType(), length=None)
    try:
        ndarray_type.check_value(ndarray)
    except ValueError:
        pytest.fail(
            "NumpyNdarrayType.check_value() raised ValueError unexpectedly for empty ndarray."
        )


# -----------------------
# Test Cases for ExtendedType with Nested Structures
# -----------------------


def test_extended_type_nested_list_of_dicts():
    value = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    extended_type = ExtendedType.from_value(value)
    assert (
        extended_type.to_python_type()
        == "List[Record('AutoGeneratedRecord', {'id': int, 'name': str})]"
    )
    try:
        extended_type.check_value(value)
    except ValueError:
        pytest.fail(
            "ExtendedType.check_value() raised ValueError unexpectedly for nested list of dicts."
        )

    invalid_value = [
        {"id": 1, "name": "Alice"},
        {"id": "two", "name": "Bob"},  # Invalid id
    ]
    with pytest.raises(ValueError) as exc_info:
        extended_type.check_value(invalid_value)
    assert "List element at index 1: Key 'id': Expected int, got str" in str(
        exc_info.value
    )


def test_extended_type_nested_dict_with_list():
    value = {"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}
    extended_type = ExtendedType.from_value(value)
    try:
        extended_type.check_value(value)
    except ValueError:
        pytest.fail(
            "ExtendedType.check_value() raised ValueError unexpectedly for nested dict with list."
        )

    invalid_value = {
        "users": [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": "twenty-five"},  # Invalid age
        ]
    }
    with pytest.raises(ValueError) as exc_info:
        extended_type.check_value(invalid_value)
    assert "List element at index 1: Key 'age': Expected int, got str" in str(
        exc_info.value
    )


# -----------------------
# Test Cases for ExtendedType with Mixed Types
# -----------------------


def test_extended_type_mixed_types():
    value = {
        "count": 10,
        "active": True,
        "name": "Sample",
        "values": [1.1, 2.2, 3.3],
        "metadata": {"key1": "value1", "key2": "value2"},
    }
    extended_type = ExtendedType.from_value(value)
    assert extended_type.to_python_type().startswith("Record")
    try:
        extended_type.check_value(value)
    except ValueError:
        pytest.fail(
            "ExtendedType.check_value() raised ValueError unexpectedly for mixed types."
        )

    invalid_value = {
        "count": "ten",  # Invalid count
        "active": True,
        "name": "Sample",
        "values": [1.1, 2.2, "three"],  # Invalid value in list
        "metadata": {"key1": "value1", "key2": "value2"},
    }
    with pytest.raises(ValueError) as exc_info:
        extended_type.check_value(invalid_value)
    assert "Key 'count': Expected int, got str" in str(
        exc_info.value
    ) or "List element at index 2: Expected float, got str" in str(exc_info.value)


# -----------------------
# Test Cases for ExtendedType with Numpy Arrays
# -----------------------


def test_extended_type_numpy_array_valid():
    value = np.array([1, 2, 3])
    extended_type = ExtendedType.from_value(value)
    assert "np.ndarray" in extended_type.to_python_type()
    try:
        extended_type.check_value(value)
    except ValueError:
        pytest.fail(
            "ExtendedType.check_value() raised ValueError unexpectedly for valid numpy array."
        )


# Okay it all becomes strings
# def test_extended_type_numpy_array_invalid_element():
#     value = np.array([1, 2, "three"])
#     extended_type = ExtendedType.from_value(value)
#     with pytest.raises(ValueError) as exc_info:
#         extended_type.check_value(value)
#     assert "ndarray element at flat index 2: Expected int, got str" in str(
#         exc_info.value
#     )


def test_extended_type_numpy_array_invalid_length():
    value = np.array([1, 2, 3, 4])
    extended_type = ExtendedType(
        the_type=NumpyNdarrayType(element_type=IntType(), length=3),
        description="Fixed length array",
    )
    with pytest.raises(ValueError) as exc_info:
        extended_type.check_value(value)
    assert "Expected ndarray of size 3, got size 4" in str(exc_info.value)


# -----------------------
# Test Cases for ExtendedType with Record and DictType
# -----------------------


def test_extended_type_typedict_vs_dict():
    # Record with string keys
    record_value = {"name": "Alice", "age": 30}
    extended_record = ExtendedType.from_value(record_value)
    assert isinstance(extended_record.the_type, RecordType)

    # Regular Dict with mixed keys
    dict_value = {1: "one", "two": 2}
    extended_dict = ExtendedType.from_value(dict_value)
    assert isinstance(extended_dict.the_type, DictType)

    try:
        extended_record.check_value(record_value)
    except ValueError:
        pytest.fail(
            "ExtendedType.check_value() raised ValueError unexpectedly for valid Record."
        )

    try:
        extended_dict.check_value(dict_value)
    except ValueError:
        pytest.fail(
            "ExtendedType.check_value() raised ValueError unexpectedly for valid DictType."
        )

    # Invalid Record
    invalid_record = {"name": "Alice", "age": "thirty"}  # age should be int
    with pytest.raises(ValueError) as exc_info:
        extended_record.check_value(invalid_record)
    assert "Key 'age': Expected int, got str" in str(exc_info.value)


# -----------------------
# Running the Tests
# -----------------------

# To run these tests, navigate to the directory containing `test_type_representation.py` and execute:
# pytest test_type_representation.py

# Alternatively, to see more detailed output, run:
# pytest test_type_representation.py -v

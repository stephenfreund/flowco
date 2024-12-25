# from typing import Any, Type, Callable, Union, Optional, Tuple
# import numpy as np
# import pandas as pd
# import re
# from pydantic import BaseModel, Field
# from flowco.builder.type_ops import str_to_type
# from flowco.dataflow.extended_type import ExtendedType, extended_repr, from_value
# from flowco.util.output import warn


# class TypeParser:
#     def __init__(self, type_str):
#         self.type_str = type_str
#         self.pos = 0
#         self.length = len(type_str)

#     def parse(self):
#         node = self.parse_type()
#         self.skip_whitespace()
#         if self.pos != self.length:
#             raise ValueError(f"Unexpected characters at position {self.pos}")
#         return node

#     def skip_whitespace(self):
#         while self.pos < self.length and self.type_str[self.pos].isspace():
#             self.pos += 1

#     def parse_type(self):
#         self.skip_whitespace()
#         base_type = self.parse_identifier()
#         self.skip_whitespace()
#         params = []
#         if self.pos < self.length and self.type_str[self.pos] == '[':
#             self.pos += 1  # skip '['
#             self.skip_whitespace()
#             params = self.parse_type_params()
#             self.skip_whitespace()
#             if self.pos >= self.length or self.type_str[self.pos] != ']':
#                 raise ValueError(f"Expected ']' at position {self.pos}")
#             self.pos += 1  # skip ']'
#         return {"base_type": base_type, "params": params}

#     def parse_identifier(self):
#         self.skip_whitespace()
#         start = self.pos
#         while self.pos < self.length and (self.type_str[self.pos].isalnum() or self.type_str[self.pos] in "._"):
#             self.pos += 1
#         if start == self.pos:
#             raise ValueError(f"Expected identifier at position {self.pos}")
#         return self.type_str[start:self.pos]

#     def parse_type_params(self):
#         self.skip_whitespace()
#         params = []
#         while self.pos < self.length and self.type_str[self.pos] != ']':
#             if self.type_str[self.pos] == "'":
#                 # Parse key: type
#                 key, value = self.parse_key_value_pair()
#                 params.append({"key": key, "value": value})
#             elif self.type_str[self.pos] == '[':
#                 # Parse a list of types
#                 list_types = self.parse_nested_list()
#                 params.append(list_types)
#             else:
#                 # Parse type
#                 param = self.parse_type()
#                 params.append(param)
#             self.skip_whitespace()
#             if self.pos < self.length and self.type_str[self.pos] == ',':
#                 self.pos += 1  # skip ','
#                 self.skip_whitespace()
#         return params

#     def parse_key_value_pair(self):
#         # Parse a key-value pair like "'a': int"
#         # Assume current position is at the opening "'"
#         if self.type_str[self.pos] != "'":
#             raise ValueError(f"Expected \"'\" at position {self.pos}")
#         self.pos += 1  # skip opening "'"
#         start = self.pos
#         while self.pos < self.length and self.type_str[self.pos] != "'":
#             self.pos += 1
#         if self.pos >= self.length:
#             raise ValueError(f"Unterminated key string starting at position {start}")
#         key = self.type_str[start:self.pos]
#         self.pos += 1  # skip closing "'"
#         self.skip_whitespace()
#         if self.pos >= self.length or self.type_str[self.pos] != ':':
#             raise ValueError(f"Expected ':' after key at position {self.pos}")
#         self.pos += 1  # skip ':'
#         self.skip_whitespace()
#         value = self.parse_type()
#         return key, value

#     def parse_nested_list(self):
#         # Parses a nested list of types
#         if self.type_str[self.pos] != '[':
#             raise ValueError(f"Expected '[' at position {self.pos}")
#         self.pos += 1  # skip '['
#         self.skip_whitespace()
#         nested_types = []
#         while self.pos < self.length and self.type_str[self.pos] != ']':
#             type_node = self.parse_type()
#             nested_types.append(type_node)
#             self.skip_whitespace()
#             if self.pos < self.length and self.type_str[self.pos] == ',':
#                 self.pos += 1
#                 self.skip_whitespace()
#         if self.pos >= self.length or self.type_str[self.pos] != ']':
#             raise ValueError(f"Expected ']' at position {self.pos}")
#         self.pos += 1  # skip ']'
#         return nested_types


# def validate_type_node(node):
#     """
#     Recursively validate a parsed type node.

#     Args:
#         node: A dictionary representing the type node.
#     Raises:
#         ValueError: If the type node is invalid.
#     """
#     allowed_base_types = {
#         "int",
#         "bool",
#         "str",
#         "None",
#         "float",
#         "List",
#         "Dict",
#         "Set",
#         "Tuple",
#         "Callable",
#         "pd.DataFrame",
#         "pd.Series",
#         "np.ndarray",
#     }
#     base_type = node["base_type"]
#     if base_type not in allowed_base_types:
#         raise ValueError(f"Unsupported base type '{base_type}'")

#     params = node["params"]
#     if base_type in {"int", "bool", "str", "None", "float"}:
#         if params:
#             raise ValueError(f"Basic type '{base_type}' should not have type parameters")

#     elif base_type == "List":
#         if len(params) != 1:
#             raise ValueError(f"List should have exactly one type parameter, got {len(params)}")
#         param = params[0]
#         validate_param(param)

#     elif base_type == "Set":
#         if len(params) != 1:
#             raise ValueError(f"Set should have exactly one type parameter, got {len(params)}")
#         param = params[0]
#         validate_param(param)

#     elif base_type == "Tuple":
#         if not params:
#             raise ValueError("Tuples must have type parameters")
#         for param in params:
#             validate_param(param)

#     elif base_type == "Dict":
#         if not params:
#             raise ValueError("Dict must have type parameters")
#         # Two cases:
#         # 1. Dict[key_type, value_type] -- generic
#         # 2. Dict['key1': type1, 'key2': type2, ...] -- specific string keys
#         if all("key" in p and "value" in p for p in params):
#             # Specific string keys
#             for p in params:
#                 key = p["key"]
#                 value = p["value"]
#                 if not isinstance(key, str):
#                     raise ValueError(f"Dict keys must be strings, got {type(key)}")
#                 validate_type_node(value)
#         else:
#             # Generic Dict[key_type, value_type]
#             if len(params) != 2:
#                 raise ValueError(f"Dict must have exactly two type parameters, got {len(params)}")
#             key_param = params[0]
#             value_param = params[1]
#             validate_param(key_param)
#             validate_param(value_param)

#     elif base_type == "Callable":
#         # Callable[[arg1, arg2], return_type]
#         # params should have two elements: [ [arg types], return type ]
#         if len(params) != 2:
#             raise ValueError(f"Callable must have exactly two type parameters, got {len(params)}")
#         args_param = params[0]
#         return_param = params[1]
#         if isinstance(args_param, list):
#             for arg in args_param:
#                 validate_type_node(arg)
#         else:
#             raise ValueError("Callable's first parameter should be a list of argument types")
#         validate_param(return_param)

#     elif base_type == "pd.DataFrame":
#         # pd.DataFrame['column1': type1, 'column2': type2, ...]
#         if not params:
#             # pd.DataFrame[] is allowed (unspecified columns)
#             pass
#         else:
#             for param in params:
#                 # param should have "key" and "value"
#                 if not ("key" in param and "value" in param):
#                     raise ValueError("pd.DataFrame type parameters should be 'column': type")
#                 key = param["key"]
#                 value = param["value"]
#                 if not isinstance(key, str):
#                     raise ValueError(f"pd.DataFrame column names must be strings, got {type(key)}")
#                 validate_type_node(value)

#     elif base_type == "pd.Series":
#         if len(params) != 1:
#             raise ValueError(f"pd.Series should have exactly one type parameter, got {len(params)}")
#         param = params[0]
#         validate_param(param)

#     elif base_type == "np.ndarray":
#         if len(params) != 1:
#             raise ValueError(f"np.ndarray should have exactly one type parameter, got {len(params)}")
#         param = params[0]
#         validate_param(param)

#     else:
#         raise ValueError(f"Unhandled base type '{base_type}'")


# def validate_param(param):
#     """
#     Validate a single type parameter, which can be a type node or a generic type.

#     Args:
#         param: A dictionary representing a type node or a list of type nodes.
#     Raises:
#         ValueError: If the type parameter is invalid.
#     """
#     if isinstance(param, dict) and "base_type" in param:
#         # It's a type node
#         validate_type_node(param)
#     elif isinstance(param, list):
#         # It's a list of type nodes (e.g., arguments for Callable)
#         for sub_param in param:
#             validate_param(sub_param)
#     else:
#         # It should not happen, because all params are either type nodes or key-value pairs
#         raise ValueError("Invalid type parameter structure")


# def check_type(extended_type: ExtendedType):
#     """
#     Verify that an ExtendedType instance is well-formed.
#     This includes ensuring that the type string conforms to expected syntax and
#     that all nested types are also well-formed.

#     Args:
#         extended_type: The ExtendedType instance to verify.

#     Raises:
#         ValueError: If the type is not well-formed.
#     """
#     type_str = extended_type.type
#     try:
#         parser = TypeParser(type_str)
#         parsed = parser.parse()
#         validate_type_node(parsed)
#     except Exception as e:
#         raise ValueError(f"Type '{type_str}' is not well-formed: {e}")

# def is_valid_extended_type(extended_type: ExtendedType) -> bool:
#     """
#     Verify that an ExtendedType instance is well-formed.
#     This includes ensuring that the type string conforms to expected syntax and
#     that all nested types are also well-formed.

#     Args:
#         extended_type: The ExtendedType instance to verify.

#     Returns:
#         bool: True if the type is well-formed, False otherwise.
#     """
#     try:
#         check_type(extended_type)
#         return True
#     except Exception as e:
#         warn(e)
#         return False


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
#         from_value((1, "a")).type == "Tuple[int, str]"
#     ), "Failed on Tuple[int, str]"
#     assert from_value((1,)).type == "Tuple[int]", "Failed on Tuple[int]"
#     # Modified dict tests with specific string keys
#     assert from_value({"a": 1, "b": 2}).type == "Dict['a': int, 'b': int]", "Failed on Dict with string keys"
#     assert from_value({"a": 1, "b": "two"}).type == "Dict['a': int, 'b': str]", "Failed on Dict with string keys with different value types"
#     assert from_value({"a": 1, "b": {"c": 3}}).type == "Dict['a': int, 'b': Dict['c': int]]", "Failed on nested Dict with string keys"
#     assert from_value({1, 2, 3}).type == "Set[int]", "Failed on Set[int]"

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
#     assert repr_str == "(1,)", "Failed on extended_repr(single-element tuple)"
#     assert not clipped, "Clipping occurred on extended_repr(single-element tuple)"

#     # Modified dict tests with specific string keys
#     repr_str, clipped = extended_repr({"a": 1, "b": 2})
#     expected_repr = "{'a': 1, 'b': 2}"
#     assert repr_str == expected_repr, "Failed on extended_repr(dict)"
#     assert not clipped, "Clipping occurred on extended_repr(dict)"

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

#     # Tests for check_type
#     # Basic types
#     check_type(from_value(42))
#     check_type(from_value(3.14))
#     check_type(from_value("Hello"))
#     check_type(from_value(True))
#     check_type(from_value(False))
#     check_type(from_value(None))

#     # Collections
#     check_type(from_value([1, 2, 3]))
#     check_type(from_value((1,)))
#     check_type(from_value({"a": 1, "b": 2}))
#     check_type(from_value({"a": 1, "b": "two"}))
#     check_type(from_value({"a": 1, "b": {"c": 3}}))
#     check_type(from_value({1, 2, 3}))

#     # NumPy array
#     check_type(from_value(arr))

#     # Pandas Series
#     check_type(from_value(series))

#     # Pandas DataFrame
#     check_type(from_value(df))
#     check_type(from_value(empty_df))

#     # Nested Structures
#     check_type(from_value(nested_obj))
#     check_type(from_value(nested_list))

#     print("All assertions passed successfully!")

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

#     # Check that well-formed types pass
#     check_type(e1)
#     check_type(e2)
#     check_type(e3)

#     check_type(ExtendedType(type="pd.DataFrame['triple': float, 'vertical': float]"))
#     check_type(ExtendedType(type="Tuple"))

#     print("ExtendedType remove_nonstandard_params assertions passed successfully!")


# if __name__ == "__main__":
#     run_tests()

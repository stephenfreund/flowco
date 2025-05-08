import ast
import json
import re
from typing import Any, Dict, List, Optional

import markdown
from bs4 import BeautifulSoup
from pydantic import BaseModel, ValidationError

from flowco.util.errors import FlowcoError
from flowthon.flowthon import FlowthonNode, FlowthonProgram


# 8. Example Usage
if __name__ == "__main__":
    # Sample input text with imports, table declarations, and functions
    sample_input = '''
import os
from typing import List, Optional

table data.csv

def f(arg1, arg2, arg3):
    """
    This function performs a specific task.

    # Requirements
    - Must handle edge cases
    - Should be efficient

    # Algorithm
    The function uses a recursive approach to solve the problem.

    # Assertions
    - arg1 must be an integer
    - arg2 should not be None
    """
    # Function implementation here
    if arg1 > 0:
        return arg1 + f(arg2, arg3, arg1 - 1)
    else:
        return arg2

table results.csv

def g(x, y):
    """
    Another function that does something else.

    # Requirements
    - Needs to be thread-safe
    - Should return a value promptly

    # Algorithm
    Utilizes an iterative method to process inputs.

    # Assertions
    - x must be a string
    - y must be a list
    """
    # Function implementation here
    result = []
    for item in y:
        result.append(x + str(item))
    return result

def h(a, b):
    """
    A third function with different sections.

    # Preconditions
    - a must be positive
    - b must be non-empty

    # Steps
    Executes a series of operations to achieve the desired outcome.
    """
    # Function implementation here
    if a <= 0:
        raise ValueError("a must be positive")
    if not b:
        raise ValueError("b must be non-empty")
    for i in range(a):
        print(b)
    return True

def invalid_function(c):
    """
    This function has an undefined section.

    # Requirements
    - Must work correctly

    # UndefinedSection
    - This section is not defined in FlowthonNode
    """
    return c
    '''

    try:
        # Parse the input text into a ParsedFile instance
        parsed = FlowthonProgram.from_source(sample_input)

        # Serialize the ParsedFile to JSON
        json_output = json.dumps(parsed.model_dump(), indent=2)
        print(json_output)
    except FlowcoError as e:
        print()
        print(f"{e}")
        print()

    sample_input = '''
import os
from typing import List, Optional

table data.csv

def arg1():
    pass
    
def arg2():
    """
    """
    
def arg3():
    pass

def f(arg1, arg2, arg3):
    """
    This function performs a specific task.

    # Requirements
    - Must handle edge cases
    - Should be efficient

    # Algorithm
    The function uses a recursive approach to solve the problem.

    # Assertions
    - arg1 must be an integer
    - arg2 should not be None
    """
    # Function implementation here
    if arg1 > 0:
        return arg1 + f(arg2, arg3, arg1 - 1)
    else:
        return arg2

table results.csv

def g(f):
    """
    Another function that does something else.

    # Requirements
    - Needs to be thread-safe
    - Should return a value promptly

    # Algorithm
    Utilizes an iterative method to process inputs.

    # Assertions
    - x must be a string
    - y must be a list
    """
    # Function implementation here
    result = []
    for item in y:
        result.append(x + str(item))
    return result
'''

    print("-----------")
    parsed = FlowthonProgram.from_source(sample_input)
    print(parsed.to_source())

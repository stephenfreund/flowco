from tempfile import TemporaryFile
from typing import *

import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Set, Tuple

import pickle
import base64
import pandas as pd
import numpy as np
from typing import Any

from regex import E

from flowco.util.output import error, log, logger

# These are for the typchecker in str_to_type
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import scipy
from typing import *


def str_to_type(type_str: str) -> type:

    try:
        t = eval(type_str)
        #        log(f"Converted {type_str} to {t}")
        return t
    except Exception as e:
        raise SyntaxError(
            f"`{type_str}` is not a valid type.  If that is a TypedDict, inline the definition.  If it is a library class, use the fully qualified type name, e.g. `sklean.linear_model.LinearRegression`."
        )


def types_equal(type1_str: str, type2_str: str) -> bool:
    def check() -> bool:
        if type1_str == type2_str:
            return True

        type1 = str_to_type(
            type1_str
        )  # TODO: Implement a safe parsing method instead of eval.
        type2 = str_to_type(type2_str)

        origin1, origin2 = get_origin(type1), get_origin(type2)
        args1, args2 = get_args(type1), get_args(type2)

        # log(f"Type1: {type1}, Type2: {type2}")
        # log(f"Origin1: {origin1}, Origin2: {origin2}")
        # log(f"Args1: {args1}, Args2: {args2}")

        if origin1 is None and origin2 is None:
            # Both types do not have generic parameters; compare them directly
            return type1 == type2
        elif origin1 is not None and origin2 is not None:
            # Both types have origins; compare origins and their arguments
            return origin1 == origin2 and args1 == args2
        else:
            # One type has generic parameters while the other does not; they are not equal
            return False

    # with logger(f"Comparing '{type1_str}' and '{type2_str}'"):
    result = check()
    # log(f"Result: {result}")
    return result


def encode(value: Any) -> str:
    """
    Encodes a Python value into a base64 string.

    Supported types:
    - int, bool, str, float, None
    - pandas.DataFrame, pandas.Series
    - numpy.ndarray
    - tuple, dict
    """
    # Serialize the object using pickle
    pickled_bytes = pickle.dumps(value)
    # Encode the bytes to a base64 string
    encoded_str = base64.b64encode(pickled_bytes).decode("utf-8")
    return encoded_str


def decode(encoded_str: str) -> Any:
    """
    Decodes a base64 string back into the original Python value.
    """
    # Decode the base64 string to bytes
    pickled_bytes = base64.b64decode(encoded_str.encode("utf-8"))
    # Deserialize the bytes back to a Python object
    value = pickle.loads(pickled_bytes)
    return value


def convert_np_float64(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: convert_np_float64(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_float64(element) for element in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_np_float64(element) for element in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float64):
        return float(obj)
    else:
        return obj


# Example Usage
if __name__ == "__main__":
    # Example data
    data_int = 42
    data_bool = True
    data_str = "Hello, World!"
    data_float = 3.14159
    data_none = None
    data_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    data_series = pd.Series([5, 6, 7])
    data_array = np.array([8, 9, 10])
    data_tuple = (11, 12, 13)
    data_dict = {"key1": "value1", "key2": 2}

    # List of all data
    all_data = [
        data_int,
        data_bool,
        data_str,
        data_float,
        data_none,
        data_df,
        data_series,
        data_array,
        data_tuple,
        data_dict,
    ]

    # Encode and decode each item
    for item in all_data:
        encoded = encode(item)
        decoded = decode(encoded)
        print(f"Original: {item}")
        print(f"Encoded: {encoded}")
        print(f"Decoded: {decoded}")
        print(f"Match: {item == decoded}\n")

    print(str_to_type("plt.Axes"))
    print(str_to_type("None"))
    print(types_equal("None", "plt.Axes"))

    encoded = encode("moo")
    print(encoded)
    with TemporaryFile() as f:
        f.write(encoded.encode())
        f.seek(0)
        print(decode(f.read().decode()))

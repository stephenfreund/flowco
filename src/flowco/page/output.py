import enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr
import pprint

from flowco.builder import type_ops
from flowco.dataflow import extended_type
from flowco.util.output import error, log

from openai.types.chat import ChatCompletionContentPartParam
from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam,
    ImageURL,
)


class OutputType(str, enum.Enum):
    text = "text"
    image = "image"


class ResultValue(BaseModel):
    pickle: str = Field(description="The value of the result, encoded by pickle")
    text: str = Field(description="The text representation of the result")

    def __str__(self) -> str:
        return f"ResultValue(pickle=..., text={self.text})"

    def __init__(self, **data):
        super().__init__(**data)
        self._value = None

    def clipped_repr(self, o: Any, clip_size: Optional[int] = None) -> Tuple[str, bool]:
        """
        Convert the object `o` to a string expression that can be evaluated to reconstruct `o`,
        and indicate whether any clipping was performed.

        Parameters:
        - o: The object to represent.
        - clip_size: If None, represent all elements. Otherwise, limit the representation to
                    at most `clip_size` elements in collections.

        Returns:
        - A tuple containing:
            - The string representation of `o`.
            - A boolean indicating whether any clipping occurred.

        Supported types:
        - Basic Types: int, float, bool, str, None
        - Collections: list, tuple, dict, set
        - NumPy Arrays: np.ndarray
        - Pandas Objects: pd.DataFrame, pd.Series
        - Callables: Function references by name (if possible)
        """

        if clip_size is not None and clip_size <= 0:
            raise ValueError("clip_size must be a positive integer or None.")

        clipping_occurred = False  # Flag to track if any clipping has occurred

        def clip_collection(collection):
            nonlocal clipping_occurred
            if isinstance(collection, (list, tuple, set)):
                if clip_size is not None and len(collection) > clip_size:
                    clipping_occurred = True
                    return list(collection)[:clip_size]
            elif isinstance(collection, dict):
                if clip_size is not None and len(collection) > clip_size:
                    clipping_occurred = True
                    # Sort dict items by key to ensure consistent clipping
                    clipped_items = sorted(
                        collection.items(), key=lambda x: repr(x[0])
                    )[:clip_size]
                    return dict(clipped_items)
            return collection  # No clipping needed

        def process(o: Any) -> Tuple[str, bool]:
            nonlocal clipping_occurred

            if (
                isinstance(o, (bool, int, float, np.signedinteger, np.floating))
                or o is None
            ):
                return repr(o), False

            elif isinstance(o, str):
                return repr(o), False  # use repr to keep the quotes!

            elif isinstance(o, list):
                elements = o
                if clip_size is not None:
                    elements = clip_collection(o)
                elements_reprs = []
                local_clipping = False
                for item in elements:
                    item_repr, item_clipped = process(item)
                    elements_reprs.append(item_repr)
                    if item_clipped:
                        local_clipping = True
                if clip_size is not None and len(o) > clip_size:
                    elements_reprs.append("...")
                    local_clipping = True
                return f"[{', '.join(elements_reprs)}]", local_clipping

            elif isinstance(o, tuple):
                elements = o
                if clip_size is not None:
                    elements = clip_collection(o)
                elements_reprs = []
                local_clipping = False
                for item in elements:
                    item_repr, item_clipped = process(item)
                    elements_reprs.append(item_repr)
                    if item_clipped:
                        local_clipping = True
                if clip_size is not None and len(o) > clip_size:
                    elements_reprs.append("...")
                    local_clipping = True
                # Add a comma for single-element tuples
                if len(o) == 1:
                    return f"({elements_reprs[0]},)", local_clipping
                else:
                    return f"({', '.join(elements_reprs)})", local_clipping

            elif isinstance(o, dict):
                items = o
                if clip_size is not None:
                    items = clip_collection(o)
                # Sort items by key to ensure consistent representation
                sorted_items = sorted(items.items(), key=lambda x: repr(x[0]))
                items_reprs = []
                local_clipping = False
                for k, v in sorted_items:
                    key_repr, key_clipped = process(k)
                    value_repr, value_clipped = process(v)
                    items_reprs.append(f"{key_repr}: {value_repr}")
                    if key_clipped or value_clipped:
                        local_clipping = True
                if clip_size is not None and len(o) > clip_size:
                    items_reprs.append("...")
                    local_clipping = True
                return f"{{{', '.join(items_reprs)}}}", local_clipping

            elif isinstance(o, set):
                elements = o
                if clip_size is not None:
                    elements = clip_collection(o)
                # Sets are unordered; represent as sorted list for consistency
                sorted_elements = sorted(elements, key=lambda x: repr(x))
                elements_reprs = []
                local_clipping = False
                for item in sorted_elements:
                    item_repr, item_clipped = process(item)
                    elements_reprs.append(item_repr)
                    if item_clipped:
                        local_clipping = True
                if clip_size is not None and len(o) > clip_size:
                    elements_reprs.append("...")
                    local_clipping = True
                return f"set([{', '.join(elements_reprs)}])", local_clipping

            elif isinstance(o, np.ndarray):
                if clip_size is not None:
                    # Clip the array to the first clip_size elements along the first axis
                    if o.ndim == 1:
                        clipped = o[:clip_size]
                    else:
                        clipped = o[:clip_size]
                    clipped_flag = len(o) > clip_size
                else:
                    clipped = o
                    clipped_flag = False
                array_list = clipped.tolist()
                dtype = str(o.dtype)
                array_repr, nested_clipping = process(array_list)
                total_clipping = clipped_flag or nested_clipping
                if clipped_flag:
                    if array_repr.endswith("]"):
                        array_repr = array_repr[:-1] + ", ...]"
                return f"np.array({array_repr}, dtype='{dtype}')", total_clipping

            elif isinstance(o, pd.Series):
                data = o
                if clip_size is not None:
                    data = o.iloc[:clip_size]
                    clipped_flag = len(o) > clip_size
                else:
                    clipped_flag = False
                data_list = data.tolist()
                dtype = str(data.dtype)
                name = repr(o.name) if o.name is not None else "None"
                index_list = o.index.tolist()
                if clip_size is not None:
                    index_list = index_list[:clip_size]
                index_repr, nested_clipping = process(index_list)
                data_repr, data_clipping = process(data_list)
                series_repr = (
                    f"pd.Series({data_repr}, "
                    f"index={index_repr}, "
                    f"name={name}, "
                    f"dtype='{dtype}')"
                )
                total_clipping = clipped_flag or nested_clipping or data_clipping
                if total_clipping:
                    series_repr = series_repr.rstrip(")") + ", ...)"
                return series_repr, total_clipping

            elif isinstance(o, pd.DataFrame):
                if clip_size is not None:
                    clipped = o.iloc[:clip_size]
                    clipped_flag = len(o) > clip_size
                else:
                    clipped = o
                    clipped_flag = False
                # Convert DataFrame data to a dictionary with lists
                data = clipped.to_dict(orient="list")
                # Convert index and columns to lists
                index = clipped.index.tolist()
                columns = clipped.columns.tolist()
                data_repr, data_clipping = process(data)
                index_repr, index_clipping = process(index)
                columns_repr, columns_clipping = process(columns)
                df_repr = (
                    f"pd.DataFrame({data_repr}, "
                    f"index={index_repr}, "
                    f"columns={columns_repr})"
                )
                total_clipping = (
                    clipped_flag or data_clipping or index_clipping or columns_clipping
                )
                if total_clipping:
                    df_repr = df_repr.rstrip(")") + ", ...)"
                return df_repr, total_clipping

            # elif isinstance(o, Callable):
            #     # Represent callable by its source code
            #     try:
            #         import inspect

            #         source = inspect.getsource(o).strip()
            #         return repr(source), False
            #     except Exception as e:
            #         raise TypeError(f"Cannot represent callable: {e}")

            else:
                return str(o), False
                # raise TypeError(f"Type {type(o)} for {o} not supported.")

        representation, was_clipped = process(o)
        return representation, was_clipped

    def to_repr(self, clip_size: Optional[int] = None) -> Tuple[str, bool]:
        return self.clipped_repr(type_ops.decode(self.pickle), clip_size=clip_size)

    def to_value(self) -> Any:
        try:
            return type_ops.decode(self.pickle)
        except Exception as e:
            error(f"Error decoding pickle", e)
            return None

    def to_text(self) -> str:
        return self.text

    def to_content_part(self) -> ChatCompletionContentPartTextParam:
        repr, clipped = self.to_repr(10)
        if clipped:
            log("Clipped repr: " + repr)
        return ChatCompletionContentPartTextParam(
            type="text",
            text=repr,
        )


class ResultOutput(BaseModel):
    output_type: OutputType = Field(description="The type of the output")
    data: str = Field(description="The data of the output")

    def to_content_part(self) -> ChatCompletionContentPartParam:
        if self.output_type == OutputType.text:
            return ChatCompletionContentPartTextParam(
                type="text",
                text=self.data,
            )
        elif self.output_type == OutputType.image:
            return ChatCompletionContentPartImageParam(
                type="image_url",
                image_url=ImageURL(
                    url=self.data.replace("data:image/png,", "data:image/png;base64,"),
                    detail="high",
                ),
            )
        else:
            return ChatCompletionContentPartTextParam(
                type="text",
                text="Unknown output type: {self.output_type}: {self.data}",
            )


class NodeResult(BaseModel):
    result: Optional[ResultValue]
    output: Optional[ResultOutput]

    def to_content_parts(self) -> List[ChatCompletionContentPartParam]:
        messages = []
        if self.result and self.result.text.strip() != "None":
            messages.append(self.result.to_content_part())
        if self.output:
            messages.append(self.output.to_content_part())
        return messages

    def _clip(self, text, clip: int | None = None) -> str:
        text = text.splitlines()
        if text and clip and len(text) > clip:
            clipped = text[0:clip] + [f"... ({len(text) - 15} more lines)"]
        else:
            clipped = text
        return "\n".join(clipped)

    def pp_result_text(self, clip: int | None = None) -> str | None:
        if not self.result or not self.result.text:
            return None

        return self._clip(self.result.text, clip)

    def pp_output_text(self, clip: int | None = None) -> str | None:
        if not self.output or not self.output.data:
            return None

        if self.output.output_type == OutputType.text:
            return self._clip(self.output.data, clip)
        else:
            return None

    def output_image(self) -> str | None:
        if not self.output or not self.output.data:
            return None

        if self.output.output_type == OutputType.image:
            return self.output.data.replace("data:image/png;base64,", "data:image/png,")
        else:
            return None

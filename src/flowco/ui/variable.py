from typing import Literal
from pydantic import BaseModel, Field
import streamlit as st
from flowco.dataflow.extended_type import ExtendedType, from_value
import pandas as pd
import numpy as np


class InputVariable(BaseModel):
    name: str = Field(
        description="The name of the variable.",
    )
    type: Literal["int", "float", "str", "bool", "array", "DataFrame"] = Field(
        description="The type of the variable.",
    )
    extended_type: ExtendedType = Field(
        description="The extended type of the variable, based on the value and type.",
    )
    value: str = Field(
        description="An expression to compute the value of the variable.",
    )


def edit_variable(variable: InputVariable):
    basic_types = ["int", "float", "str", "bool", "array", "DataFrame"]
    initial_values = {
        "int": "0",
        "float": "0.0",
        "str": "",
        "bool": "False",
        "array": "[]",
        "DataFrame": "pd.DataFrame(columns=['X', 'Y'])",
    }
    # Edit the variable
    st.write(f"Editing variable {variable.name}")
    variable.name = st.text_input("Name", variable.name)
    new_type = st.selectbox("Type", basic_types, index=basic_types.index(variable.type))
    if new_type != variable.type:
        variable.type = new_type
        variable.value = eval(initial_values[new_type])

    if variable.type == "array":
        variable.value = st.data_editor(
            variable.value, key=variable.name, num_rows="dynamic"
        )
    elif variable.type == "DataFrame":
        variable.value = st.data_editor(
            variable.value, key=variable.name, num_rows="dynamic"
        )
    elif variable.type == "str":
        variable.value = st.text_input("Value", variable.value)
    elif variable.type == "int":
        variable.value = st.number_input("Value", value=int(variable.value))
    elif variable.type == "float":
        variable.value = st.number_input("Value", value=float(variable.value))
    elif variable.type == "bool":
        variable.value = st.checkbox("Value", value=bool(variable.value))

    # Compute the extended type
    variable.extended_type = ExtendedType.from_value(variable.value)

    st.write(variable)


variable = InputVariable(
    name="x", type="int", extended_type=ExtendedType(type="int"), value="0"
)
edit_variable(variable)

from dataclasses import dataclass
from typing import List
from pydantic import BaseModel, Field

from flowco.dataflow.extended_type import ExtendedType


class ParameterPreconditions(BaseModel):
    name: str
    type: ExtendedType | None
    preconditions: List[str]


class FunctionPreconditions(BaseModel):
    parameters: List[ParameterPreconditions] = Field(
        default=[],
        description="The preconditions for the function parameters.",
    )

    def get(self, name: str) -> ParameterPreconditions | None:
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        return None

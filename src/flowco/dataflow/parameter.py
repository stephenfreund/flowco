from pydantic import BaseModel, Field

from flowco.dataflow.extended_type import ExtendedType


class Parameter(BaseModel):
    """
    A parameter to a function.
    """

    name: str = Field(description="The name of the parameter.")
    type: ExtendedType = Field(description="The type of the parameter.")

    def __str__(self) -> str:
        return f"{self.name}: {self.type.to_python_type()}"

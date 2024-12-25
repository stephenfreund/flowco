from typing import List
from pydantic import BaseModel, Field


class FunctionCall(BaseModel):
    """
    A function call.
    """

    node_id: str = Field(
        description="The id of the node that this function call is associated with."
    )
    function_name: str = Field(description="The name of the function to call.")
    arguments: List[str] = Field(description="The arguments to the function call.")
    result: str = Field(
        description="The variable name to store the result of this function call."
    )

    def __str__(self) -> str:
        return f"{self.result} = {self.function_name}({', '.join(self.arguments)})"

from pydantic import BaseModel, Field


class UnitTest(BaseModel):
    description: str = Field(description="A very short description of the test.")
    inputs: str = Field(description="A description of the input values.")
    expected: str = Field(description="The expected result.")

    def __str__(self):
        return f"{self.inputs} -> {self.expected}"

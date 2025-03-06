from pydantic import BaseModel, Field


class UnitTest(BaseModel):
    requirement: str = Field(description="A short description of this unit test.")
    inputs: str = Field(description="A description of the input values.")
    expected: str = Field(description="The expected result.")

    def __str__(self):
        return f"{self.requirement}: {self.inputs} -> {self.expected}"

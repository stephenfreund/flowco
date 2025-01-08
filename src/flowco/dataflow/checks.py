from pydantic import BaseModel, Field
from typing import Any, List, Dict, Literal, Optional, Union

from flowco.page.output import ResultValue


class QuantitiveCheck(BaseModel):
    type: Literal["quantitative"]
    code: List[str] = Field(
        description="Code to run to verify the this requirement is met.  The code is stored as a list of source lines."
    )


class QualitativeCheck(BaseModel):
    type: Literal["qualitative"]
    requirement: str = Field(
        description="A description of the requirement for this test."
    )


class CheckOutcomes(BaseModel):
    outcomes: Dict[str, str | None] = Field(
        default={},
        description="The error message if the test failed, or None if it succeeded.",
    )
    context: Optional[Dict[str, ResultValue]] = Field(
        default={}, description="The context for the checks."
    )


Check = Union[QuantitiveCheck, QualitativeCheck]


class UnitTest(BaseModel):
    description: str = Field(
        description="A short description of the requirement for this test."
    )
    inputs: Dict[str, Any] = Field(
        description="Python expressions to use as the parameter values."
    )
    check: Check = Field(description="The check to perform.")
    outcome: Optional[str] = Field(
        default=None,
        description="The error message if the test failed, or None if it succeeded.",
    )

    def update(self, **kwargs):
        new_test = UnitTest(
            description=kwargs.get("description", self.description),
            inputs=kwargs.get("inputs", self.inputs),
            check=kwargs.get("check", self.check),
            outcome=kwargs.get("outcome", self.outcome),
        )
        if new_test == self:
            return self
        else:
            return new_test


# class UnitTestValue(BaseModel):
#     code: str = Field(description="The code to generate the value.")
#     value: str = Field(description="The value, encoded as a JSON string.")

#     def __init__(self, **data):
#         if "value" not in data:
#             import numpy as np
#             import pandas as pd

#             # Create the context with necessary imports
#             context = {"np": np, "pd": pd}
#             with logger(f"Creating value for {data['code']}"):
#                 data["value"] = types.encode(eval(data["code"], {}, context))
#                 log(f"Value: {data['value']}")

#         super().__init__(**data)

#     def update(self, **kwargs):
#         new_value = UnitTestValue(
#             type=kwargs.get("type", self.type),
#             spec=kwargs.get("spec", self.spec),
#             code=kwargs.get("code", self.code),
#             value=kwargs.get("value", self.value),
#         )
#         if new_value == self:
#             return self
#         else:
#             return new_value

# class UnitTestSpec(BaseModel):
#     description: str = Field(
#         description="A description of the specification for the input."
#     )
#     inputs: Dict[str, str] = Field(description="The inputs to the function, encoded in JSON.")
#     expected: str = Field(description="The expected output of the function, encoded in JSON.")

#     def update(self, **kwargs):
#         new_spec = UnitTestSpec(
#             description=kwargs.get("description", self.description),
#             inputs=kwargs.get("inputs", self.inputs),
#             expected=kwargs.get("expected", self.expected),
#         )
#         if new_spec == self:
#             return self
#         else:
#             return new_spec

#     def eval(self, code: str) -> UnitTestValue:
#         import numpy as np
#         import pandas as pd

#         # Create the context with necessary imports
#         context = {"np": np, "pd": pd}
#         with logger(f"Creating value for {code}"):
#             value = types.encode(eval(code, {}, context))
#             return value

#     def input_value(self, name: str) -> UnitTestValue:
#         return self.eval(self.inputs[name])

#     def expected_value(self) -> UnitTestValue:
#         return self.eval(self.expected)


# class Assert(BaseModel):
#     kind: Literal["assert"] = Field(description="A code-based unit test.")
#     code: str = Field(description="A Python boolean expression for the test.")

#     def expected_code(self) -> str:
#         return f"assert {self.code}"

#     def expected_text(self) -> str:
#         return self.code


# class Inspect(BaseModel):
#     kind: Literal["inspect"] = Field(description="An inspection unit test.")
#     property: str = Field(description="The property that must hold for the output")

#     def expected_code(self) -> str:
#         return f"assert True   # placeholder for inspection test"

#     def expected_text(self) -> str:
#         return f"{self.property}"


# class UnitTest(BaseModel, TestCase):
#     uuid: str = Field(
#         default_factory=lambda: str(uuid.uuid4()),
#         description="A unique identifier for this sanity check",
#     )
#     requirement: str = Field(
#         description="A short description of the requirement this test checks."
#     )

#     # spec: str = Field(
#     #     default=None,
#     #     description="A description of how the inputs and expected test were chosen.",
#     # )

#     inputs: Dict[str, str] = Field(
#         description="Python expressions to use as the parameter values."
#     )

#     expected: Optional[str] = Field(
#         description="A Python boolean expression for the test."
#     )

#     function_name: str = Field(description="The name of the function to test.")
#     result_name: str = Field(description="The name of the result to test.")

#     def get_code(self) -> List[str]:
#         inputs_code = "\n".join([f"{k} = {v}" for k, v in self.inputs.items()])
#         call_code = f"{self.result_name} = {self.function_name}({', '.join(f'{x} = {x}' for x in self.inputs.keys())})"
#         return [
#             f"{inputs_code}\n\n{call_code}\n\nencode_result({self.result_name})",
#             self.expected_code(),
#         ]

#     def expected_code(self) -> str:
#         if self.expected is None:
#             return "# No code"
#         else:
#             return f"assert {self.expected}"

#     def eval(self, code: str) -> Any:
#         import numpy as np
#         import pandas as pd

#         # Create the context with necessary imports
#         context = {"np": np, "pd": pd}
#         value = eval(code, {}, context)
#         return value

#     def eval_input(self, name: str) -> Any:
#         return self.eval(self.inputs[name])

#     def update(self, **kwargs):
#         new_test = UnitTest(
#             uuid=kwargs.get("uuid", self.uuid),
#             requirement=kwargs.get("requirement", self.requirement),
#             inputs=kwargs.get("inputs", self.inputs),
#             expected=kwargs.get("expected", self.expected),
#             function_name=kwargs.get("function_name", self.function_name),
#             result_name=kwargs.get("result_name", self.result_name),
#         )
#         if new_test == self:
#             return self
#         else:
#             return new_test


# class SanityCheckResult(BaseModel):
#     outcome: Optional[str] = Field(
#         default=None,
#         description="The error message if the test failed, or None if it succeeded.",
#     )


# class UnitTestResult(BaseModel):
#     outcome: Optional[str] = Field(
#         default=None,
#         description="The error message if the test failed, or None if it succeeded.",
#     )
#     computed_results: Optional[ResultValue] = Field(
#         default=None, description="The actual value returned by the function."
#     )

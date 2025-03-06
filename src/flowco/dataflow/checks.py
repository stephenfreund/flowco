from pydantic import BaseModel, Field
from typing import Any, List, Dict, Literal, Optional, Union

from flowco.page.output import ResultValue


class QuantitiveCheck(BaseModel):
    type: Literal["quantitative"]
    code: List[str] = Field(
        description="Code to run to verify the this requirement is met.  The code is stored as a list of source lines."
    )
    warning: Optional[str] = Field(
        default=None,
        description="A warning message if the check is not guaranteed by the requirements.",
    )


class QualitativeCheck(BaseModel):
    type: Literal["qualitative"]
    requirement: str = Field(
        description="A description of the requirement for this test."
    )
    warning: Optional[str] = Field(
        default=None,
        description="A warning message if the check is not guaranteed by the requirements.",
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

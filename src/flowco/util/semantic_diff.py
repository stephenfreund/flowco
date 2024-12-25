from dataclasses import dataclass
import json
from typing import Dict, Optional
from typing import Any, List
from pydantic import BaseModel, create_model, Field

from flowco.assistant.assistant import Assistant
from flowco.util.output import log


# A changes to A'.  What's different in each component?
def semantic_diff(
    old: BaseModel | Dict[str, Any], new: BaseModel | Dict[str, Any]
) -> Dict[str, List[str]]:
    old_str = (
        old.model_dump_json(indent=2)
        if isinstance(old, BaseModel)
        else json.dumps(old, indent=2)
    )
    new_str = (
        new.model_dump_json(indent=2)
        if isinstance(new, BaseModel)
        else json.dumps(new, indent=2)
    )

    include = old.model_fields_set if isinstance(old, BaseModel) else old.keys()

    Diff = create_model(
        "Diff",
        **{
            f"{k}-differences": (
                List[str],
                Field(description="Semantic differences in the `{k}` components."),
            )
            for k in include
        },
        summary=(
            Optional[str],
            Field(
                description="A short summary of the semantic differences.  Keep it under 10-12 words.  Leave blank if no changes"
            ),
        ),
    )

    assistant = Assistant("semantic-diff", old=old_str, new=new_str)
    completion: BaseModel = assistant.model_completion(Diff)
    result = completion.model_dump()
    log("Semantic diff:", result)
    return result


def semantic_diff_strings(key: str, old: str, new: str) -> Dict[str, Any]:
    Diff = create_model(
        "Diff",
        **{
            "differences": (
                List[str],
                Field(description="Semantic differences in the `{key}` values."),
            )
        },
        summary=(
            Optional[str],
            Field(
                description="A short summary of the semantic differences.  Keep it under 10-12 words.  Leave blank if no changes"
            ),
        ),
    )

    assistant = Assistant("semantic-diff-string", key=key, old=old, new=new)
    completion: BaseModel = assistant.model_completion(Diff)
    result = completion.model_dump()
    log("Semantic diff:", result)
    return result

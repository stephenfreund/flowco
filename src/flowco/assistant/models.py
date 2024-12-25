from typing import List
import openai


def supported_models() -> List[str]:
    return ["o1", "gpt-4o", "gpt-4o-2024-11-20", "gpt-4o-mini", "gpt-3.5-turbo"]
    # return [model.id for model in openai.models.list()]

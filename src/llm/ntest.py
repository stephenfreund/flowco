import random
from typing import Annotated
from llm.assistant import Assistant, ToolCallResult
from pydantic import BaseModel

from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)


class MyModel(BaseModel):
    city: str
    country: str


assistant = Assistant(model="gpt-4o", temperature=1)
assistant.add_text("user", "Give me a city in Europe.")
response = assistant.model_n_completions(MyModel, n=5)
print(response)

print("======")

import random
from typing import Annotated
from flowco.llm.assistant import Assistant, ToolCallResult
from pydantic import BaseModel

from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam,
    ImageURL,
)

print("======")

assistant = Assistant(model="gpt-4o")
assistant.add_text("user", "What is the capital of France?")
response = assistant.completion()
print(response)

print("---")

assistant.add_text("user", "What is the capital of Paraguay?  Repeat it 20 times.")
for t in assistant.stream():
    print(t, end="", flush=True)
print("\n")

print("======")


class MyModel(BaseModel):
    city: str
    country: str


assistant = Assistant(model="gpt-4o")
assistant.add_text("user", "Windy city in the US of A.")
response = assistant.model_completion(MyModel)
print(response)

print("======")


def random_number(x: int) -> ToolCallResult:
    """
    Return a random number between 0 and x.

    Args:
        x: The maximum value of the random number.
    """
    return ToolCallResult(
        user_message="Rolling",
        content=ChatCompletionContentPartTextParam(
            type="text", text=str(random.randint(0, x))
        ),
    )


assistant = Assistant(model="gpt-4o", functions=[random_number])
assistant.add_text("user", "Give me three random numbers between 0 and 10.")
text = assistant.completion()
print(text)

assistant.add_text("user", "Give me 5 random numbers between -100 and 100.")
for t in assistant.stream():
    print(t, end="", flush=True)
print("\n")

print("======")


class City(BaseModel):
    city: str
    country: str


class Weather(BaseModel):
    temperature: int
    weather: str


def get_temperature(x: Annotated[City, "The city/country you want"]) -> ToolCallResult:
    """
    Return the temperature in the city.
    """
    return ToolCallResult(
        user_message="getting temp",
        content=ChatCompletionContentPartTextParam(type="text", text="20"),
    )


def get_weather(x: Annotated[City, "The city/country you want"]) -> ToolCallResult:
    """
    Return the weather in the city.
    """
    return ToolCallResult(
        user_message="getting weather",
        content=ChatCompletionContentPartTextParam(type="text", text="Sunny"),
    )


assistant = Assistant(model="gpt-4o", functions=[get_temperature, get_weather])
assistant.add_text("user", "What is the temperature in Paris, France?")
text = assistant.completion()
print(text)

assistant.add_text("user", "What is the weather and temperature in Chicago?")
print(assistant.model_completion(Weather))

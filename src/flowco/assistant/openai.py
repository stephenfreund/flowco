from dataclasses import dataclass
import json
from typing import Any, Dict, Iterable, List, TypeVar
from litellm import ChatCompletionMessageToolCall
from openai import OpenAI
from pydantic import BaseModel

from flowco.util.config import config
from flowco.util.costs import add_cost
from flowco.util.output import debug


@dataclass
class Pricing:
    input_tokens: float
    output_tokens: float


openai_models = {
    "o1": Pricing(input_tokens=15.00 / 1_000_000, output_tokens=60.00 / 1_000_000),
    "o1-mini": Pricing(input_tokens=3.00 / 1_000_000, output_tokens=12.00 / 1_000_000),
    #
    "gpt-4o": Pricing(input_tokens=2.50 / 1_000_000, output_tokens=10.00 / 1_000_000),
    "gpt-4o-2024-11-20": Pricing(
        input_tokens=2.50 / 1_000_000, output_tokens=10.00 / 1_000_000
    ),
    "gpt-4o-mini": Pricing(
        input_tokens=0.150 / 1_000_000, output_tokens=0.600 / 1_000_000
    ),
}


class OpenAIAssistant:

    @staticmethod
    def supported_models() -> List[str]:
        return list(openai_models.keys())

    def __init__(
        self,
        model: str,
        interactive: bool,
        system_prompt_key: str | list[str],
        **system_prompt_substitutions,
    ):
        assert model in openai_models, f"Invalid model: {model}"
        self.model = model
        self.interactive = interactive
        self.conversation: List[Dict[str, Any]] = []
        self.interactive: bool = interactive
        self.question_queue: List["ChatCompletionMessageToolCall"] = []
        self.done = False

        if isinstance(system_prompt_key, str):
            self.add_message(
                "system",
                config.get_prompt(system_prompt_key, **system_prompt_substitutions),
            )
        else:
            for key in system_prompt_key:
                self.add_message(
                    "system",
                    config.get_prompt(key, **system_prompt_substitutions),
                )

    def add_json_object(self, description: str, json_object: dict[str, any]) -> None:
        self.add_message(
            "user",
            [
                {"type": "text", "text": description},
                {"type": "text", "text": json.dumps(json_object, indent=2)},
            ],
        )

    def add_prompt_by_key(self, key: str, **prompt_substitutions) -> None:
        self.add_message("system", config.get_prompt(key, **prompt_substitutions))

    def add_message(self, role: str, content: str | list[dict[str, Any]]):
        assert role in ["user", "assistant", "system", "tool"], f"Invalid role: {role}"

        debug(
            "Add Message: " + json.dumps({"role": role, "content": content}, indent=2)
        )

        if isinstance(content, str):
            self.conversation.append({"role": role, "content": content})
        else:
            self.conversation.append({"role": role, "content": content})

    def _cost(self, completion):
        pricing = openai_models[self.model]
        return (
            completion.usage.completion_tokens * pricing.output_tokens
            + completion.usage.prompt_tokens * pricing.input_tokens
        )

    T = TypeVar("T", bound=BaseModel)

    def completion(self, t: type[T], prompt: str | None = None) -> T | None:

        assert self.question_queue == [], "There are still questions in the queue."

        if prompt is not None:
            self.conversation.append({"role": "user", "content": prompt})

        client = OpenAI()

        completion = self._completion(t, client)

        response_message = completion.choices[0].message
        self.add_message("assistant", response_message.parsed.model_dump_json(indent=2))

        add_cost(self._cost(completion))
        if completion.choices[0].finish_reason == "tool_calls":
            assert self.interactive, "Tool calls are only allowed in interactive mode."
            self._add_questions_to_queue(response_message)
            return None
        else:
            self.done = True
            return response_message.parsed

    T = TypeVar("T", bound=BaseModel)

    def stream(self, t: type[T], prompt: str | None = None) -> Iterable[T]:
        assert self.question_queue == [], "There are still questions in the queue."

        if prompt is not None:
            self.conversation.append({"role": "user", "content": prompt})

        client = OpenAI()

        completion = self._completion(t, client, stream=True)

        with completion as stream:
            for response_message in stream:
                if response_message.type == "chunk":
                    if response_message.snapshot.choices[0].message.parsed != None:
                        yield response_message.snapshot.choices[0].message.parsed

        content = response_message.snapshot.choices[0].message.content

        completion = stream.get_final_completion()
        add_cost(self._cost(completion))

        final_response_message = completion.choices[0].message

        if completion.choices[0].finish_reason == "tool_calls":
            assert self.interactive, "Tool calls are only allowed in interactive mode."
            self._add_questions_to_queue(final_response_message)
            self.conversation.append(final_response_message)
        else:
            self.done = True
            self.add_message("assistant", content)

    def _completion(self, t: type[T], client: OpenAI, stream=False):
        self.done = False

        args = {
            "model": self.model,
            "messages": self.conversation,
            "response_format": t,
        }
        if self.interactive:
            args["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "answer_question",
                        "description": "Ask the user a question and get the answer.",
                        "parameters": {
                            "type": "object",
                            "required": [
                                "phase",
                                "question",
                            ],
                            "properties": {
                                "phase": {
                                    "type": "string",
                                    "description": "The phase of the question.  Either 'Requirements', 'Algorithm', or 'Code'.",
                                },
                                "question": {
                                    "type": "string",
                                    "description": "The question to be answered.",
                                },
                            },
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                }
            ]

        if stream:
            completion = client.beta.chat.completions.stream(
                **args, stream_options={"include_usage": True}
            )
        else:
            completion = client.beta.chat.completions.parse(**args)

        return completion

    def _add_questions_to_queue(self, response_message):
        tool_calls = response_message.tool_calls
        for tool_call in tool_calls:
            if tool_call.function.name == "answer_question":
                self.question_queue.append(tool_call)

    def peek_question(self) -> str | None:
        if len(self.question_queue) == 0:
            return None
        else:
            arguments = self.question_queue[0].function.parsed_arguments
            return f"**{arguments['phase']}**: {arguments['question']}"

    def questions(self) -> List[str]:
        return [
            f"**{tool_call.function.parsed_arguments['phase']}**: {tool_call.function.parsed_arguments['question']}"
            for tool_call in self.question_queue
        ]

    def has_questions(self) -> bool:
        return len(self.question_queue) > 0

    def answer_question(self, answer: str):
        tool_call = self.question_queue.pop(0)
        response = {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": tool_call.function.name,
            "content": answer,
        }
        self.conversation.append(response)

    def is_done(self) -> bool:
        return self.done

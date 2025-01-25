import pprint
import textwrap
import traceback
from flowco.assistant.base import AssistantBase
from flowco.assistant.trim import sandwich_tokens, trim_messages
from flowco.util.output import error, log, warn
from flowco.util.costs import add_cost
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Tuple,
    TypeVar,
)

import litellm
import tenacity
from pydantic import BaseModel

import json
from flowco.util.config import config
from flowco.util.output import log, logger, warn

import uuid


class StreamingAssistant(AssistantBase):
    def str_completion(self, model: str = config.model) -> Iterator[str]:
        return self._str_stream_completion(model)

    def _str_stream_completion(self, model: str) -> Iterator[str]:
        with logger("Streaming Completion"):
            stream = litellm.completion(
                model=config.model,
                messages=self.messages,
                stream=True,
                temperature=0 if config.zero_temp else None,
            )

            for chunk in stream:
                yield chunk.choices[0].delta.content  # type: ignore

    T = TypeVar("T", bound=BaseModel)

    def model_completion(
        self, response_model: type[T], model: str = config.model
    ) -> Iterator[T]:
        return self._model_stream_completion(response_model, model)

    def _model_stream_completion(
        self, response_model: type[T], model: str
    ) -> Iterator[T]:
        with logger("Streaming Completion"):
            self.add_message("system", "Respond with JSON.")
            stream = self.instructor_client.chat.completions.create_partial(
                # model=config.model,
                messages=self.messages,
                response_model=response_model,
                stream=True,
                response_format={"type": "json_object"},
                max_retries=tenacity.Retrying(
                    stop=tenacity.stop_after_attempt(3),
                    after=lambda e: log(f"Bad response: {e}"),
                ),  # type: ignore
                temperature=0 if config.zero_temp else None,
            )
            for response in stream:
                yield response


class StreamingTextAssistant(AssistantBase, Iterable[str]):
    def __init__(
        self, system_prompt_key: str | List[str], **system_prompt_substitutions
    ) -> None:
        super().__init__(system_prompt_key, **system_prompt_substitutions)

    def __iter__(self) -> Iterator[str]:
        stream = litellm.completion(
            model=config.model,
            messages=self.messages,
            stream=True,
            response_format={"type": "text"},
            temperature=0 if config.zero_temp else None,
        )
        self.chunks: List[Any] = []
        for chunk in stream:
            self.chunks.append(chunk)
            if chunk.choices[0].delta.content != None:  # type: ignore
                yield chunk.choices[0].delta.content  # type: ignore
        add_cost(self.cost())

    def cost(self):
        response = litellm.stream_chunk_builder(self.chunks, messages=self.messages)
        cost = litellm.completion_cost(response)  # type: ignore
        log(f"Cost: {cost:.2f} USD")
        return cost


class StreamingAssistantWithFunctionCalls(AssistantBase):
    def __init__(
        self,
        functions=[],
        system_prompt_key: str | List[str] = [],
        **system_prompt_substitutions,
    ) -> None:
        super().__init__(system_prompt_key, **system_prompt_substitutions)
        self._functions = {}
        for f in functions:
            self._add_function(f)
        self.model = config.model
        self.image_cache = {}

    def _add_function(self, function):
        """
        Add a new function to the list of function tools.
        The function should have the necessary json spec as its docstring
        """
        schema = json.loads(function.__doc__)
        assert "name" in schema, "Bad JSON in docstring for function tool."
        self._functions[schema["name"]] = {"function": function, "schema": schema}

    def set_functions(self, functions):
        self._functions = {}
        for f in functions:
            self._add_function(f)

    def _make_call(self, tool_call) -> Tuple[str, str | Dict[str, Any] | None]:
        name = tool_call.function.name
        try:
            with logger(f"Calling function {name}"):
                args = json.loads(tool_call.function.arguments)
                log(f"Arguments: {pprint.pformat(args)}")
                function = self._functions[name]
                user_response, result = function["function"](**args)
                log(f"User response: {user_response}")
                log(f"Function call result: {result}")
        except KeyboardInterrupt as e:
            warn(f"Keyboard interrupt in function call: {e}")
            raise e
        except Exception as e:
            # likely to be an exception from the code we ran, not a bug...
            result = f"Exception in function call: {e}\n{traceback.format_exc()}"
            warn(e)
            user_response = "An error occurred while processing the tool call."

        return user_response, result

    def str_completion(self) -> Iterator[str]:
        cost = 0

        while True:
            # litellm.stream_chunk_builder is broken for new GPT models
            # that have content before calls, so...

            # stream the response, collecting the tool_call parts separately
            # from the content
            self._trim_conversation()
            stream = litellm.completion(
                model=self.model,
                messages=self.messages,
                tools=[
                    {"type": "function", "function": f["schema"]}
                    for f in self._functions.values()
                ],
                stream=True,
                response_format={"type": "text"},
                max_completion_tokens=8192 * 2,
                temperature=0 if config.zero_temp else None,
            )

            chunks = []
            tool_chunks = []
            for chunk in stream:
                chunks.append(chunk)
                if chunk.choices[0].delta.content == None:
                    tool_chunks.append(chunk)
                else:
                    yield chunk.choices[0].delta.content

            # then compute for the part that litellm gives back.
            completion = litellm.stream_chunk_builder(chunks, messages=self.messages)
            cost += litellm.completion_cost(completion)

            # add content to conversation, but if there is no content, then the message
            # has only tool calls, and skip this step
            response_message = completion.choices[0].message
            # if response_message.content != None:
            self.messages.append(response_message.json())

            if completion.choices[0].finish_reason == "tool_calls":
                # create a message with just the tool calls, append that to the conversation, and generate the responses.
                tool_completion = litellm.stream_chunk_builder(
                    tool_chunks, self.messages
                )

                # this part wasn't counted above...
                cost += litellm.completion_cost(tool_completion)

                tool_message = tool_completion.choices[0].message

                tool_json = tool_message.json()

                # patch for litellm sometimes putting index fields in the tool calls it constructs
                # in stream_chunk_builder.  gpt-4-turbo-2024-04-09 can't handle those index fields, so
                # just remove them for the moment.
                for tool_call in tool_json.get("tool_calls", []):
                    _ = tool_call.pop("index", None)

                tool_json["role"] = "assistant"
                # self.messages.append(tool_json)
                yield from self._add_function_results_to_conversation(tool_message)
            else:
                break

        add_cost(cost)

    T = TypeVar("T", bound=BaseModel)

    def model_completion(self, response_model: type[T]) -> T:
        response, completion = (
            self.instructor_client.chat.completions.create_with_completion(
                model=config.model,
                messages=self.messages,
                response_model=response_model,
                response_format={"type": "json_object"},
                max_retries=tenacity.Retrying(
                    stop=tenacity.stop_after_attempt(3),
                    after=lambda e: log(
                        f"Bad response, retrying up to limit:\n{textwrap.indent(str(e), prefix='    ')}"
                    ),
                ),  # type: ignore
                temperature=0 if config.zero_temp else None,
            )
        )
        self.add_message("assistant", response.model_dump_json(indent=2))  # type: ignore
        # cost = litellm.completion_cost(completion)  # type: ignore
        cost = 0
        return response

    def _trim_conversation(self):
        old_len = litellm.utils.token_counter(self.model, messages=self.messages)
        log(f"Conversation has {old_len} tokens.")
        if old_len > 50000:
            for m in self.messages:
                print(litellm.utils.token_counter(self.model, messages=[m]), m)
        self.messages = trim_messages(self.messages, self.model)

        new_len = litellm.utils.token_counter(self.model, messages=self.messages)
        if old_len != new_len:
            warn(f"Trimming conversation from {old_len} to {new_len} tokens.")

    def _add_function_results_to_conversation(self, response_message):

        def make_response(tool_call, content):
            response = {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": content,
            }
            self.messages.append(response)

        response_message["role"] = "assistant"
        tool_calls = response_message.tool_calls
        try:
            for tool_call in tool_calls:
                user_response, function_response = self._make_call(tool_call)
                yield f"\n\n{user_response}\n\n"
                if function_response is None:
                    make_response(tool_call, user_response)
                elif isinstance(function_response, str):
                    make_response(tool_call, function_response)
                elif function_response["type"] == "text":
                    content = function_response["text"]
                    content = sandwich_tokens(content, self.model, 8192 * 2, 0.5)
                    make_response(tool_call, content)
                elif function_response["type"] == "image_url":
                    id = str(uuid.uuid4())[:6]
                    key = f"result-{id}"
                    self.image_cache[key] = function_response["image_url"]["url"]
                    content = f"The result is the image '{key}.png'."
                    make_response(tool_call, content)
                    self.add_message(
                        role="user",
                        content=[
                            {
                                "type": "text",
                                "text": f"Here is image '{key}.png'.",
                            },
                            {
                                "type": "image_url",
                                "image_url": function_response["image_url"]["url"],
                            },
                        ],
                    )
                    yield f"![code_result]({key}.png)"
                else:
                    content = f"Unknown response type: {function_response['type']}"
                    content = sandwich_tokens(content, self.model, 8192 * 2, 0.5)
                    make_response(tool_call, content)

        except Exception as e:
            error(f"An exception occurred while processing tool calls", e)

    def replace_placeholders_with_base64_images(self, markdown: str) -> str:
        for key, image_url in self.image_cache.items():
            markdown = markdown.replace(
                f"![code_result]({key}.png)", f'\n\n<img src="{image_url}"/>\n\n'
            )
        return markdown


if __name__ == "__main__":

    def get_current_temperature(location: str, unit: str) -> Tuple[str, str]:
        """
        {
            "name": "get_current_temperature",
            "description": "Get the current temperature for a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["Celsius", "Fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the user's location."
                    }
                },
            "required": ["location", "unit"]
            }
        }
        """
        return "beep", "Rainy"

    def python_eval(code: str) -> Tuple[str, str]:
        """
        {
            "name": "python_eval",
            "description": "Exec python code.  Returns the values of all locals written to in the code.  You may assume numpy, scipy, and pandas are available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to evaluate"
                    }
                },
                "required": ["code"]
            }
        }
        """
        from IPython.core.interactiveshell import InteractiveShell

        def execute_code(code, globals_dict=None):
            """
            Executes the given Python code within an IPython kernel,
            defines global variables, and returns the result of the last expression.

            :param code: The Python code to execute as a string.
            :param globals_dict: A dictionary of global variables to define before execution.
            :return: The result of the last expression in the executed code.
            """
            # Create or get an existing InteractiveShell instance
            shell = InteractiveShell.instance()

            # Update the shell's user namespace with provided globals
            if globals_dict:
                shell.user_ns.update(globals_dict)

            # Execute the code
            result = shell.run_cell(code)

            # Return the result of the last expression
            return result.result

        return "code", str(execute_code(code))

    assistant = StreamingAssistantWithFunctionCalls(
        [python_eval], "system-prompt", imports=""
    )
    assistant.add_message(
        "user",
        "I have an array with the numbers 1 to 100 in it.  Bootstrap a sample of 10 numbers from it.",
    )
    for response in assistant:
        print(response, end="")
    print()

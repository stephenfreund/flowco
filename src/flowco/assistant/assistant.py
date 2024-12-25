from flowco.assistant.base import AssistantBase
from flowco.assistant.trim import sandwich_tokens
from flowco.util.output import error, log, warn, logger
from flowco.util.costs import add_cost
import textwrap
from typing import (
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


class Assistant(AssistantBase):
    def str_completion(self, model: str = config.model) -> str:
        response, cost = self._str_batch_completion(model)
        log(f"Cost: {cost:.2f} USD")
        add_cost(cost)
        return response

    def _str_batch_completion(self, model: str) -> tuple[str, float]:
        with logger("Completion"):
            response = litellm.completion(
                model=model,
                messages=self.messages,
                temperature=0 if config.zero_temp else None,
            )
            message: str = response.choices[0].message.content  # type: ignore
            self.add_message("assistant", message)
            cost = litellm.completion_cost(response)  # type: ignore
            return message, cost

    T = TypeVar("T", bound=BaseModel)

    def model_completion(self, response_model: type[T], model: str = config.model) -> T:
        response, cost = self._model_batch_completion(response_model, model)
        add_cost(cost)
        return response

    def _model_batch_completion(
        self, response_model: type[T], model: str
    ) -> tuple[T, float]:
        with logger("Completion"):
            self.add_message("system", "Respond with JSON.")
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
            return response, cost


class StreamingAssistant(AssistantBase):
    def str_completion(self, model: str = config.model) -> Iterator[str]:
        return self._str_stream_completion(model)

    def _str_stream_completion(self, model: str) -> Iterator[str]:
        with logger("Streaming Completion"):
            stream = litellm.completion(
                model=model,
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
                model=model,
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


class StreamingAssistantWithFunctionCalls(AssistantBase, Iterable[str]):
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

    def _add_function(self, function):
        """
        Add a new function to the list of function tools.
        The function should have the necessary json spec as its docstring
        """
        schema = json.loads(function.__doc__)
        assert "name" in schema, "Bad JSON in docstring for function tool."
        self._functions[schema["name"]] = {"function": function, "schema": schema}

    def _make_call(self, tool_call) -> Tuple[str, str | None]:
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
            function = self._functions[name]
            user_message, result = function["function"](**args)
            # result = remove_non_printable_chars(strip_ansi(result).expandtabs())
            log(name, args, "->", (user_message, result))
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            # likely to be an exception from the code we ran, not a bug...
            result = f"Exception in function call: {e}"
            warn(result)
        return result

    def stream(self, model: str = config.model) -> Iterator[str]:
        cost = 0

        while True:
            self._trim_conversation()
            print(json.dumps(self.messages, indent=2))
            stream = litellm.completion(
                model=model,
                messages=self.messages,
                tools=[
                    {"type": "function", "function": f["schema"]}
                    for f in self._functions.values()
                ],
                stream=True,
                response_format={"type": "text"},
                temperature=0 if config.zero_temp else None,
            )

            # litellm.stream_chunk_builder is broken for new GPT models
            # that have content before calls, so...

            # stream the response, collecting the tool_call parts separately
            # from the content
            chunks = []
            tool_chunks = []
            for chunk in stream:
                chunks.append(chunk)
                if chunk.choices[0].delta.content == None:
                    tool_chunks.append(chunk)
                else:
                    yield (chunk.choices[0].delta.content)

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

    def _trim_conversation(self):
        pass
        # old_len = litellm.token_counter(self.model, messages=self.messages)

        # self.messages = trim_messages(self.messages, self.model)

        # new_len = litellm.token_counter(self.model, messages=self.messages)
        # if old_len != new_len:
        #     warn(f"Trimming conversation from {old_len} to {new_len} tokens.")

    def _add_function_results_to_conversation(self, response_message):
        response_message["role"] = "assistant"
        tool_calls = response_message.tool_calls
        try:
            for tool_call in tool_calls:
                user_message, function_response = self._make_call(tool_call)
                function_response = sandwich_tokens(
                    function_response, self.model, 8192, 0.5
                )
                response = {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": function_response,
                }
                yield f"\n{user_message}\n"
                self.messages.append(response)
        except Exception as e:
            error(f"An exception occurred while processing tool calls: {e}")

    def cost(self):
        response = litellm.stream_chunk_builder(self.chunks, messages=self.messages)
        cost = litellm.completion_cost(response)  # type: ignore
        log(f"Cost: {cost:.2f} USD")
        return cost


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

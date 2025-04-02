from copy import deepcopy
from dataclasses import dataclass
import inspect
import os
import subprocess
import threading
from time import sleep
from typing import Any, Callable, Dict, Iterator, List, Literal, Type, TypeVar


import openai
from flowco.util.output import logger
from llm.message_format import process_chat_message
from llm.models import Model, get_model
from pydantic import BaseModel

from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionContentPartParam,
    ChatCompletionMessageToolCall,
    ParsedChoice,
    ParsedChatCompletionMessage,
    ParsedFunctionToolCall,
)


from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCall,
)

from openai.types.chat.chat_completion_prediction_content_param import (
    ChatCompletionPredictionContentParam,
)

from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam,
    ImageURL,
)
from openai.types.chat.chat_completion_content_part_input_audio_param import (
    ChatCompletionContentPartInputAudioParam,
    InputAudio,
)

from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_developer_message_param import (
    ChatCompletionDeveloperMessageParam,
)


import json

from typing import Annotated, Callable, get_args, get_origin
from inspect import signature, Parameter
import openai
from pydantic import BaseModel, Field, create_model

from openai.types.chat import ChatCompletionToolParam


def to_base_model(func: Callable) -> Type[BaseModel]:
    """
    Create a Pydantic BaseModel class representing the parameters of `func`.

    For parameters whose type is Annotated with a string, that string is used as the field description.
    """
    from flowco.dataflow.extended_type import ExtendedType

    sig = signature(func)
    fields = {}

    for name, param in sig.parameters.items():
        # Start with the parameter's annotation.
        ann = param.annotation

        field_description = None

        # If the annotation is an Annotated type, extract the inner type and description.
        if get_origin(ann) is Annotated:
            inner_type, *extras = get_args(ann)
            ann = inner_type
            # Look for the first string in the extras to use as the description.
            for extra in extras:
                if isinstance(extra, str):
                    field_description = extra
                    break

        # Determine the default value: if the parameter has no default, mark it as required.
        default = param.default if param.default is not Parameter.empty else ...

        # If we have a description, use Field to attach it.
        if field_description:
            fields[name] = (ann, Field(default, description=field_description))
        else:
            fields[name] = (ann, default)

    # Create a model whose name is based on the function name.
    model_name = func.__name__
    return create_model(model_name, **fields, __module__=func.__module__)


def function_to_schema(func: Callable) -> ChatCompletionToolParam:
    """
    Create a JSON schema representing the parameters of `func`.

    For parameters whose type is Annotated with a string, that string is used as the field description.
    """
    model = to_base_model(func)
    return openai.pydantic_function_tool(model)


class ToolCallResult(BaseModel):
    user_message: str
    content: ChatCompletionContentPartParam | None = None


class ToolDefinition(BaseModel):
    name: str
    description: str
    function_schema: ChatCompletionToolParam
    function: Callable[..., ToolCallResult]


T = TypeVar("T", bound=BaseModel)


@dataclass
class NthCompletion:
    t: T
    assistant: "Assistant"


class AssistantError(Exception):
    messages = {
        openai.APIConnectionError: "Issue connecting to the LLM.  Try again.",
        openai.APITimeoutError: "LLM request timed out.",
        openai.AuthenticationError: "Your API key or token was invalid, expired, or revoked.  Add a valid API key under Settings.  You can find your API key at https://platform.openai.com/account/api-keys.",
        openai.BadRequestError: "Internal error.  Bad Request.  Please click the 'Report Bug' button.",
        openai.ConflictError: "Internal error.  Conflict Error.  Please click the 'Report Bug' button.",
        openai.InternalServerError: "LLM Server error.  Try again.",
        openai.NotFoundError: "Internal Error.  Requested resource does not exist.  Please click the 'Report Bug' button.",
        openai.PermissionDeniedError: "Internal Error.  You don't have access to the requested resource. Please click the 'Report Bug' button.",
        openai.RateLimitError: "You have hit your LLM rate limit.",
        openai.UnprocessableEntityError: "The request was well-formed but was unable to be followed due to semantic errors.",
    }

    def __init__(self, message: str) -> None:
        self.message = message

    @staticmethod
    def from_openai_error(e: openai.OpenAIError) -> "AssistantError":
        return AssistantError(
            AssistantError.messages.get(type(e), "An unknown error occurred.")
        )


class AssistantLogger:
    def log(self, message: str) -> None:
        print(message)

    def warn(self, message: str) -> None:
        print(f"Warning: {message}")

    def error(self, message: str) -> None:
        print(f"Error: {message}")

    def debug(self, message: str) -> None:
        print(f"Debug: {message}")

    def charge_cost(self, cost: float) -> None:
        print(f"Charged cost: {cost}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


proxy_url = "http://0.0.0.0"
proxy_port = 4001


class Assistant:

    def __init__(
        self,
        model: str,
        api_key: str,
        functions: List[Callable[..., ToolCallResult]] = [],
        logger: AssistantLogger = AssistantLogger(),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.model = get_model(model)
        self.api_key = api_key
        self.messages: List[ChatCompletionMessageParam] = []
        self.set_functions(functions)
        self.cached_images = {}
        self.logger = logger
        self.temperature = temperature
        self.max_tokens = max_tokens

        if self.model.use_proxy:
            self.logger.warn(
                f"Using proxy model {self.model.name}. This model is not suitable for production use."
            )
            self.start_proxy(api_key=api_key)
            # shouldn't need a real key here...
            self.client = openai.Client(
                base_url=f"{proxy_url}:{proxy_port}", api_key=api_key
            )
        else:
            self.client = openai.Client(api_key=api_key)

    proxy_lock = threading.Lock()

    def start_proxy(self, api_key: str) -> None:
        with Assistant.proxy_lock:
            import socket

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("localhost", proxy_port)) != 0:
                    self.logger.log("Starting proxy model")
                    config_file = os.path.dirname(__file__) + "/litellm-config.yaml"
                    env = os.environ.copy()
                    env[self.model.api_key_name] = api_key
                    p = subprocess.Popen(
                        f"litellm --port {proxy_port} --config {config_file}".split(
                            " "
                        ),
                        env=env,
                    )
                    sleep(3)  # wait for the proxy to start

    def append(self, message: ChatCompletionMessageParam) -> None:
        self.messages.append(message)
        self.logger.log(json.dumps(process_chat_message(message), indent=2))

    def set_functions(self, functions: List[Callable[..., ToolCallResult]]) -> None:
        defs = [self.function_def(f) for f in functions]
        self.functions = {def_.name: def_ for def_ in defs}

    def function_def(self, function: Callable[..., ToolCallResult]) -> ToolDefinition:
        schema = function_to_schema(function)
        return ToolDefinition(
            name=schema["function"]["name"],
            description=inspect.getdoc(function) or "",
            function_schema=schema,
            function=function,
        )

    def add_text(self, role: str, text: str) -> None:
        self.add_content_parts(
            role, ChatCompletionContentPartTextParam(type="text", text=text)
        )

    def add_image(self, role: str, url: str) -> None:
        self.add_content_parts(
            role,
            ChatCompletionContentPartImageParam(
                type="image_url", image_url=ImageURL(url=url, detail="high")
            ),
        )

    def add_json(self, role: str, content: Dict[str, Any]) -> None:
        text = json.dumps(content, indent=2)
        self.add_content_parts(
            role,
            ChatCompletionContentPartTextParam(type="text", text=text),
        )

    def add_audio(
        self, role: str, base64_audio: str, format: Literal["wav", "mp3"]
    ) -> None:
        self.add_content_parts(
            role,
            ChatCompletionContentPartInputAudioParam(
                type="input_audio",
                input_audio=InputAudio(data=base64_audio, format=format),
            ),
        )

    def add_list_of_text_or_json(
        self, role: str, items: List[str | Dict[str, Any]]
    ) -> None:
        for item in items:
            if isinstance(item, str):
                self.add_text(role, item)
            else:
                self.add_json(role, item)

    def make_call(
        self, tool_call: ChatCompletionMessageToolCall | ParsedFunctionToolCall
    ) -> str:
        call_text = ""
        call_id = tool_call.id
        function = tool_call.function
        function_name = function.name
        try:
            args = json.loads(function.arguments)
            function_def = self.functions[function_name]

            self.logger.log(f"Tool call: {function_name}")

            result = function_def.function(**args)
        except Exception as e:
            self.logger.error(f"Error in tool call {function_name}: {e}")
            result = ToolCallResult(user_message=f"An error occurred: {e}")

        self.logger.log(f"Tool result: {result}")

        if result.user_message:
            call_text += result.user_message

        user_parts = []
        content = result.content
        if content:
            content_type = content["type"]
            if content_type == "text":
                response = str(content["text"])  # type: ignore
            elif content_type == "image_url":
                n = len(self.messages)
                key = f"image{n}"
                image_url = content["image_url"]["url"]  # type: ignore
                self.cached_images[key] = image_url
                user_parts = [
                    ChatCompletionContentPartTextParam(
                        type="text", text=f"Here is the image `{key}.png`"
                    ),
                    ChatCompletionContentPartImageParam(
                        type="image_url",
                        image_url=ImageURL(url=image_url, detail="high"),
                    ),
                ]

                response = f"The result is the image `{key}.png`."
                call_text += f"![tool_result]({key}.png)"
            else:
                raise AssistantError(f"Unknown result type: {content_type}")
        else:
            response = result.user_message

        response = self._sandwich(response, 8192 * 2, 0.5)

        message_param = ChatCompletionToolMessageParam(
            role="tool", tool_call_id=call_id, content=response
        )
        self.append(message_param)

        if user_parts:
            self.add_content_parts("user", user_parts)

        return f"\n\n{call_text}\n\n"

    def _sandwich(self, text: str, max_chars: int, trim_ratio: float) -> str:
        if len(text) <= max_chars:
            return text
        else:
            total_len = max_chars - 5  # some slop for the ...
            top_len = int(trim_ratio * total_len)
            bot_start = len(text) - (total_len - top_len)
            return text[0:top_len] + " [...] " + text[bot_start:]

    def _add_completion(self, message: ChatCompletionMessage) -> None:
        message_param = message.model_dump()
        if (
            message_param["tool_calls"] is not None
            and len(message_param["tool_calls"]) == 0
        ):
            del message_param["tool_calls"]
        self.append(ChatCompletionAssistantMessageParam(**message_param))

    def add_content_parts(
        self,
        role: str,
        content: ChatCompletionContentPartParam | List[ChatCompletionContentPartParam],
    ):
        if not isinstance(content, List):
            content = [content]

        if not self.model.supports_vision:
            image_exists = any(message["type"] == "image_url" for message in content)
            if image_exists:
                self.logger.warn(
                    f"Skipping image message because model {self.model} does not support vision."
                )

            content = [
                message for message in content if not (message["type"] == "image_url")
            ]

        role_to_param = {
            "user": ChatCompletionUserMessageParam,
            "assistant": ChatCompletionAssistantMessageParam,
            "system": ChatCompletionSystemMessageParam,
            "tool": ChatCompletionToolMessageParam,
            "developer": ChatCompletionDeveloperMessageParam,
        }

        T = role_to_param.get(role)
        assert T is not None, f"Unknown role: {role}"

        # Remove any text message that is empty
        content = [
            message
            for message in content
            if not (message["type"] == "text" and not message["text"])
        ]

        self.append(
            T(
                role=role,
                content=content,
            )
        )

    # def add_prompt_by_key(self, key: str, **prompt_substitutions) -> None:
    #     self.add_text("system", config().get_prompt(key, **prompt_substitutions))

    def compute_and_log_cost(self, usage: CompletionUsage | None) -> None:
        assert usage is not None, "No usage"
        cost = self.model.cost(usage)
        self.logger.charge_cost(cost)

    def _args(self, prediction: str | None = None) -> Dict[str, Any]:
        args = {}
        if self.model.supports_temperature:
            args["temperature"] = self.temperature

        if self.model.supports_prediction and prediction is not None:
            args["prediction"] = ChatCompletionPredictionContentParam(
                type="content", content=prediction
            )
        if self.max_tokens is not None:
            args["max_tokens"] = self.max_tokens

        if len(self.functions) > 0:
            args["tools"] = [tool.function_schema for tool in self.functions.values()]
        return args

    def completion(self, prediction: str | None = None) -> str:
        with self.logger:
            try:
                full_completion_text = ""
                while True:
                    client = self.client
                    completion = client.chat.completions.create(
                        model=self.model.name,
                        messages=self.messages,
                        **self._args(prediction),
                    )
                    # cost
                    self.compute_and_log_cost(completion.usage)

                    # message
                    choice: Choice = completion.choices[0]
                    message: ChatCompletionMessage = choice.message
                    assert message is not None, "Message is None"
                    self._add_completion(message)

                    # yield any text
                    content = message.content
                    if content is not None:
                        full_completion_text += content

                    # stop or do tool calls
                    assert choice.finish_reason in [
                        "stop",
                        "tool_calls",
                    ], f"Unexpected finish reason: {choice.finish_reason}"
                    if choice.finish_reason == "stop":
                        break
                    assert choice.message.tool_calls, "No tool calls"
                    for tool_call in choice.message.tool_calls:
                        full_completion_text += self.make_call(tool_call)
                return full_completion_text
            except openai.OpenAIError as e:
                self.logger.error(str(e))
                for i, m in enumerate(self.messages):
                    self.logger.error(f"Message {i}:")
                    self.logger.error(json.dumps(m, indent=2))
                raise AssistantError.from_openai_error(e)

    def stream(self, prediction: str | None = None) -> Iterator[str]:
        with self.logger:
            try:
                while True:
                    client = self.client
                    stream = client.chat.completions.create(
                        model=self.model.name,
                        messages=self.messages,
                        **self._args(prediction),
                        stream=True,
                        stream_options={"include_usage": True},
                    )
                    final_tool_calls: Dict[int, ChoiceDeltaToolCall] = {}
                    final_content = ""
                    for chunk in stream:
                        if chunk.choices:
                            choice = chunk.choices[0]
                            delta = choice.delta
                            if delta.content is not None:
                                final_content += delta.content
                                yield delta.content
                            for tool_call in chunk.choices[0].delta.tool_calls or []:
                                index = tool_call.index

                                if index not in final_tool_calls:
                                    final_tool_calls[index] = tool_call

                                if tool_call.function:
                                    function = final_tool_calls[index].function
                                    assert function is not None, "No function"
                                    if tool_call.function.name:
                                        function.name = tool_call.function.name
                                    if function.arguments is None:
                                        function.arguments = ""
                                    if tool_call.function.arguments:
                                        function.arguments += (
                                            tool_call.function.arguments
                                        )
                        if chunk.usage:
                            self.compute_and_log_cost(chunk.usage)

                    message = ChatCompletionMessage(
                        role="assistant",
                        content=final_content,
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                **tool_call.model_dump(),
                            )
                            for tool_call in final_tool_calls.values()
                        ],
                    )

                    self._add_completion(message)

                    if message.tool_calls:
                        for tool_call in message.tool_calls:
                            yield self.make_call(tool_call)
                    else:
                        return

            except openai.OpenAIError as e:
                self.logger.error(str(e))
                for i, m in enumerate(self.messages):
                    self.logger.error(f"Message {i}:")
                    self.logger.error(json.dumps(m, indent=2))
                raise AssistantError.from_openai_error(e)

    T = TypeVar("T", bound=BaseModel)

    def model_completion(self, response_model: Type[T]) -> T:
        with self.logger:
            try:
                while True:
                    client = self.client
                    completion = client.beta.chat.completions.parse(
                        model=self.model.name,
                        messages=self.messages,
                        response_format=response_model,  # type: ignore
                        **self._args(),
                    )

                    # cost
                    self.compute_and_log_cost(completion.usage)

                    # message
                    choice: ParsedChoice = completion.choices[0]
                    message: ParsedChatCompletionMessage = choice.message
                    assert message is not None, "Message is None"

                    self._add_completion(message)

                    # stop or do tool calls
                    assert choice.finish_reason in [
                        "stop",
                        "tool_calls",
                    ], f"Unexpected finish reason: {choice.finish_reason}"
                    if choice.finish_reason == "stop":
                        parsed = completion.choices[0].message.parsed
                        assert parsed is not None, "parsed is None"
                        return parsed
                    else:
                        assert choice.message.tool_calls, "No tool calls"
                        for tool_call in choice.message.tool_calls:
                            _ = self.make_call(tool_call)

            except openai.OpenAIError as e:
                self.logger.error(str(e))
                for i, m in enumerate(self.messages):
                    self.logger.error(f"Message {i}:")
                    self.logger.error(json.dumps(m, indent=2))
                raise AssistantError.from_openai_error(e)

    def model_k_completions(self, response_model: Type[T], k: int = 1) -> List[T]:
        with self.logger:
            try:
                while True:
                    client = self.client
                    completion = client.beta.chat.completions.parse(
                        model=self.model.name,
                        messages=self.messages,
                        response_format=response_model,  # type: ignore
                        n=1,
                        **self._args(),
                    )

                    # cost
                    self.compute_and_log_cost(completion.usage)

                    # message
                    choice: ParsedChoice = completion.choices[0]
                    message: ParsedChatCompletionMessage = choice.message
                    assert message is not None, "Message is None"

                    self._add_completion(message)

                    # stop or do tool calls
                    assert choice.finish_reason in [
                        "stop",
                        "tool_calls",
                    ], f"Unexpected finish reason: {choice.finish_reason}"
                    if choice.finish_reason == "stop":
                        parsed = completion.choices[0].message.parsed
                        assert parsed is not None, "parsed is None"
                        return parsed
                    else:
                        assert choice.message.tool_calls, "No tool calls"
                        for tool_call in choice.message.tool_calls:
                            _ = self.make_call(tool_call)

            except openai.OpenAIError as e:
                self.logger.error(str(e))
                for i, m in enumerate(self.messages):
                    self.logger.error(f"Message {i}:")
                    self.logger.error(json.dumps(m, indent=2))
                raise AssistantError.from_openai_error(e)

    def replace_placeholders_with_base64_images(self, markdown: str) -> str:
        for key, image_url in self.cached_images.items():
            markdown = markdown.replace(
                f"![tool_result]({key}.png)", f'\n\n<img src="{image_url}"/>\n\n'
            )
        return markdown

    def model_n_completions(
        self, response_model: Type[T], n: int = 1
    ) -> List[NthCompletion]:
        with self.logger:
            try:
                while True:
                    client = self.client
                    completion = client.beta.chat.completions.parse(
                        model=self.model.name,
                        messages=self.messages,
                        response_format=response_model,  # type: ignore
                        n=n,
                        **self._args(),
                    )

                    # cost
                    self.compute_and_log_cost(completion.usage)

                    results = []
                    print(len(completion.choices))
                    for choice in completion.choices:
                        parsed_choice: ParsedChoice = choice
                        message: ParsedChatCompletionMessage = parsed_choice.message
                        assert message is not None, "Message is None"

                        # make new assistant, preserving all fields and extending the messages to have message
                        assistant = Assistant(
                            model=self.model.name,
                            api_key=self.api_key,
                            functions=self.functions,
                            logger=self.logger,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                        )
                        assistant.messages = deepcopy(self.messages)
                        assistant._add_completion(message)

                        # stop or do tool calls
                        assert choice.finish_reason in [
                            "stop",
                        ], f"Unexpected finish reason: {choice.finish_reason}"

                        parsed = parsed_choice.message.parsed
                        assert parsed is not None, "parsed is None"
                        results.append(NthCompletion(t=parsed, assistant=assistant))
                    return results

            except openai.OpenAIError as e:
                self.logger.error(str(e))
                for i, m in enumerate(self.messages):
                    self.logger.error(f"Message {i}:")
                    self.logger.error(json.dumps(m, indent=2))
                raise AssistantError.from_openai_error(e)

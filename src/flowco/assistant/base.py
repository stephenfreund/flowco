# import inspect
# from typing import Any, Callable, Dict, Iterator, List, Literal, Type, TypeVar


# import openai
# from flowco.assistant.models import get_model
# from pydantic import BaseModel

# from openai.types.chat import (
#     ChatCompletionMessage,
#     ChatCompletionMessageParam,
#     ChatCompletionContentPartParam,
#     ChatCompletionMessageToolCall,
#     ParsedChoice,
#     ParsedChatCompletionMessage,
#     ParsedFunctionToolCall,
# )


# from openai.types.completion_usage import CompletionUsage
# from openai.types.chat.chat_completion import Choice
# from openai.types.chat.chat_completion_chunk import (
#     ChoiceDeltaToolCall,
# )

# from openai.types.chat.chat_completion_content_part_text_param import (
#     ChatCompletionContentPartTextParam,
# )
# from openai.types.chat.chat_completion_content_part_image_param import (
#     ChatCompletionContentPartImageParam,
#     ImageURL,
# )
# from openai.types.chat.chat_completion_content_part_input_audio_param import (
#     ChatCompletionContentPartInputAudioParam,
#     InputAudio,
# )

# from openai.types.chat.chat_completion_tool_message_param import (
#     ChatCompletionToolMessageParam,
# )
# from openai.types.chat.chat_completion_user_message_param import (
#     ChatCompletionUserMessageParam,
# )
# from openai.types.chat.chat_completion_system_message_param import (
#     ChatCompletionSystemMessageParam,
# )
# from openai.types.chat.chat_completion_assistant_message_param import (
#     ChatCompletionAssistantMessageParam,
# )
# from openai.types.chat.chat_completion_developer_message_param import (
#     ChatCompletionDeveloperMessageParam,
# )


# import json
# from flowco.util.config import config

# from typing import Annotated, Callable, get_args, get_origin
# from inspect import signature, Parameter
# import openai
# from pydantic import BaseModel, Field, create_model

# from openai.types.chat import ChatCompletionToolParam


# def to_base_model(func: Callable) -> Type[BaseModel]:
#     """
#     Create a Pydantic BaseModel class representing the parameters of `func`.

#     For parameters whose type is Annotated with a string, that string is used as the field description.
#     """
#     sig = signature(func)
#     fields = {}

#     for name, param in sig.parameters.items():
#         # Start with the parameter's annotation.
#         ann = param.annotation
#         field_description = None

#         # If the annotation is an Annotated type, extract the inner type and description.
#         if get_origin(ann) is Annotated:
#             inner_type, *extras = get_args(ann)
#             ann = inner_type
#             # Look for the first string in the extras to use as the description.
#             for extra in extras:
#                 if isinstance(extra, str):
#                     field_description = extra
#                     break

#         # Determine the default value: if the parameter has no default, mark it as required.
#         default = param.default if param.default is not Parameter.empty else ...

#         # If we have a description, use Field to attach it.
#         if field_description:
#             fields[name] = (ann, Field(default, description=field_description))
#         else:
#             fields[name] = (ann, default)

#     # Create a model whose name is based on the function name.
#     model_name = func.__name__.title() + "Params"
#     return create_model(model_name, **fields)


# def function_to_schema(func: Callable) -> ChatCompletionToolParam:
#     """
#     Create a JSON schema representing the parameters of `func`.

#     For parameters whose type is Annotated with a string, that string is used as the field description.
#     """
#     model = to_base_model(func)
#     return openai.pydantic_function_tool(model)


# class ToolCallResult(BaseModel):
#     user_message: str
#     type: Literal["text", "image", "json"]
#     data: Any


# class ToolDefinition(BaseModel):
#     name: str
#     description: str
#     function_schema: ChatCompletionToolParam
#     function: Callable[..., ToolCallResult]


# class AssistantLogger:
#     def log_message(self, message: ChatCompletionMessageParam) -> None:
#         print(json.dumps(message, indent=2))

#     def log_tool_call(
#         self, tool_call: ChatCompletionMessageToolCall, result: Any
#     ) -> None:
#         print(f"Tool call: {tool_call}")
#         print(f"Result: {result}")

#     def log_cost(self, cost: float) -> None:
#         print(f"Cost: {cost:.3f}")

#     def log(self, message: str) -> None:
#         print(message)

#     def warn(self, message: str) -> None:
#         print(f"Warning: {message}")

#     def error(self, message: str) -> None:
#         print(f"Error: {message}")


# class NewAssistantBase:
#     def __init__(
#         self,
#         model: str,
#         functions: List[Callable[..., ToolCallResult]] = [],
#         logger: AssistantLogger | None = None,
#     ) -> None:
#         self.model = get_model(model)
#         self.messages: List[ChatCompletionMessageParam] = []
#         self.set_functions(functions)
#         self.cached_images = {}
#         self.logger = logger

#     def append(self, message: ChatCompletionMessageParam) -> None:
#         self.log_message(message)
#         self.messages.append(message)

#     def set_functions(self, functions: List[Callable[..., ToolCallResult]]) -> None:
#         defs = [self.function_def(f) for f in functions]
#         self.functions = {def_.name: def_ for def_ in defs}

#     def function_def(self, function: Callable[..., ToolCallResult]) -> ToolDefinition:
#         schema = function_to_schema(function)
#         return ToolDefinition(
#             name=schema["function"]["name"],
#             description=inspect.getdoc(function) or "",
#             function_schema=schema,
#             function=function,
#         )

#     def add_text(self, role: str, text: str) -> None:
#         self._add_message(
#             role, ChatCompletionContentPartTextParam(type="text", text=text)
#         )

#     def add_image(self, role: str, url: str) -> None:
#         self._add_message(
#             role,
#             ChatCompletionContentPartImageParam(
#                 type="image_url", image_url=ImageURL(url=url, detail="high")
#             ),
#         )

#     def add_json(self, role: str, content: Dict[str, Any]) -> None:
#         self._add_message(
#             role,
#             ChatCompletionContentPartTextParam(
#                 type="text", text=json.dumps(content, indent=2)
#             ),
#         )

#     def add_audio(
#         self, role: str, base64_audio: str, format: Literal["wav", "mp3"]
#     ) -> None:
#         self._add_message(
#             role,
#             ChatCompletionContentPartInputAudioParam(
#                 type="input_audio",
#                 input_audio=InputAudio(data=base64_audio, format=format),
#             ),
#         )

#     def make_call(
#         self, tool_call: ChatCompletionMessageToolCall | ParsedFunctionToolCall
#     ) -> str:
#         call_text = ""

#         call_id = tool_call.id
#         function = tool_call.function
#         function_name = function.name
#         args = json.loads(function.arguments)
#         function_def = self.functions[function_name]
#         result = function_def.function(**args)

#         self.log_tool_call(tool_call, result)

#         if result.user_message:
#             call_text += result.user_message

#         if result.type == "text":
#             content = str(result.data)
#         elif result.type == "image":
#             n = len(self.messages)
#             key = f"image{n}"
#             self.cached_images[key] = result.data
#             self._add_message(
#                 "system",
#                 [
#                     ChatCompletionContentPartTextParam(
#                         type="text", text=f"Here is the image `{key}.png`"
#                     ),
#                     ChatCompletionContentPartImageParam(
#                         type="image_url",
#                         image_url=ImageURL(url=result.data, detail="high"),
#                     ),
#                 ],
#             )
#             content = f"The result is the image `{key}.png`."
#             call_text += f"![tool_result]({key}.png)"
#         elif result.type == "json":
#             content = json.dumps(result.data, indent=2)
#         else:
#             raise ValueError(f"Unknown result type: {result.type}")

#         content = self._sandwich(content, 8192 * 2, 0.5)

#         message_param = ChatCompletionToolMessageParam(
#             role="tool", tool_call_id=call_id, content=content
#         )
#         self.append(message_param)
#         return call_text

#     def _sandwich(self, text: str, max_chars: int, trim_ratio: float) -> str:
#         if len(text) <= max_chars:
#             return text
#         else:
#             total_len = max_chars - 5  # some slop for the ...
#             top_len = int(trim_ratio * total_len)
#             bot_start = len(text) - (total_len - top_len)
#             return text[0:top_len] + " [...] " + text[bot_start:]

#     def add_completion(self, message: ChatCompletionMessage) -> None:
#         message_param = ChatCompletionAssistantMessageParam(**message.model_dump())
#         self.append(message_param)

#     def _add_message(
#         self,
#         role: str,
#         content: ChatCompletionContentPartParam | List[ChatCompletionContentPartParam],
#     ):
#         if not isinstance(content, List):
#             content = [content]

#         if not self.model.supports_vision:
#             image_exists = any(message["type"] == "image_url" for message in content)
#             if image_exists:
#                 warn(
#                     f"Skipping image message because model {config.model} does not support vision."
#                 )

#             content = [
#                 message for message in content if not (message["type"] == "image_url")
#             ]

#         role_to_param = {
#             "user": ChatCompletionUserMessageParam,
#             "assistant": ChatCompletionAssistantMessageParam,
#             "system": ChatCompletionSystemMessageParam,
#             "tool": ChatCompletionToolMessageParam,
#             "developer": ChatCompletionDeveloperMessageParam,
#         }

#         T = role_to_param.get(role)
#         assert T is not None, f"Unknown role: {role}"

#         self.append(
#             T(
#                 role=role,
#                 content=content,
#             )
#         )

#     def add_prompt_by_key(self, key: str, **prompt_substitutions) -> None:
#         self.add_text("system", config.get_prompt(key, **prompt_substitutions))

#     def compute_and_log_cost(self, usage: CompletionUsage | None) -> None:
#         assert usage is not None, "No usage"
#         cost = self.model.cost(usage)
#         self.log_cost(cost)

#     def log_cost(self, cost: float) -> None:
#         if self.logger:
#             self.logger.log_cost(cost)

#     def log_message(self, message: ChatCompletionMessageParam) -> None:
#         if self.logger:
#             self.logger.log_message(message)

#     def log_tool_call(
#         self, tool_call: ChatCompletionMessageToolCall, result: Any
#     ) -> None:
#         if self.logger:
#             self.logger.log_tool_call(tool_call, result)

#     def _args(self) -> Dict[str, Any]:
#         args = {}
#         if self.model.supports_temperature:
#             args["temperature"] = 0 if config.zero_temp else None

#         if len(self.functions) > 0:
#             args["tools"] = [tool.function_schema for tool in self.functions.values()]
#         return args

#     def completion(self) -> str:
#         full_completion_text = ""
#         while True:
#             completion = openai.chat.completions.create(
#                 model=self.model.name, messages=self.messages, **self._args()
#             )

#             # cost
#             self.compute_and_log_cost(completion.usage)

#             # message
#             choice: Choice = completion.choices[0]
#             message: ChatCompletionMessage = choice.message
#             assert message is not None, "Message is None"
#             self.add_completion(message)

#             # yield any text
#             content = message.content
#             if content is not None:
#                 full_completion_text += content

#             # stop or do tool calls
#             assert choice.finish_reason in [
#                 "stop",
#                 "tool_calls",
#             ], f"Unexpected finish reason: {choice.finish_reason}"
#             if choice.finish_reason == "stop":
#                 break
#             assert choice.message.tool_calls, "No tool calls"
#             for tool_call in choice.message.tool_calls:
#                 full_completion_text += self.make_call(tool_call)
#         return full_completion_text

#     def stream(self) -> Iterator[str]:
#         while True:
#             stream = openai.chat.completions.create(
#                 model=self.model.name,
#                 messages=self.messages,
#                 **self._args(),
#                 stream=True,
#                 stream_options={"include_usage": True},
#             )

#             final_tool_calls: Dict[int, ChoiceDeltaToolCall] = {}
#             final_content = ""
#             for chunk in stream:
#                 if chunk.choices:
#                     choice = chunk.choices[0]
#                     delta = choice.delta
#                     if delta.content is not None:
#                         final_content += delta.content
#                         yield delta.content
#                     for tool_call in chunk.choices[0].delta.tool_calls or []:
#                         index = tool_call.index

#                         if index not in final_tool_calls:
#                             final_tool_calls[index] = tool_call

#                         if tool_call.function:
#                             function = final_tool_calls[index].function
#                             assert function is not None, "No function"
#                             if tool_call.function.name:
#                                 function.name = tool_call.function.name
#                             if function.arguments is None:
#                                 function.arguments = ""
#                             if tool_call.function.arguments:
#                                 function.arguments += tool_call.function.arguments
#                 if chunk.usage:
#                     self.compute_and_log_cost(chunk.usage)

#             message = ChatCompletionMessage(
#                 role="assistant",
#                 content=final_content,
#                 tool_calls=[
#                     ChatCompletionMessageToolCall(
#                         **tool_call.model_dump(),
#                     )
#                     for tool_call in final_tool_calls.values()
#                 ],
#             )

#             self.add_completion(message)

#             if not message.tool_calls:
#                 break

#             for tool_call in message.tool_calls:
#                 yield self.make_call(tool_call)

#     T = TypeVar("T", bound=BaseModel)

#     def model_completion(self, response_model: Type[T]) -> T:

#         while True:
#             completion = openai.beta.chat.completions.parse(
#                 model=self.model.name,
#                 messages=self.messages,
#                 response_format=response_model,  # type: ignore
#                 **self._args(),
#             )

#             # cost
#             self.compute_and_log_cost(completion.usage)

#             # message
#             choice: ParsedChoice = completion.choices[0]
#             message: ParsedChatCompletionMessage = choice.message
#             assert message is not None, "Message is None"
#             self.add_completion(message)

#             # stop or do tool calls
#             assert choice.finish_reason in [
#                 "stop",
#                 "tool_calls",
#             ], f"Unexpected finish reason: {choice.finish_reason}"
#             if choice.finish_reason == "stop":
#                 parsed = completion.choices[0].message.parsed
#                 assert parsed is not None, "parsed is None"
#                 return parsed
#             else:
#                 assert choice.message.tool_calls, "No tool calls"
#                 for tool_call in choice.message.tool_calls:
#                     _ = self.make_call(tool_call)

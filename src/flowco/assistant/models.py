# from dataclasses import dataclass
# from typing import List
# from openai.types.completion_usage import CompletionUsage


# @dataclass
# class Model:
#     name: str
#     supports_vision: bool
#     supports_audio: bool
#     supports_temperature: bool

#     # completion token costs
#     completion_token_rate: float  # Rate per completion token
#     prompt_token_rate: float  # Rate per prompt token
#     cached_token_rate: float  # Rate per cached token

#     def cost(self, usage: CompletionUsage) -> float:
#         completion_cost = usage.completion_tokens * self.completion_token_rate

#         if usage.prompt_tokens_details:
#             prompt_details = usage.prompt_tokens_details
#             full_cost_tokens = usage.prompt_tokens - (prompt_details.cached_tokens or 0)

#             prompt_cost = (
#                 full_cost_tokens * self.prompt_token_rate
#                 + (prompt_details.cached_tokens or 0) * self.cached_token_rate
#             )
#         else:
#             prompt_cost = usage.prompt_tokens * self.prompt_token_rate

#         return completion_cost + prompt_cost


# _models = {
#     "o3-mini": Model(
#         name="o3-mini",
#         supports_vision=False,
#         supports_audio=False,
#         supports_temperature=False,
#         completion_token_rate=4.4 / 10**6,
#         prompt_token_rate=1.1 / 10**6,
#         cached_token_rate=0.55 / 10**6,
#     ),
#     "o1": Model(
#         name="o1",
#         supports_vision=True,
#         supports_audio=False,
#         supports_temperature=True,
#         completion_token_rate=60 / 10**6,
#         prompt_token_rate=15 / 10**6,
#         cached_token_rate=7.5 / 10**6,
#     ),
#     "gpt-4o": Model(
#         name="gpt-4o",
#         supports_vision=True,
#         supports_audio=False,
#         supports_temperature=True,
#         completion_token_rate=4.4 / 10**6,
#         prompt_token_rate=2.50 / 10**6,
#         cached_token_rate=1.25 / 10**6,
#     ),
# }


# def supported_models() -> List[str]:
#     return list(_models.keys())


# def get_model(model_name: str) -> Model:
#     return _models[model_name]

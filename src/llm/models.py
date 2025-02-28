from dataclasses import dataclass
from typing import List
from openai.types.completion_usage import CompletionUsage


@dataclass
class Model:
    name: str
    use_proxy: bool

    supports_vision: bool
    supports_audio: bool
    supports_temperature: bool
    supports_prediction: bool

    # completion token costs
    completion_token_rate: float  # Rate per completion token
    prompt_token_rate: float  # Rate per prompt token
    cached_token_rate: float  # Rate per cached token

    def cost(self, usage: CompletionUsage) -> float:
        completion_cost = usage.completion_tokens * self.completion_token_rate

        if usage.prompt_tokens_details:
            prompt_details = usage.prompt_tokens_details
            full_cost_tokens = usage.prompt_tokens - (prompt_details.cached_tokens or 0)

            prompt_cost = (
                full_cost_tokens * self.prompt_token_rate
                + (prompt_details.cached_tokens or 0) * self.cached_token_rate
            )
        else:
            prompt_cost = usage.prompt_tokens * self.prompt_token_rate

        return completion_cost + prompt_cost


_models = {
    "o3-mini": Model(
        name="o3-mini",
        use_proxy=False,
        supports_vision=False,
        supports_audio=False,
        supports_temperature=False,
        supports_prediction=False,
        completion_token_rate=4.4 / 10**6,
        prompt_token_rate=1.1 / 10**6,
        cached_token_rate=0.55 / 10**6,
    ),
    "o1": Model(
        name="o1",
        use_proxy=False,
        supports_vision=True,
        supports_audio=False,
        supports_temperature=True,
        supports_prediction=False,
        completion_token_rate=60 / 10**6,
        prompt_token_rate=15 / 10**6,
        cached_token_rate=7.5 / 10**6,
    ),
    "gpt-4o": Model(
        name="gpt-4o-2024-11-20",
        use_proxy=False,
        supports_vision=True,
        supports_audio=False,
        supports_temperature=True,
        supports_prediction=True,
        completion_token_rate=10 / 10**6,
        prompt_token_rate=2.50 / 10**6,
        cached_token_rate=1.25 / 10**6,
    ),
    "gpt-4.5": Model(
        name="gpt-4.5-preview",
        use_proxy=False,
        supports_vision=True,
        supports_audio=False,
        supports_temperature=True,
        supports_prediction=True,
        completion_token_rate=150 / 10**6,
        prompt_token_rate=75 / 10**6,
        cached_token_rate=35 / 10**6,
    ),
    "gpt-4o-mini": Model(
        name="gpt-4o-mini",
        use_proxy=False,
        supports_vision=False,
        supports_audio=False,
        supports_temperature=True,
        supports_prediction=True,
        completion_token_rate=0.60 / 10**6,
        prompt_token_rate=0.15 / 10**6,
        cached_token_rate=0.075 / 10**6,
    ),
    "claude-3-sonnet": Model(
        name="claude-3-sonnet",
        use_proxy=True,
        supports_vision=True,
        supports_audio=False,
        supports_temperature=False,
        supports_prediction=False,
        completion_token_rate=15 / 10**6,
        prompt_token_rate=3 / 10**6,
        cached_token_rate=3 / 10**6,
    ),
    "claude-3-haiku": Model(
        name="claude-3-haiku",
        use_proxy=True,
        supports_vision=False,
        supports_audio=False,
        supports_temperature=False,
        supports_prediction=False,
        completion_token_rate=15 / 10**6,
        prompt_token_rate=3 / 10**6,
        cached_token_rate=3 / 10**6,
    ),
}


def supported_models() -> List[str]:
    return list(_models.keys())


def get_model(model_name: str) -> Model:
    return _models[model_name]

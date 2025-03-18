import openai
from flowco.util.config import config
from flowco.util.costs import add_cost, decrement_inflight, increment_inflight
from flowco.util.output import error, log, warn, debug
from llm.assistant import Assistant, AssistantLogger


class FlowcoLogger(AssistantLogger):
    def log(self, message: str):
        log(message)

    def error(self, message: str):
        error(message)

    def warn(self, message: str):
        warn(message)

    def debug(self, message: str):
        debug(message)

    def charge_cost(self, cost: float) -> None:
        log("Charging cost: ", cost)
        add_cost(cost)

    def __enter__(self):
        increment_inflight()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        decrement_inflight()


def flowco_assistant(
    prompt_key: str | None = None, **prompt_substitutions
) -> Assistant:
    model = config().model
    temperature = 0 if config().zero_temp else None

    assert "functions" not in prompt_substitutions, "functions is a reserved key"

    assistant = Assistant(model, logger=FlowcoLogger(), temperature=temperature)
    assistant.add_text("system", config().get_prompt("system-prompt"))
    if prompt_key:
        prompt = config().get_prompt(prompt_key, **prompt_substitutions)
        assistant.add_text("system", prompt)
    return assistant


def flowco_assistant_fast(
    prompt_key: str | None = None, **prompt_substitutions
) -> Assistant:
    temperature = 0 if config().zero_temp else None

    assert "functions" not in prompt_substitutions, "functions is a reserved key"

    assistant = Assistant("gpt-4o-mini", logger=FlowcoLogger(), temperature=temperature)
    if prompt_key:
        prompt = config().get_prompt(prompt_key, **prompt_substitutions)
        assistant.add_text("system", prompt)
    return assistant


def fast_text_complete(prompt: str) -> str:
    assistant = Assistant("gpt-4o-mini", logger=FlowcoLogger(), max_tokens=10)
    assistant.add_text("system", prompt)
    return assistant.completion()


def fast_transcription(voice):
    transcription = openai.audio.transcriptions.create(
        model="whisper-1", file=voice, response_format="verbose_json"
    )
    cost = round(float(transcription.duration)) * 0.006 / 60
    add_cost(cost)  # this one is handled by the assistant
    return transcription.text

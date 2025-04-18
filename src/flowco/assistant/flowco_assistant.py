import datetime
import json
import sqlite3
import sys
from typing import Any, Dict, Type, TypeVar

import openai
from pydantic import BaseModel

from flowco.assistant.flowco_keys import get_api_key
from flowco.llm.assistant import Assistant, AssistantLogger
from flowco.llm.models import get_model
from flowco.util.config import config
from flowco.util.costs import add_cost, decrement_inflight, increment_inflight
from flowco.util.output import debug, error, log, warn


# ---- single SQLite DB setup ----

# DB_PATH = "responses.db"


# def _init_db():
#     conn = sqlite3.connect(DB_PATH)
#     c = conn.cursor()
#     c.execute(
#         """
#         CREATE TABLE IF NOT EXISTS responses (
#             id TEXT,
#             timestamp REAL,
#             model TEXT,
#             response_model TEXT,
#             messages TEXT
#         )
#     """
#     )
#     conn.commit()
#     conn.close()


# _init_db()


# ---- logger ----


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
        log("Charging cost:", cost)
        add_cost(cost)

    def __enter__(self):
        increment_inflight()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        decrement_inflight()


class FlowcoCompletion(BaseModel):
    model: str
    response_model: Dict[str, Any] | None = None
    messages: list[Dict[str, str]] | None = None


class FlowcoAssistant(Assistant):

    def __init__(self, id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = id

    def _dump(self, info: dict) -> None:
        pass
        # """Insert one record into the SQLite database."""
        # conn = sqlite3.connect(DB_PATH)
        # c = conn.cursor()
        # c.execute(
        #     "INSERT INTO responses (id, timestamp, model, response_model, messages) VALUES (?, ?, ?, ?, ?)",
        #     (
        #         self.id,
        #         info["timestamp"],
        #         info["model"],
        #         json.dumps(info.get("response_model")),
        #         json.dumps(info.get("messages")),
        #     ),
        # )
        # conn.commit()
        # conn.close()

    T = TypeVar("T", bound=BaseModel)

    def model_completion(self, response_model: Type[T]) -> T:
        result: T = super().model_completion(response_model)
        info = {
            "model": self.model.name,
            "response_model": response_model.model_json_schema(),
            "messages": self.messages,
            "timestamp": datetime.datetime.now().timestamp(),
        }
        self._dump(info)
        return result

    def completion(self, prediction: str | None = None) -> str:
        result = super().completion()
        info = {
            "model": self.model.name,
            "messages": self.messages,
            "timestamp": datetime.datetime.now().timestamp(),
        }
        self._dump(info)
        return result


def flowco_assistant(
    id: str, prompt_key: str | None = None, **prompt_substitutions
) -> Assistant:
    model_name = config().model
    model = get_model(model_name)
    temperature = 0 if config().zero_temp else None
    api_key = get_api_key(model.api_key_name)
    assert "functions" not in prompt_substitutions, "functions is a reserved key"

    assistant = FlowcoAssistant(
        id, model_name, api_key=api_key, logger=FlowcoLogger(), temperature=temperature
    )
    assistant.add_text("system", config().get_prompt("system-prompt"))
    if prompt_key:
        assistant.add_text(
            "user", config().get_prompt(prompt_key, **prompt_substitutions)
        )
    return assistant


def flowco_assistant_fast(
    id: str, prompt_key: str | None = None, **prompt_substitutions
) -> Assistant:
    temperature = 0 if config().zero_temp else None
    assert "functions" not in prompt_substitutions, "functions is a reserved key"

    model = get_model("gpt-4o-mini")
    api_key = get_api_key(model.api_key_name)
    assistant = FlowcoAssistant(
        id, model.name, api_key=api_key, logger=FlowcoLogger(), temperature=temperature
    )
    if prompt_key:
        assistant.add_text(
            "user", config().get_prompt(prompt_key, **prompt_substitutions)
        )
    return assistant


def fast_text_complete(id: str, prompt: str) -> str:
    model = get_model("gpt-4o-mini")
    api_key = get_api_key(model.api_key_name)
    assistant = FlowcoAssistant(
        id, model.name, api_key=api_key, logger=FlowcoLogger(), max_tokens=10
    )
    assistant.add_text("user", prompt)
    return assistant.completion()


def fast_transcription(voice):
    api_key = get_api_key("OPENAI_API_KEY")
    client = openai.Client(api_key=api_key)
    transcription = client.audio.transcriptions.create(
        model="whisper-1", file=voice, response_format="verbose_json"
    )
    cost = round(float(transcription.duration)) * 0.006 / 60
    add_cost(cost)
    return transcription.text


def test_openai_key() -> bool:
    try:
        model = get_model("gpt-4o-mini")
        api_key = get_api_key(model.api_key_name)
        assistant = Assistant(
            model.name, api_key=api_key, logger=FlowcoLogger(), max_tokens=10
        )
        assistant.add_text("user", "Say hello")
        assistant.completion()
        return True
    except Exception:
        return False


def test_anthropic_key() -> bool:
    try:
        model = get_model("claude-3-haiku")
        api_key = get_api_key(model.api_key_name)
        assistant = Assistant(
            model.name, api_key=api_key, logger=FlowcoLogger(), max_tokens=10
        )
        assistant.add_text("user", "Say hello")
        assistant.completion()
        return True
    except Exception:
        return False

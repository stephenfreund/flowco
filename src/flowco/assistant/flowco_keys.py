from datetime import datetime, timedelta
import os
from typing import Dict, List

from pydantic import BaseModel

from flowco.session.session import session
from flowco.session.session_file_system import fs_read, fs_write
from flowco.util.output import log
from flowco.llm.assistant import AssistantError


class FlowcoKeys:
    def get_api_key(self, api_key_name: str) -> str | None:
        pass

    def set_api_key(self, key_name: str, key_value: str):
        pass

    def get_api_key_status(self, api_key_name: str) -> str:
        pass


class KeyEnv(FlowcoKeys):
    def get_api_key(self, api_key_name: str) -> str | None:
        key = os.environ.get(api_key_name, None)
        if key is None:
            key = os.environ.get("DEFAULT_" + api_key_name, None)
        return key

    def set_api_key(self, key_name: str, key_value: str):
        os.environ[key_name] = key_value

    def get_api_key_status(self, api_key_name: str) -> str:
        key = self.get_api_key(api_key_name)
        if key is None:
            return "Key {api_key_name} is not in the environment variables."
        else:
            return f"Key {api_key_name} is set."


class ApiKey(BaseModel):
    key: str
    expiration: datetime


key_file_name = "keys.json"


class KeyFile(BaseModel):
    keys: Dict[str, ApiKey]

    @staticmethod
    def load() -> "KeyFile":
        try:
            keys_json = fs_read(key_file_name)
            return KeyFile.model_validate_json(keys_json)
        except FileNotFoundError:
            keys = KeyFile._from_defaults()
            keys.save()
            return keys

    def save(self):
        fs_write(key_file_name, self.model_dump_json(indent=2))

    @staticmethod
    def _from_defaults() -> "KeyFile":
        openai_key = ApiKey(
            key=os.environ.get("OPENAI_API_KEY", ""),
            expiration=datetime.now() + timedelta(hours=1),
        )
        # anthropic_key = ApiKey(
        #     key=os.environ.get("DEFAULT_ANTHROPIC_API_KEY", ""),
        #     expiration=datetime.now() + timedelta(hours=0.1),
        # )
        return KeyFile(
            keys={
                "OPENAI_API_KEY": openai_key,
                # "ANTHROPIC_API_KEY": anthropic_key,
            }
        )

    def get_api_key(self, api_key_name: str) -> str | None:
        key = self.keys.get(api_key_name)
        if key is not None:
            if key.expiration > datetime.now():
                return key.key
            else:
                del self.keys[api_key_name]
                self.save()
        return None

    def set_api_key(self, key_name: str, key_value: str):
        self.keys[key_name] = ApiKey(
            key=key_value,
            expiration=datetime.now() + timedelta(weeks=52 * 10),
        )
        self.save()

    def get_api_key_status(self, api_key_name: str) -> str:
        key = self.get_api_key(api_key_name)
        if key is None:
            return f"Key {api_key_name} has not been set."

        if self.keys[api_key_name].expiration < datetime.now():
            return f"Key {api_key_name} has expired."

        minutes_to_expire = (
            self.keys[api_key_name].expiration - datetime.now()
        ).total_seconds() / 60
        if minutes_to_expire < 60:
            return f"Key {api_key_name} will expire in {minutes_to_expire:.1f} minutes."

        return f"Key {api_key_name} is set."


class UserKeys(FlowcoKeys):

    def __init__(self):
        self.key_file = None

    def load(self) -> KeyFile:
        log("Loading keys")
        if self.key_file is None:
            self.key_file = KeyFile.load()
        return self.key_file

    def get_api_key(self, api_key_name: str) -> str | None:
        log("Getting key", api_key_name)
        return self.load().get_api_key(api_key_name)

    def set_api_key(self, key_name: str, key_value: str):
        return self.load().set_api_key(key_name, key_value)

    def get_api_key_status(self, api_key_name: str) -> str:
        return self.load().get_api_key_status(api_key_name)


def get_api_key(api_key_name: str) -> str:
    key = session.get("keys", FlowcoKeys).get_api_key(api_key_name)
    if key is None:
        raise AssistantError(
            f"Your API key {api_key_name} is invalid, expired, or revoked.  Add a valid API key under Settings.  For OpenAI, you can find your API key at https://platform.openai.com/account/api-keys."
        )
    # log(f"Using {api_key_name} key {key}")
    return key


def set_api_key(key_name: str, key_value: str):
    session.get("keys", FlowcoKeys).set_api_key(key_name, key_value)


def get_api_key_status(api_key_name: str) -> str:
    return session.get("keys", FlowcoKeys).get_api_key_status(api_key_name)

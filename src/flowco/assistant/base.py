from flowco.util.output import debug, warn

import instructor
import litellm
from litellm.utils import supports_vision

import json
from flowco.util.config import config


class AssistantBase:
    def __init__(
        self, system_prompt_key: str | list[str], **system_prompt_substitutions
    ) -> None:
        self.instructor_client = instructor.from_litellm(litellm.completion)
        self.messages: list[dict[str, any]] = []

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

    def add_message(
        self, role: str, content: str | dict[str, any] | list[str | dict[str, any]]
    ):
        debug(
            "Add Message: " + json.dumps({"role": role, "content": content}, indent=2)
        )

        if isinstance(content, dict):
            content = [content]

        if not isinstance(content, str) and not supports_vision(config.model):
            for message in content:
                if isinstance(message, dict) and message["type"] == "image_url":
                    warn(
                        f"Skipping image message because model {config.model} does not support vision."
                    )

            content = [
                message
                for message in content
                if not (isinstance(message, dict) and message["type"] == "image_url")
            ]

        self.messages.append({"role": role, "content": content})

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

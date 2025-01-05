import argparse
from enum import StrEnum
from typing import Any, Dict, List
import os

import yaml

from typing import TypeVar
from pydantic import BaseModel


class AbstractionLevel(StrEnum):
    # label = "Graph"
    spec = "Requirements"
    algorithm = "Algorithm"
    code = "Code"

    @staticmethod
    def show_requirements(x):
        return True
        # return x != AbstractionLevel.label

    @staticmethod
    def show_algorithm(x):
        return x == AbstractionLevel.algorithm or x == AbstractionLevel.code

    @staticmethod
    def show_code(x):
        return x == AbstractionLevel.code


T = TypeVar("T", bool, int, str)


def _flowco_get_env(option_name: str, default_value: T) -> T:
    """
    Get the value of an environment variable, or return the default value if it is not set.
    """
    env_name = "FLOWCO_" + option_name.upper()
    v = os.getenv(env_name, str(default_value))
    if isinstance(default_value, bool):
        return v.lower() == "true"
    elif isinstance(default_value, int):
        return int(v)
    else:
        return v


class Config:
    """
    Configuration class for Flowco. This class is used to store the global configuration of the Flowco system.
    """

    def __init__(self):
        """
        Initialize the configuration class with default values.
        """
        self.model = _flowco_get_env("model", "gpt-4o-2024-11-20")
        self.stream = bool(int(_flowco_get_env("stream", "0")))
        self.debug = bool(int(_flowco_get_env("debug", "0")))
        self.quiet = int(_flowco_get_env("quiet", "0"))
        self.log = _flowco_get_env("log", "log.yaml")
        self.prompts = _flowco_get_env(
            "prompts", os.path.join(os.path.dirname(__file__), "prompts.yaml")
        )
        self.diff = bool(int(_flowco_get_env("diff", "0")))
        self.retries = int(_flowco_get_env("retries", "3"))
        self.builder = _flowco_get_env("builder", "node-passes")
        self.sequential = bool(int(_flowco_get_env("sequential", "0")))
        self.zero_temp = _flowco_get_env("zero_temp", "1") != "0"
        self.abstraction_level = AbstractionLevel(
            _flowco_get_env("abstraction_level", "Requirements")
        )

        # experimental features
        self.x_no_descriptions = _flowco_get_env("x_no_descriptions", "1") != "0"
        self.x_shortcurcuit_requirements = (
            _flowco_get_env("x_shortcurcuit_requirements", "0") != "0"
        )
        self.x_no_dfg_image_in_prompt = (
            _flowco_get_env("x_no_dfg_image_in_prompt", "0") != "0"
        )
        self.x_trust_ama = _flowco_get_env("x_trust_ama", "1") != "0"
        self.x_algorithm_phase = _flowco_get_env("a_algorithm_phase", "0") != "0"

    def get_x_options(self) -> Dict[str, Any]:
        return {k: v for k, v in vars(self).items() if k.startswith("x_")}

    def set_x_options(self, options: Dict[str, Any]):
        for k, v in options.items():
            setattr(self, k, v)

    def parser(self):
        """
        Create an argument parser for the configuration class.
        """
        parser = argparse.ArgumentParser(description="Flowco", allow_abbrev=False)
        parser.add_argument(
            "--model",
            default=self.model,
            type=str,
            help="The LLM model",
            action=UpdateConfigAction,
            config=self,
        )
        parser.add_argument(
            "--stream",
            type=bool,
            nargs=0,
            default=self.stream,
            help="Stream the full graph completions",
            action=StoreTrueConfigAction,
            config=self,
        )
        parser.add_argument(
            "--debug",
            type=bool,
            nargs=0,
            default=self.debug,
            help="Print debug information",
            action=StoreTrueConfigAction,
            config=self,
        )
        parser.add_argument(
            "--quiet",
            type=bool,
            nargs=0,
            default=self.quiet,
            help="Turn off flowco.util.output messages",
            action=StoreTrueConfigAction,
            config=self,
        )
        parser.add_argument(
            "--log",
            default=self.log,
            type=str,
            help="The log file",
            action=UpdateConfigAction,
            config=self,
        )
        parser.add_argument(
            "--prompts",
            default=self.prompts,
            type=str,
            help="The prompts for the LLM",
            action=UpdateConfigAction,
            config=self,
        )
        parser.add_argument(
            "--diff",
            type=bool,
            nargs=0,
            default=self.debug,
            help="Print version diffs for atomic page updates",
            action=StoreTrueConfigAction,
            config=self,
        )
        parser.add_argument(
            "--retries",
            default=self.retries,
            type=int,
            help="Number of retries for LLM completions",
            action=UpdateConfigAction,
            config=self,
        )
        parser.add_argument(
            "--builder",
            default=self.builder,
            type=str,
            help="The passes key for the build process (see build.yaml)",
            action=UpdateConfigAction,
            config=self,
        )
        parser.add_argument(
            "--sequential",
            type=bool,
            nargs=0,
            default=self.sequential,
            help="Run the build process in single thread",
            action=StoreTrueConfigAction,
            config=self,
        )
        parser.add_argument(
            "--zero_temp",
            type=bool,
            nargs=0,
            default=self.zero_temp,
            help="Set LLM temp to 0",
            action=StoreTrueConfigAction,
            config=self,
        )
        # parser.add_argument(
        #     "--regenerate_policy",
        #     default=self.regenerate_policy,
        #     type=RegenerationPolicy,
        #     help="The policy for regenerating LLM completions",
        #     action=UpdateConfigAction,
        #     config=self,
        # )
        parser.add_argument(
            "--abstraction_level",
            default=self.abstraction_level,
            type=AbstractionLevel,
            help="The abstraction level",
            action=UpdateConfigAction,
            config=self,
        )
        parser.add_argument(
            "--x_no_descriptions",
            type=bool,
            nargs=0,
            default=self.x_no_descriptions,
            help="Compute descriptions of nodes and computed values",
            action=StoreTrueConfigAction,
            config=self,
        )
        parser.add_argument(
            "--x_shortcurcuit_requirements",
            type=bool,
            nargs=0,
            default=self.x_shortcurcuit_requirements,
            help="Use the LLM to check for precondition changes",
            action=StoreTrueConfigAction,
            config=self,
        )
        parser.add_argument(
            "--x_no_dfg_image_in_prompt",
            type=bool,
            nargs=0,
            default=self.x_no_dfg_image_in_prompt,
            help="Don't send dataflow image in prompt",
            action=StoreTrueConfigAction,
            config=self,
        )
        parser.add_argument(
            "--x_trust_ama",
            type=bool,
            nargs=0,
            default=self.x_trust_ama,
            help="Trust the AMA to provide the correct output",
            action=StoreTrueConfigAction,
            config=self,
        )
        parser.add_argument(
            "--x_algorithm_phase",
            type=bool,
            nargs=0,
            default=self.x_algorithm_phase,
            help="Include the algorithm phase",
            action=StoreTrueConfigAction,
            config=self,
        )
        return parser

    def get_yaml(self, file_path: str) -> dict:
        """
        Get a dictionary entry from a yaml file.
        """
        with open(file_path, "r") as f:
            x = f.read()
            db = yaml.safe_load(x)
        return db

    def get_prompt(self, prompt_key: str, **substitutions) -> str:
        """
        Get a prompt from the prompt database.
        """
        if not hasattr(self, "prompt_dict"):
            self.prompt_dict = self.get_yaml(self.prompts)
        prompt = self.prompt_dict[prompt_key]

        json_subs = {}
        for k, v in substitutions.items():
            if isinstance(v, BaseModel):
                json_subs[k] = v.model_dump_json(indent=2)
            else:
                json_subs[k] = str(v)

        return prompt.format_map(json_subs)

    def get_build_passes(self) -> List[str]:
        """
        Get the passes for the build process.  key is at least build/repair
        """
        return self.get_yaml(os.path.join(os.path.dirname(__file__), "builder.yaml"))[
            self.builder
        ]

    def get_build_passes_keys(self) -> List[str]:
        """
        Get the keys for the build process.
        """
        return list(
            self.get_yaml(
                os.path.join(os.path.dirname(__file__), "builder.yaml")
            ).keys()
        )


class UpdateConfigAction(argparse.Action):
    def __init__(self, option_strings, dest, config, **kwargs):
        self.config = config
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if (
            values is None and self.default is True
        ):  # Handle store_true case for boolean fields
            values = True
        setattr(self.config, self.dest, values)
        setattr(namespace, self.dest, values)


class StoreTrueConfigAction(argparse.Action):
    def __init__(self, option_strings, dest, config, **kwargs):
        self.config = config
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        values = True
        setattr(self.config, self.dest, values)
        setattr(namespace, self.dest, values)


config = Config()

# Example usage:
if __name__ == "__main__":
    config = Config()
    parser = config.parser()
    args = parser.parse_args()

    print(f"Config: {vars(config)}")
    print(f"Args: {vars(args)}")

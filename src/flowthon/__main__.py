import argparse
from genericpath import exists
import inspect
import os
import re
import sys
from abc import ABC, abstractmethod
from typing import Dict, List

from matplotlib import interactive
import nbformat

from flowthon.nbflowthon import convert_notebook_to_flowthon
from flowthon.nbsplit import split_notebook_by_heading_level
from flowco.builder.build import BuildEngine
from flowco.dataflow.phase import Phase
from flowco.page.page import Page
from flowco.pythonshell.shells import PythonShells
from flowco.session.session import session
from flowco.session.session_file_system import SessionFileSystem
from flowco.util.config import AbstractionLevel, config
from flowco.util.costs import CostTracker, call_count, total_cost
from flowco.util.output import Output, error, message, logger

from flowco.util.errors import FlowcoError
from flowco.util.stoppable import Stoppable

### Base


class Command(ABC):
    @abstractmethod
    def __init__(self, subparsers):
        pass

    @abstractmethod
    def run(self, args: argparse.Namespace):
        pass


### Page Commands


class ConvertCommand(Command):
    def __init__(self, subparsers):
        super().__init__(subparsers)
        self.parser = subparsers.add_parser(
            "convert", help="Convert notebook to Flowthon"
        )
        self.parser.add_argument(
            "--split",
            type=int,
            default=0,
            help="Heading level for splitting notebook (or 0 for no split)",
        )
        self.parser.add_argument(
            "--force", help="overwrite file if exsts", action="store_true"
        )

        self.parser.add_argument(
            "notebook_name",
            type=str,
            help="Notebook name",
        )

    def run(self, args):
        def convert_to_flowthon(original_nb, flowthon_file):
            if exists(flowthon_file) and not args.force:
                raise FlowcoError(f"File {flowthon_file} already exists.")
            flowthon = convert_notebook_to_flowthon(args.model, original_nb)
            with open(flowthon_file, "w", encoding="utf-8") as f:
                print(flowthon, file=f)
            message(f"Saved: {flowthon_file}")

        notebook_name = args.notebook_name
        notebook_root = notebook_name.split(".ipynb")[0]

        with open(notebook_name, "r", encoding="utf-8") as f:
            original_nb = nbformat.read(f, as_version=4)

        if args.split == 0:
            flowthon_file = f"{notebook_root}.flowthon"
            convert_to_flowthon(original_nb, flowthon_file)
        else:
            notebooks = split_notebook_by_heading_level(original_nb, args.split)
            for idx, nb in enumerate(notebooks, start=1):
                flowthon_file = f"{notebook_root}_{idx}.flowthon"
                convert_to_flowthon(nb, flowthon_file)


###


class CreateCommand(Command):
    def __init__(self, subparsers):
        super().__init__(subparsers)
        self.parser = subparsers.add_parser("create", help="Make Flowthon file")
        self.parser.add_argument("--flowco", type=str, help="Convert from flowco file")
        self.parser.add_argument(
            "--algorithm", help="show algorithm", action="store_true"
        )
        self.parser.add_argument("--code", help="show code", action="store_true")
        self.parser.add_argument(
            "--force", help="overwrite file if exsts", action="store_true"
        )

        self.parser.add_argument(
            "file_name",
            type=str,
            help="Flowthon file name",
        )

    def run(self, args):
        if args.code:
            level = AbstractionLevel.code
        elif args.algorithm:
            level = AbstractionLevel.algorithm
        else:
            level = AbstractionLevel.spec

        file_name = args.file_name
        if exists(file_name) and not args.force:
            raise FlowcoError(f"File {file_name} already exists.")

        if not file_name.endswith(".flowthon"):
            raise FlowcoError("File name must end with .flowthon")

        if args.flowco:
            flowco_file = args.flowco
        else:
            flowco_file = file_name.replace(".flowthon", ".flowco")
            Page.create(flowco_file)
            message(f"Created {flowco_file}.")

        page = Page.from_file(flowco_file)

        page.to_flowthon(level, file_name)
        message(f"Created {file_name}.")


###


class RunCommand(Command):
    def __init__(self, subparsers):
        super().__init__(subparsers)
        self.parser = subparsers.add_parser("run", help="Run Flowthon file")
        self.parser.add_argument(
            "--nobuild", help="Do not rebuild before running", action="store_true"
        )
        self.parser.add_argument("--flowco", type=str, help="Underlying Flowco file")
        self.parser.add_argument(
            "--html", help="Open the html file", action="store_true"
        )
        self.parser.add_argument(
            "--interactive", help="Interactive mode", action="store_true"
        )
        self.parser.add_argument(
            "file_name",
            type=str,
            help="File to build",
        )

    def run(self, args):
        file_name = args.file_name
        if not exists(file_name):
            raise FlowcoError(f"File {file_name} does not exist.")

        if args.flowco:
            flowco_file = args.flowco
        else:
            flowco_file = file_name.replace(".flowthon", ".flowco")

        if exists(flowco_file):
            page = Page.from_file(flowco_file)
        else:
            page = Page.create(flowco_file)
            message(f"Created {flowco_file}.")

        if args.nobuild:
            builder = BuildEngine.get_builder()
            page.clean(None, Phase.runnable)
            for _ in page.build(builder, Phase.run_checked, False, None):
                pass
        else:
            page.merge_flowthon(args.file_name, interactive=args.interactive)

        page.save()

        html_file = file_name.replace(".flowthon", ".html")
        with open(html_file, "w") as f:
            f.write(page.to_html())
            message(f"Wrote HTML to {html_file}")

        if args.html:
            os.system(f"open {html_file}")


class CleanCommand(Command):
    def __init__(self, subparsers):
        super().__init__(subparsers)
        self.parser = subparsers.add_parser("clean", help="Clean Flowthon file")
        self.parser.add_argument(
            "file_name",
            type=str,
            help="File to clean",
        )

    def run(self, args):
        file_name = args.file_name
        flowco_file = file_name.replace(".flowthon", ".flowco")

        page = Page.from_file(flowco_file)
        page = Page.create(flowco_file)
        page.clean()


def class_name_to_key(name):
    base_name = name[:-7]  # Remove "Command" suffix
    key = re.sub(r"(?<!^)(?=[A-Z])", "-", base_name).lower()
    return key


def create_commands(
    subparsers: argparse._SubParsersAction,
) -> Dict[str, Command]:
    commands = {}
    current_module = sys.modules[__name__]
    for name, obj in inspect.getmembers(current_module, inspect.isclass):
        if name.endswith("Command") and obj.__module__ == __name__:
            if not inspect.isabstract(obj):
                key = class_name_to_key(name)
                commands[key] = obj(subparsers)
    return commands


def main_core(argv: List[str]):
    parser = config.parser()

    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")
    subparsers.required = True  # Ensures that a subcommand is required

    commands = create_commands(subparsers=subparsers)

    args = parser.parse_args(argv)

    command = commands.get(args.command)

    if command:
        with logger(f"Running `{args.command}` command"):
            command.run(args)
    else:
        raise FlowcoError(f"Unknown command: {args.command}")


def main(argv: List[str] = sys.argv[1:]):
    session.set(
        output=Output(),
        costs=CostTracker(),
        filesystem=SessionFileSystem(f"file://{os.getcwd()}"),
    )

    try:
        with Stoppable():
            main_core(argv=argv)
    except FlowcoError as e:
        error(f"Error: {e}")
    finally:
        if call_count() == 1:
            message(f"Total cost: {total_cost():.2f} for 1 completion.")
        elif call_count() > 1:
            message(f"Total cost: {total_cost():.2f} for {call_count()} completions.")


if __name__ == "__main__":
    # Gotta do this somewhere, lest a missing key exception gets generated inside a library
    # that doesn't report it to us properly.
    if os.environ.get("OPENAI_API_KEY", None) is None:
        raise FlowcoError("You must set the OPENAI_API_KEY environment variable")

    main(page=None, argv=sys.argv)

import argparse
from genericpath import exists
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import sys
from abc import ABC, abstractmethod
from typing import Dict, List
import webbrowser

import nbformat
from sklearn import base

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
from flowco.util.stopper import Stopper
from flowthon.flowthon import FlowthonProgram

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
            flowthon_file = f"{notebook_root}.flowjson"
            convert_to_flowthon(original_nb, flowthon_file)
        else:
            notebooks = split_notebook_by_heading_level(original_nb, args.split)
            for idx, nb in enumerate(notebooks, start=1):
                flowthon_file = f"{notebook_root}_{idx}.flowjson"
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

        base_name, ext = os.path.splitext(file_name)

        if ext != ".flowthon" and ext != ".flowjson":
            raise FlowcoError("File name must end with .flowthon or .flowjson")

        if exists(file_name) and not args.force:
            raise FlowcoError(f"File {file_name} already exists.")

        if args.flowco:
            flowco_file = args.flowco
        else:
            flowco_file = base_name + ".flowco"
            Page.create(flowco_file)
            message(f"Created {flowco_file}.")

        page = Page.from_file(flowco_file)
        flowthon = page.to_flowthon(level)

        with open(file_name, "w", encoding="utf-8") as f:
            if ext == ".flowthon":
                print(flowthon.to_source(level), file=f)
            else:
                print(flowthon.to_json(level), file=f)

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

        base_name, ext = os.path.splitext(file_name)

        if ext != ".flowthon" and ext != ".flowjson":
            raise FlowcoError("File name must end with .flowthon or .flowjson")

        if not exists(file_name):
            raise FlowcoError(f"File {file_name} does not exist.")

        if args.flowco:
            flowco_file = args.flowco
        else:
            flowco_file = base_name + ".flowco"

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
            with open(file_name, "r", encoding="utf-8") as f:
                contents = f.read()

            if ext == ".flowthon":
                flowthon = FlowthonProgram.from_source(contents)
            else:
                flowthon = FlowthonProgram.from_json(json.loads(contents))

            if any("    ..." not in x.code for x in flowthon.nodes.values()):
                level = AbstractionLevel.code
            elif any(x.algorithm for x in flowthon.nodes.values()):
                level = AbstractionLevel.algorithm
            else:
                level = AbstractionLevel.spec

            page.merge_flowthon(flowthon, interactive=args.interactive)

            if ext == ".flowthon":
                contents = flowthon.to_source(level)
            else:
                contents = json.dumps(flowthon.to_json(level), indent=2)

            # copy original file to file.bak
            bak_file = base_name + ".bak"
            shutil.copy(file_name, bak_file)
            message(f"Created {bak_file}.")

            with open(file_name, "w", encoding="utf-8") as f:
                print(contents, file=f)

        page.save()

        html_file = base_name + ".html"
        with open(html_file, "w") as f:
            f.write(page.to_html())
            message(f"Wrote HTML to {html_file}")

        path = Path(base_name + ".html")
        file_url = path.absolute().as_uri()
        webbrowser.open(file_url)


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

        base_name, ext = os.path.splitext(file_name)

        if ext != ".flowthon" and ext != ".flowjson":
            raise FlowcoError("File name must end with .flowthon or .flowjson")

        flowco_file = base_name + ".flowco"

        page = Page.from_file(flowco_file)
        page = Page.create(flowco_file)
        page.clean()


class ToSourceCommand(Command):
    def __init__(self, subparsers):
        super().__init__(subparsers)
        self.parser = subparsers.add_parser(
            "to-source", help="Convert flowjson to flowthon"
        )
        self.parser.add_argument(
            "file_name",
            type=str,
            help="File to build",
        )

    def run(self, args):
        file_name = args.file_name

        base_name, ext = os.path.splitext(file_name)

        if ext != ".flowjson":
            raise FlowcoError("File name must end with .flowjson")

        if not exists(file_name):
            raise FlowcoError(f"File {file_name} does not exist.")

        with open(file_name, "r", encoding="utf-8") as f:
            contents = f.read()

        flowthon = FlowthonProgram.from_json(json.loads(contents))

        if any(x.code != None for x in flowthon.nodes.values()):
            level = AbstractionLevel.code
        elif any(x.algorithm != None for x in flowthon.nodes.values()):
            level = AbstractionLevel.algorithm
        else:
            level = AbstractionLevel.spec

        contents = flowthon.to_source(level)

        # copy original file to file.bak
        out_file = base_name + ".flowthon"
        bak_file = base_name + ".bak"
        shutil.copy(file_name, bak_file)
        message(f"Created {bak_file}.")

        with open(out_file, "w", encoding="utf-8") as f:
            print(contents, file=f)


class ToJsonCommand(Command):
    def __init__(self, subparsers):
        super().__init__(subparsers)
        self.parser = subparsers.add_parser(
            "to-json", help="Convert flowthon to flowjson"
        )
        self.parser.add_argument(
            "file_name",
            type=str,
            help="File to build",
        )

    def run(self, args):
        file_name = args.file_name

        base_name, ext = os.path.splitext(file_name)

        if ext != ".flowthon":
            raise FlowcoError("File name must end with .flowthon")

        if not exists(file_name):
            raise FlowcoError(f"File {file_name} does not exist.")

        with open(file_name, "r", encoding="utf-8") as f:
            contents = f.read()

        flowthon = FlowthonProgram.from_source(contents)

        if any(x.code != None for x in flowthon.nodes.values()):
            level = AbstractionLevel.code
        elif any(x.algorithm != None for x in flowthon.nodes.values()):
            level = AbstractionLevel.algorithm
        else:
            level = AbstractionLevel.spec

        contents = json.dumps(flowthon.to_json(level), indent=2)

        # copy original file to file.bak
        out_file = base_name + ".flowjson"
        bak_file = base_name + ".bak"
        shutil.copy(file_name, bak_file)
        message(f"Created {bak_file}.")

        with open(out_file, "w", encoding="utf-8") as f:
            print(contents, file=f)


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
        stopper=Stopper(),
        shells=PythonShells(),
        filesystem=SessionFileSystem(f"file://{os.getcwd()}"),
    )

    try:
        with session.get("stopper", Stopper):
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

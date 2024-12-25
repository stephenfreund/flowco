import argparse
from genericpath import exists
import inspect
import os
import re
import sys
import textwrap
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import markdown

from flowco.dataflow.phase import Phase
from flowco.builder.build import BuildEngine
from flowco.page.ama import AskMeAnything
from flowco.page.page import Page
from flowco.pythonshell.shells import PythonShells
from flowco.session.session import session
from flowco.session.session_file_system import SessionFileSystem
from flowco.util.config import AbstractionLevel, config
from flowco.util.costs import CostTracker, call_count, total_cost
from flowco.util.output import Output, error, log, message, logger

from flowco.util.errors import FlowcoError
from flowco.util.stoppable import Stoppable

### Base


class PagelessCommand(ABC):
    @abstractmethod
    def __init__(self, subparsers):
        pass

    @abstractmethod
    def run(self, args):
        pass


class Command(ABC):
    @abstractmethod
    def __init__(self, subparsers):
        pass

    @abstractmethod
    def run(self, page: Page, args: argparse.Namespace):
        pass


### Page Commands


class CreateCommand(PagelessCommand):
    def __init__(self, subparsers):
        super().__init__(subparsers)
        self.parser = subparsers.add_parser("create", help="Create a new Page")
        self.parser.add_argument(
            "--diagram",
            type=str,
            help="The image file to start with for the initial diagram",
        )
        self.parser.add_argument("files", metavar="files", type=str, nargs="*")

    def run(self, args):
        if os.path.exists(args.page):
            raise FlowcoError(f"Page {args.page} already exists.")

        page = Page.create(file_name=args.page)
        log(f"Created page {args.page}")


class AddCommand(Command):
    def __init__(self, subparsers):
        super().__init__(subparsers)
        self.parser = subparsers.add_parser("add", help="Add files to page")
        self.parser.add_argument("files", metavar="files", type=str, nargs="+")

    def run(self, page, args):
        for file_name in args.files:
            if not os.path.exists(file_name):
                raise FlowcoError(f"File {file_name} does not exist.")
            with open(file_name, "r") as f:
                page.add_table(file_name)
        log(f"Added files to page {args.page}")


class RemoveCommand(Command):
    def __init__(self, subparsers):
        super().__init__(subparsers)
        self.parser = subparsers.add_parser("remove", help="Remove files from page")
        self.parser.add_argument("files", metavar="files", type=str, nargs="+")

    def run(self, page, args):
        for f in args.files:
            page.remove_table(f)
        log(f"Removed files from page {args.page}")


class ResetCommand(Command):
    def __init__(self, subparsers):
        super().__init__(subparsers)
        self.parser = subparsers.add_parser("reset", help="Hard reset of page")

    def run(self, page, args):
        page.reset()
        page.save()


class AmaCommand(Command):
    def __init__(self, subparsers):
        self.parser = subparsers.add_parser(
            "ama", help="Call the ama method on the page with additional arguments"
        )
        self.parser.add_argument(
            "args",
            nargs=argparse.REMAINDER,
            help="Additional arguments (English sentence)",
        )
        super().__init__(subparsers)

    def run(self, page, args):
        ama = AskMeAnything(page)
        for x in ama.complete(" ".join(args.args)):
            print(x, end="")
        print()
        print("---")
        print(ama.last_message())


class CleanCommand(Command):
    def __init__(self, subparsers):
        super().__init__(subparsers)
        self.parser = subparsers.add_parser(
            "clean",
            help="Remove outputs and compiled files from page",
        )
        self.parser.add_argument(
            "--node_id", type=str, help="Node to clean", default=None
        )

    def run(self, page, args):
        page.clean(args.node_id)
        log(f"Cleaned page {args.page}")


class ExportCommand(Command):
    def __init__(self, subparsers):
        super().__init__(subparsers)
        self.parser = subparsers.add_parser(
            "export", help="Convert page to editable file"
        )
        self.parser.add_argument(
            "--algorithm", help="show algorithm", action="store_true"
        )
        self.parser.add_argument("--code", help="show code", action="store_true")
        self.parser.add_argument(
            "--force", help="overwrite file if exsts", action="store_true"
        )

        self.parser.add_argument(
            "--file_name",
            type=str,
            help="File to merge into the page",
        )

    def run(self, page, args):
        if args.code:
            level = AbstractionLevel.code
        elif args.algorithm:
            level = AbstractionLevel.algorithm
        else:
            level = AbstractionLevel.spec
        file_name = args.file_name or page.file_name.replace(".flowco", ".flowthon")
        if exists(file_name) and not args.force:
            raise FlowcoError(f"File {file_name} already exists.")
        page.to_flowthon(level, file_name)
        message(f"Exported {page.file_name} to {file_name}.")


class MergeCommand(Command):
    def __init__(self, subparsers):
        super().__init__(subparsers)
        self.parser = subparsers.add_parser(
            "merge", help="Merge editable file into page"
        )
        self.parser.add_argument(
            "file_name",
            type=str,
            help="File to merge into the page",
        )

    def run(self, page, args):
        if not exists(args.file_name):
            raise FlowcoError(f"File {args.file_name} does not exist.")
        page.merge_flowthon(args.file_name)
        message(f"Merged {args.file_name} into {page.file_name}.")


class HtmlCommand(Command):
    def __init__(self, subparsers):
        self.parser = subparsers.add_parser("html", help="Make an HTML page")
        super().__init__(subparsers)

    def run(self, page, args):
        md = page.to_markdown()
        html_content = markdown.markdown(
            md,
            extensions=[
                "extra",  # Includes several extensions like tables, fenced code, etc.
                "codehilite",  # Adds syntax highlighting to code blocks
                "toc",  # Generates a table of contents
                "sane_lists",  # Improves list handling
                "smarty",  # Converts quotes and dashes to smart quotes and dashes
            ],
        )

        with open(os.path.join(os.path.dirname(__file__), "template.html"), "r") as f:
            html = f.read()

        full_html = html.format(content=html_content, title=page.file_name)

        with open(page.file_name + ".html", "w") as f:
            f.write(full_html)
            message(f"Wrote HTML to {page.file_name}.html")
            os.system(f"open {page.file_name}.html")


class BuildCommand(Command):
    build_target_to_phase: Dict[str, Phase] = {
        "clean": Phase.clean,
        "requirements": Phase.requirements,
        "algorithm": Phase.algorithm,
        "code": Phase.code,
        "run": Phase.run_checked,
    }

    always_force_to_run = ["run", "all"]

    def __init__(self, subparsers):
        super().__init__(subparsers)
        self.parser = subparsers.add_parser("build", help="Run a compilation pass")
        self.parser.add_argument("--force", help="force", action="store_true")
        self.parser.add_argument("--repair", help="repair", action="store_true")
        self.parser.add_argument(
            "--node_id", type=str, help="Node to build", default=None
        )
        self.parser.add_argument(
            "target",
            choices=BuildCommand.build_target_to_phase.keys(),
            help="Target to build",
        )

    def run(self, page: Page, args):
        builder = BuildEngine.get_builder()
        log(f"builder:\n{textwrap.indent(str(builder), prefix=' ' * 4)}\n")

        target_phase: Phase = BuildCommand.build_target_to_phase[args.target]

        if args.force or args.target in BuildCommand.always_force_to_run:
            p = builder.passes_by_target[target_phase]
            page.clean(phase=p.required_phase())

            if target_phase in [
                Phase.requirements,
                Phase.algorithm,
                Phase.code,
                Phase.sanity_checks,
                Phase.unit_tests,
            ]:
                page.invalidate_build_cache(phase=target_phase, node_id=args.node_id)
            page.save()

        for _ in page.build(builder, target_phase, args.repair, args.node_id):
            pass


def class_name_to_key(name):
    base_name = name[:-7]  # Remove "Command" suffix
    key = re.sub(r"(?<!^)(?=[A-Z])", "-", base_name).lower()
    return key


def create_commands(
    subparsers: argparse._SubParsersAction,
) -> Dict[str, Command | PagelessCommand]:
    commands = {}
    current_module = sys.modules[__name__]
    for name, obj in inspect.getmembers(current_module, inspect.isclass):
        if name.endswith("Command") and obj.__module__ == __name__:
            if not inspect.isabstract(obj):
                key = class_name_to_key(name)
                commands[key] = obj(subparsers)
    return commands


def main_core(page: Optional[Page], argv: List[str]):
    parser = config.parser()

    if page is None:
        parser.add_argument("page", type=str, help="Name of the page")

    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")
    subparsers.required = True  # Ensures that a subcommand is required

    commands = create_commands(subparsers=subparsers)

    args = parser.parse_args(argv)

    command = commands.get(args.command)

    if command:
        with logger(f"Running `{args.command}` command"):
            if isinstance(command, PagelessCommand):
                assert page is None, "Pageless command should not have a page"
                command.run(args)
            else:
                if page is None:
                    page = Page.from_file(args.page)
                args.page = page.file_name
                command.run(page, args)
    else:
        raise FlowcoError(f"Unknown command: {args.command}")


def main(page: Optional[Page] = None, argv: List[str] = sys.argv[1:]):
    session.set(
        output=Output(),
        costs=CostTracker(),
        filesystem=SessionFileSystem(f"file://{os.getcwd()}"),
    )

    try:
        with Stoppable():
            main_core(page=page, argv=argv)
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

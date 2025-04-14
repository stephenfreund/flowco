import subprocess
import os

from flowco.util.errors import FlowcoError


def command_line():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Flowco UI. If no path is provided, the current directory will be used."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "-d", "--dev", action="store_true", help="Run in development mode"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to the Flowco project (default: current directory)",
    )

    return parser


def main():

    args = command_line().parse_args()

    if os.environ.get("OPENAI_API_KEY", None) is None:
        raise FlowcoError("You must set the OPENAI_API_KEY environment variable")

    ui_path = os.path.join(os.path.dirname(__file__), "ui")

    env = os.environ.copy()

    if args.dev:
        env["FLOWCO_DEV"] = "1"

    command = [
        "streamlit",
        "run",
        "--logger.level",
        "error",
        f"{ui_path}/ui_main.py",
        "--",
    ]

    if not args.verbose:
        command += ["--quiet"]
    command += [os.path.abspath(args.path)]

    cwd = os.path.join(os.path.dirname(__file__))
    subprocess.run(command, env=env, cwd=cwd)


if __name__ == "__main__":
    main()

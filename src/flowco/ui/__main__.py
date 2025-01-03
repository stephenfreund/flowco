import subprocess
import sys
import os

from flowco.util.errors import FlowcoError


def main():
    current_directory = os.getcwd()
    ui_path = os.path.dirname(__file__)
    os.chdir(current_directory)

    os.system(f"streamlit run {ui_path}/ui_main.py -- {' '.join(sys.argv[1:])}")


def dev():
    current_directory = os.getcwd()
    os.chdir(current_directory)

    # Add the additional directory to PYTHONPATH
    additional_path = os.path.join(
        os.path.dirname(__file__), "../../../mxgraph_component"
    )
    ui_path = os.path.join(os.path.dirname(__file__))
    pythonpath = os.environ.get("PYTHONPATH", "")
    if pythonpath:
        pythonpath = additional_path + os.pathsep + pythonpath
    else:
        pythonpath = additional_path

    # Update environment variables
    env = os.environ.copy()

    env["PYTHONPATH"] = f"{additional_path}:{env.get('PYTHONPATH', '')}"
    env["FLOWCO_DEV"] = "1"
    # env["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") 

    # Build the command
    command = [
        "streamlit",
        "run",
        f"{ui_path}/ui_main.py",
        "--",
    ] + sys.argv[1:]

    # Print the pythonpath and the command for debugging purposes
    print(env["PYTHONPATH"])
    print(" ".join(command))

    # subprocess.run(["streamlit", "config", "show"], env=env)

    # Run the command using subprocess
    subprocess.run(command, env=env)


if __name__ == "__main__":
    # Gotta do this somewhere, lest a missing key exception gets generated inside a library
    # that doesn't report it to us properly.
    if os.environ.get("OPENAI_API_KEY", None) is None:
        raise FlowcoError("You must set the OPENAI_API_KEY environment variable")

    main()

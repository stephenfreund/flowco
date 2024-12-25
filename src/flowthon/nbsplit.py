import argparse
import nbformat
from copy import deepcopy


def split_notebook_by_heading_level(notebook, heading_level):
    """
    Splits a Jupyter Notebook into multiple notebooks at each specified heading level.

    Parameters:
    - notebook (nbformat.NotebookNode): The original Jupyter Notebook object.
    - heading_level (int): The Markdown heading level to split at (e.g., 1 for '#', 2 for '##').

    Returns:
    - List[nbformat.NotebookNode]: A list of new notebook objects split at the specified headings.
    """
    if heading_level < 1 or heading_level > 6:
        raise ValueError(
            "heading_level must be a positive integer (1 for '#', 2 for '##', etc.)."
        )

    # Convert heading_level to corresponding Markdown heading string
    heading_prefix = "#" * heading_level

    # Initialize list to hold the new notebooks
    new_notebooks = []

    # Initialize the first new notebook
    current_nb = deepcopy(notebook)
    current_nb["cells"] = []

    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown":
            # Check if the cell starts with the specified heading level
            lines = cell["source"].splitlines()
            if lines:
                first_line = lines[0].strip()
                if first_line.startswith(heading_prefix + " "):
                    # If current notebook has cells, save it and start a new one
                    if current_nb["cells"]:
                        new_notebooks.append(current_nb)
                        current_nb = deepcopy(notebook)
                        current_nb["cells"] = []
        # Append the cell to the current notebook
        current_nb["cells"].append(deepcopy(cell))

    # Append the last notebook if it has any cells
    if current_nb["cells"]:
        new_notebooks.append(current_nb)

    return new_notebooks


def main():
    parser = argparse.ArgumentParser(
        description="Split a Jupyter Notebook by heading level."
    )
    parser.add_argument(
        "--split",
        type=int,
        help='The Markdown heading level to split at (e.g., 1 for "#", 2 for "##")',
        default=2,
    )
    parser.add_argument("notebook", type=str, help="Path to the Jupyter Notebook file")
    args = parser.parse_args()

    # Path to the original notebook
    original_notebook_path = args.notebook

    # Load the original notebook
    with open(original_notebook_path, "r", encoding="utf-8") as f:
        original_nb = nbformat.read(f, as_version=4)

    # Split the notebook
    split_nbs = split_notebook_by_heading_level(original_nb, args.split)

    # Save the split notebooks
    prefix = original_notebook_path.split(".ipynb")[0]
    for idx, nb in enumerate(split_nbs, start=1):
        filename = f"{prefix}_{idx}.ipynb"
        with open(filename, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"Saved: {filename}")


# Example Usage
if __name__ == "__main__":
    main()

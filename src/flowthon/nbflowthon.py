import argparse
from os import write
import nbformat
from openai import OpenAI

prompt = """\
Here is an example Flowthon description of a dataflow graph for a data science computation. 
The graph consists of three nodes: `Array-Numbers`, `Plot-Histogram`, and `Compute-Mean`. 
Each node represents a specific computation or visualization task, and the edges between nodes indicate the dependencies between them. 
The description includes the requirements for each node, as well as the dependencies between nodes.
It also includes the csv files that are used in the computation.
```
{
  "tables": {
    "tables": [
        'file.csv',
        'file2.csv'
    }
  },
  "graph": {
    "Array-Numbers": {
      "uses": [],
      "label": "make array of 500 numbers from normal distribution",
      "requirements": [
        "array_numbers_result is a NumPy array.",
        "array_numbers_result has 500 elements.",
        "Each element in array_numbers_result is sampled from a normal distribution with a default mean of 0 and a standard deviation of 1."
      ]
    },
    "Plot-Histogram": {
      "uses": [
        "Array-Numbers"
      ],
      "label": "Plot histogram",
      "requirements": [
        "A histogram of the values in array_numbers_result is plotted to visualize the distribution.",
        "The histogram includes a title 'Histogram of Normally Distributed Values'.",
        "X-axis is labeled 'Value'.",
        "Y-axis is labeled 'Frequency'.",
        "Histogram bars are displayed in orange color."
      ]
    },
    "Compute-Mean": {
      "uses": [
        "Array-Numbers"
      ],
      "label": "Compute mean of array",
      "requirements": [
        "The mean of the values in array_numbers_result is computed.",
        "The result is a single floating-point number."
      ]
    }
  }
}
```

Convert the following Jupyter notebook into a Flowthon description of the dataflow graph 
induced by the notebook.  

The requirements are a list of postconditions that will hold 
for the computed value, provided the inputs satisfy their requirements.  

Use the name `the_result` for the computed value.  The computed value
must be numbers, strings, lists, dictionaries, dataframes, arrays, or series.

The requirements should be written in
plain English and should be as specific as possible.

The label field should
describe the computation or visualization that the node represents.

The uses field lists the nodes that are required as inputs for the computation.

* Include any constants or data that are necessary for the computation.
* Include csv filenames that appear in the notebook, but no other files.
* Node names are two words separated by a hyphen, e.g., `Array-Numbers`.  They must be unique.

Provide only the Flowthon description and no other text or
prose.
"""


def convert_notebook_to_flowthon(model, original_nb, verbose=False):
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt + f"\n```\n{original_nb}\n```\n"}],
        stream=True,
    )

    flowthon = ""
    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content is not None:
            flowthon += content
            if verbose:
                print(chunk.choices[0].delta.content, end="")
        else:
            if verbose:
                print()

    lines = flowthon.splitlines()
    text = "\n".join(lines[1:-1])
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Convert a notebook into a Flowthon program."
    )
    parser.add_argument(
        "--model", type=str, help="The model to use for conversion", default="o1-mini"
    )
    parser.add_argument("--verbose", help="Print the completion", action="store_true")
    parser.add_argument("notebook", type=str, help="Path to the Jupyter Notebook file")
    args = parser.parse_args()

    # Path to the original notebook
    original_notebook_path = args.notebook

    # Load the original notebook
    with open(original_notebook_path, "r", encoding="utf-8") as f:
        original_nb = nbformat.read(f, as_version=4)

    flowthon = convert_notebook_to_flowthon(args.model, original_nb, args.verbose)

    # Save the split notebooks
    prefix = original_notebook_path.split(".ipynb")[0]
    filename = f"{prefix}.flowthon"
    with open(filename, "w", encoding="utf-8") as f:
        print(flowthon, file=f)
    print(f"Saved: {filename}")


# Example Usage
if __name__ == "__main__":
    main()

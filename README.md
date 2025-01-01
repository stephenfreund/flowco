# Flowco

Flowco needs a better description.

## Web

[https://go-flow.co](https://go-flow.co)

Log in with Google credentials.  All files are stored on AWS S3 partition.  This uses my own OpenAI key for now...

## Local

### Conda or other virtual env

Definitely use a conda virtual environment or some other virtual environment, and Python 3.11.  
Should work with other versions, but that matches the web service environtment.

### Installation

> [!IMPORTANT]
>
> Flowco currently needs to be connected to an [OpenAI account](https://openai.com/api/). _Your account will need to have a positive balance for this to work_ ([check your balance](https://platform.openai.com/account/usage)). If you have never purchased credits, you will need to purchase at least \$1 in credits (if your API account was created before August 13, 2023) or \$0.50 (if you have a newer API account) in order to have access to GPT-4, which Flowco uses. [Get a key here.](https://platform.openai.com/account/api-keys)
>
> Once you have an API key, set it as an environment variable called `OPENAI_API_KEY`.
>
> ```bash
> export OPENAI_API_KEY=<your-api-key>
> ```

Clone the repo and install with `make`: 

```bash
make
```

This installs a bunch of normal packages through a hatchling `pyproject.toml`, and then also a local wheel for the MXGraph component.

### Command line:

* Make a copy of the `test` directory so you don't inadvertently 
  commit changes to the examples.
    ```bash
    cp -r test /tmp/test
    ```

* Run `flowco` from the directory containing the `.flowco` file:
    ```bash
    cd /tmp/test/finch
    flowco finch-small.flowco build run
    ```

* View the output:
    ```bash
    flowco finch-small.flowco html
    ```

* Reset the file to clear out all generated node components:
    ```bash
    flowco finch-small.flowco reset
    ```

### GUI

* From the root of the flowco repository, run this command with the directory if the flowco files you want to use:
    ```bash
    flowco-ui /tmp/test/finch
    ```

    You will get a few new demo files in that directory.  They have some simple how-to instructions that may help you
    get started.


### Flowthon: For those who like source code.

* Make the source code for an existing file:
    ```bash
    flowco finch-small.flowco export
    ```

    Or a new one:
        ```bash
        flowthon create test.flowthon
        ```

* Edit the flowthon file in your favorite editor.

* Run it:

    ```bash
    flowthon run finch-small.flowthon
    ```

    **Note:** This will modify your flowthon file in place to keep it in sync with the dataflow graph.  Be sure to reload it before editing.

* Base structure of file:

    ```
    def create_1000():
        """
        Label

        # Requirements
        - ...

        # Algorithm
        - ...

        # Assertions
        - ...
        """
        ...

    def plot_purple(create_1000):
        """
        ...
        """
        ...
    ```

    The `...` for the code is necessary.  The other parts can be edited/omitted however you want.

* To see all parts in the flowthon file, rerun `export` as follows.

    ```bash
    flowco finch-small.flowco export --code
    ```


## "Sketchy" Behavior or Bugs?

If you see a bug -- hahaha, **when** you see a bug -- please send me the Flowco file and the `logging.txt` file from the working directory, or upload to a github issue...

## Known Issues

* So many I can't even begin to list them...


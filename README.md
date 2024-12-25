# Flowco

Flowco needs a better description.

## Conda or other virtual env

Definitely use a conda virtual environment or some other virtual environment, and Python 3.11.  Should work with other versions, but that matches what I'm using on all my machines.

## Installation

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

(Using `make` because I need to install a bunch of normal packages through a hatchling `pyproject.toml`, and then also a local wheel, which is not supporting in hatchling...)


### GUI:

* Make a copy of the `test` directory so you don't inadvertently 
  commit changes to the examples.

* Assuming you copied the whole thing to `/tmp`:
    ```bash
    cd /tmp/tests/finch
    flowco-ui --model=gpt-4o finch-demo.flowco
    ```

* Click "Run"

* Click the buttons for abstraction level while selecting different nodes.

* Same with output and description buttons

* Click "Run"

* Just use Ctrl-C in the terminal to exit the program.

### More Tasks

* Three initial tasks to try.  `They are in test/tasks`.  See the `task.txt` 
file in each directory:

* finch: Yep, birds again.  An existing sketch, with a task of making several edits.
* temps: Greenland temperature data, with a plotting / linear regression question
* nivea : The first user study task from one of the MSR papers.  Basic data cleaning and wranging.

Run as flowco-ui <name>.flowco.



### "Sketchy" Behavior or Bugs?

If you see a bug -- hahaha, **when** you see a bug -- please send me the JSON file and the `logging.txt` file from the working directory, or upload to a github issue...

### Known Issues

* Close old browser windows -- Streamlit has some wonky behavior if you leave old instances of the gui open.  Most notably, it may hang while building.
* So many I can't even begin to list them...


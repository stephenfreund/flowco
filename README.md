

# Flowco
![flowco](https://github.com/user-attachments/assets/9591f546-ef52-4c16-999a-ca9e2a149586)

Flowco is designed to run as a web service.  
However, it can also be run locally by following these instructions.  

### Configuration

* Use a conda virtual environment or some other virtual environment.
* You **must** use Python 3.11.
* Ensure `dot` is on your path.
* Ensure `npm`, `node`, and `make` are on your path to build from scratch.

### API Key

> [!IMPORTANT]
>
> Flowco currently needs to be connected to an [OpenAI account](https://openai.com/api/). _Your account will need to have a positive balance for this to work_ ([check your balance](https://platform.openai.com/account/usage)). [Get a key here.](https://platform.openai.com/account/api-keys)
>
> Once you have an API key, set it as an environment variable called `OPENAI_API_KEY`.
>
> ```bash
> export OPENAI_API_KEY=<your-api-key>
> ```

### Installing

#### From the source

Clone the repo and install:

```bash
pip3 install src/mxgraph_component/prebuilt/mxgraph_component-0.0.1-py3-none-any.whl -e .
```

This installs a bunch of normal packages, a custom component for Streamlit, and then Flowco as an 
editable package that you can run locally (rather than as the web service).

#### From scratch

To rebuild the graph component from scratch, run:

```bash
make
```

### Running

On the command line, run `flowco`, passing it the directory in which to store its files.  That directory
should already exist:

```bash
mkdir /tmp/example
flowco /tmp/example
```

The first time you run, it may take 15-20 seconds to bring up the web page properly due to a timing issue
with the libraries.  It should launch more quickly after that.

Use `-v` to turn on logging.
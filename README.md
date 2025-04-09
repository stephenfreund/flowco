

# Flowco
![flowco](https://github.com/user-attachments/assets/9591f546-ef52-4c16-999a-ca9e2a149586)

Flowco is designed to run as a web service.  
However, it can also be run locally by following these instructions.  

### Configuration

* Use a conda virtual environment or some other virtual environment.
* Use Python 3.11.  
* Ensure `dot` is on your path.
* Ensure `npm` and `node` are on your path to build the project from the source.

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
pip3 install -e .
```

This installs a bunch of normal packages and a custom component for Streamlit.  

#### From a prebuilt wheel

You can alternatively install
a prebuilt wheel:

```bash
pip3 install dist/flowco-py3-none-any.whl 
```

### Running

On the command line, run `flowco`, passing it the directory in which to store its files.  That directory
should already exist:

```bash
mkdir /tmp/example
flowco /tmp/example
```

Use `-v` to turn on logging.


# Flowco
![flowco](https://github.com/user-attachments/assets/9591f546-ef52-4c16-999a-ca9e2a149586)

## Web

[https://go-flow.co](https://go-flow.co)

## Local Install

### Conda or other virtual env

* Definitely use a conda virtual environment or some other virtual environment.
* Use Python 3.11.  

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

### Building

Clone the repo and install with `make`: 

```bash
make
```

This installs a bunch of normal packages through a hatchling `pyproject.toml`, and then also a local wheel for the MXGraph component.

### Running

On the command line, run `flowco`, passing it the directory in which to store its files.  That directory
should already exist:
```bash
mkdir /tmp/example
flowco /tmp/example
```


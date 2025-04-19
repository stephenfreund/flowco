

# Flowco
by [Emery Berger](https://emeryberger.com), [Stephen Freund](https://www.cs.williams.edu/~freund/index.html), [Eunice Jun](http://eunicemjun.com/), [Brooke Simon](https://www.linkedin.com/in/brooke-alexandra-simon/) (ordered alphabetically)

![flowco](https://github.com/user-attachments/assets/9591f546-ef52-4c16-999a-ca9e2a149586)

#### Watch Flowco in Action!

| Short Demo    | Tutorial                                                                                  |
|---------------|----------------------------------------------------------------------------------------------|
| [![](https://img.youtube.com/vi/qmMeMIrhtPs/0.jpg)](https://www.youtube.com/watch?v=qmMeMIrhtPs) | [![](https://img.youtube.com/vi/q0eAJv1vhAQ/0.jpg)](https://www.youtube.com/watch?v=q0eAJv1vhAQ) |


For technical details, see our arXiv paper, [_Flowco: Rethinking Data Analysis in the Age of LLMs_.](https://arxiv.org/abs/2403.16354)


# Web Service

Flowco is designed to run as a web service but can also be run locally by following the instructions below.  

# Local Installation

[![PyPI Latest Release](https://img.shields.io/pypi/v/chatdbg.svg)](https://pypi.org/project/chatdbg/)

### Configuration

* Use a conda virtual environment or some other virtual environment.
* Use Python 3.11+.
* Ensure `dot` is on your path.

### API Key

> [!IMPORTANT]
>
> Flowco needs to be connected to an [OpenAI account](https://openai.com/api/). _Your account will need to have a positive balance for this to work_ ([check your balance](https://platform.openai.com/account/usage)). [Get a key here.](https://platform.openai.com/account/api-keys)
>
> Once you have an API key, set it as an environment variable called `OPENAI_API_KEY`.
>
> ```bash
> export OPENAI_API_KEY=<your-api-key>
> ```

### Installing

### Latest release

With pip:
```bash
pip3 install flowco
```

#### From the source

Or clone the repo and install as an editable package.
```bash
pip3 install -e .
```
This installs a bunch of normal packages, a custom component for Streamlit, and then Flowco as an 
editable package that you can run locally (rather than as the web service).

### Running

On the command line, run `flowco`, passing it the directory in which to store its files.  That directory
should already exist:

```bash
mkdir /tmp/example
flowco /tmp/example
```

The first time you run, it may take 15-20 seconds to bring up the web page.  It should launch more quickly after that. 
Flowco will intially open a `welcome.flowco` graph.  Follow the instructions in the right-hand panel to get started.  Then proceed through the numbered tutorials to experiment with additional features.


Use `-v` to turn on logging.

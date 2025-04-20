![](https://github.com/stephenfreund/flowco/raw/main/media/flowco-small-logo.png)

# Flowco 
by [Emery Berger](https://emeryberger.com), [Stephen Freund](https://www.cs.williams.edu/~freund/index.html), [Eunice Jun](http://eunicemjun.com/), [Brooke Simon](https://www.linkedin.com/in/brooke-alexandra-simon/) (ordered alphabetically)

[![Flowco](https://img.shields.io/badge/Flowco-Online-brightgreen)](https://go-flow.co)
[![PyPI Latest Release](https://img.shields.io/pypi/v/flowco.svg)](https://pypi.org/project/flowco/)

Flowco is a system for authoring data analysis workflows with LLM assistance at every stage of the process.  Flowco employs a dataflow programming model that serves as a foundation for reliable LLM-centric programming.

![](https://github.com/stephenfreund/flowco/raw/main/media/geyser.png)

## Watch Flowco in Action!

| Demo Video | Tutorial Video | Exploratoration | Multiverse Analysis | Logistic Regression |
|------------|----------|----------|--|--|
| <a href="https://www.youtube.com/watch?v=qmMeMIrhtPs"><img src="https://img.youtube.com/vi/qmMeMIrhtPs/0.jpg" width="1200"></a> | <a href="https://www.youtube.com/watch?v=q0eAJv1vhAQ"><img src="https://img.youtube.com/vi/q0eAJv1vhAQ/0.jpg" width="1200"></a> |  <img alt="finch-3" src="https://github.com/stephenfreund/flowco//raw/main/media/finch.png" /> |  <img  alt="mortgage-wide" src="https://github.com/stephenfreund/flowco/raw/main/media/multiverse.png" /> |  <img alt="logistic-full" src="https://github.com/stephenfreund/flowco/raw/main//media/logistic.png" /> |


For technical details, see our arXiv paper, [_Flowco: Rethinking Data Analysis in the Age of LLMs_](https://github.com/stephenfreund/flowco/raw/main/media/flowco-arxiv-submission.pdf).

## Web Service

You can try Flowco on the web [here](https://go-flow.co).  

> [!NOTE]
> This web service is intended for demonstration and experimentation only.
> It should scale to a modest number of
> users, but if it is slow or unresponsive, please try again later or install locally.

## Local Installation

#### Configuration

* Use a conda virtual environment or some other virtual environment.
* Use Python 3.11+.
* Ensure [`dot`](https://graphviz.org/) is on your path.

#### OpenAI API Key

> [!IMPORTANT]
>
> Flowco needs to be connected to an [OpenAI account](https://openai.com/api/). _Your account will need to have a positive balance for this to work_ ([check your balance](https://platform.openai.com/account/usage)). [Get a key here.](https://platform.openai.com/account/api-keys)
>
> Once you have an API key, set it as an environment variable called `OPENAI_API_KEY`.
>
> ```bash
> export OPENAI_API_KEY=<your-api-key>
> ```

#### Installing

##### From Pypi [Recommended]

With pip:
```bash
pip3 install flowco
```

##### From Source

Or clone the repo and install as an editable package.
```bash
pip3 install -e .
```
This installs a bunch of normal packages, a custom component for Streamlit, and then Flowco as an 
editable package that you can run locally (rather than as the web service).

#### Running

On the command line, run `flowco`, passing it the directory in which to store its files.  That directory
should already exist:

```bash
mkdir /tmp/example
flowco /tmp/example
```

The first time you run, it may take 15-20 seconds to bring up the web page.  It should launch more quickly after that. 
Flowco will intially open a `welcome.flowco` graph.  Follow the instructions in the right-hand panel to get started.  Then proceed through the numbered tutorials to experiment with additional features.


Use `-v` to turn on logging.

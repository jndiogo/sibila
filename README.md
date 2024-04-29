# Sibila

Extract structured data from remote or local LLM models. Predictable output is important for serious use of LLMs.

- Query structured data into Pydantic objects, dataclasses or simple types.
- Access remote models from OpenAI, Anthropic, Mistral AI and other providers.
- Use local models like Llama-3, Phi-3, OpenChat or any other GGUF file model.
- Besides structured extraction, Sibila is also a general purpose model access library, to generate plain text or free JSON results, with the same API for local and remote models.
- Model management: download models, manage configuration, quickly switch between models.

No matter how well you craft a prompt begging a model for the output you need, it can always respond something else. Extracting structured data can be a big step into getting predictable behavior from your models.

See [What can you do with Sibila?](https://jndiogo.github.io/sibila/what/)

To extract structured data from a local model:

``` python
from sibila import Models
from pydantic import BaseModel

class Info(BaseModel):
    event_year: int
    first_name: str
    last_name: str
    age_at_the_time: int
    nationality: str

model = Models.create("llamacpp:openchat")

model.extract(Info, "Who was the first man in the moon?")
```

Returns an instance of class Info, created from the model's output:

``` python
Info(event_year=1969,
     first_name='Neil',
     last_name='Armstrong',
     age_at_the_time=38,
     nationality='American')
```

Or to use a remote model like OpenAI's GPT-4, we would simply replace the model's name:

``` python
model = Models.create("openai:gpt-4")

model.extract(Info, "Who was the first man in the moon?")
```

If Pydantic BaseModel objects are too much for your project, Sibila supports similar functionality with Python dataclass. Also includes asynchronous access to remote models.




## Docs

[The docs explain](https://jndiogo.github.io/sibila/) the main concepts, include examples and an API reference.


## Installation

Sibila can be installed from PyPI by doing:

```
pip install --upgrade sibila
```

See [Getting started](https://jndiogo.github.io/sibila/installing/) for more information.



## Examples

The [Examples](https://jndiogo.github.io/sibila/examples/) show what you can do with local or remote models in Sibila: structured data extraction, classification, summarization, etc.



## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/jndiogo/sibila/blob/main/LICENSE) file for details.


## Acknowledgements

Sibila wouldn't be be possible without the help of great software and people:

- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [OpenAI Python API](https://github.com/openai/openai-python)
- [TheBloke (Tom Jobbins)](https://huggingface.co/TheBloke) and [Hugging Face model hub](https://huggingface.co/)

Thank you!


## Sibila?

Sibila is the Portuguese word for Sibyl. [The Sibyls](https://en.wikipedia.org/wiki/Sibyl) were wise oracular women in ancient Greece. Their mysterious words puzzled people throughout the centuries, providing insight or prophetic predictions, "uttering things not to be laughed at".

![Michelangelo's Delphic Sibyl, Sistine Chapel ceiling](https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/DelphicSibylByMichelangelo.jpg/471px-DelphicSibylByMichelangelo.jpg)

Michelangelo's Delphic Sibyl, in the Sistine Chapel ceiling.


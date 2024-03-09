# Sibila

Extract structured data from remote or local file LLM models.

- Extract data into Pydantic objects, dataclasses or simple types.
- Same API for local file models and remote OpenAI models.
- Model management: download models, manage configuration and quickly switch between models.
- Tools for evaluating output across local/remote models, for chat-like interaction and more.

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

Or to use OpenAI's GPT-4, we would simply replace the model's name:

``` python
model = Models.create("openai:gpt-4")

model.extract(Info, "Who was the first man in the moon?")
```

If Pydantic BaseModel objects are too much for your project, Sibila supports similar functionality with Python dataclass.



## Docs

[The docs explain](https://jndiogo.github.io/sibila/) the main concepts, include examples and an API reference.


## Installation

Sibila can be installed from PyPI by doing:

```
pip install sibila
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

Sibila is the Portuguese word for Sibyl. [The Sibyls](https://en.wikipedia.org/wiki/Sibyl) were wise oracular women in ancient Greece. Their mysterious words puzzled people throughout the centuries, providing insight or prophetic predictions.

![Michelangelo's Delphic Sibyl, Sistine Chapel ceiling](https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/DelphicSibylByMichelangelo.jpg/471px-DelphicSibylByMichelangelo.jpg)

Michelangelo's Delphic Sibyl, in the Sistine Chapel ceiling.


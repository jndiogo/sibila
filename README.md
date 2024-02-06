# Sibila

Extract structured information from LLM models, using a common API to access remote models like GPT-4 or local GGUF quantized models with llama.cpp.

- Query structured information into Pydantic BaseModel objects or typed Python dicts.
- Use the same API for local and remote models.
- Thread-based interaction with chat/instruct fine-tuned models.
- Compare output across local/remote models with included utilities, text or CSV output.
- Model management directory: manage models and their configurations and quickly switch between models.
- Automatic chat templates: identifies and uses the right templates for each model.

With Sibila you can extract structured data from a local model like OpenChat-3.5 with 7B params:

```python
from sibila import (LlamaCppModel, OpenAIModel)
from pydantic import BaseModel, Field

class Info(BaseModel):
    event_year: int
    first_name: str
    last_name: str
    age_at_the_time: int
    nationality: str

openchat = LlamaCppModel("models/openchat-3.5-1210.Q5_K_M.gguf")

openchat.query_pydantic(Info,
                        "Just be helpful.", # system message
                        "Who was the first man in the moon?")
```

Outputs an object of class Info, initialized with the model's output:

```python
Info(event_year=1969,
     first_name='Neil',
     last_name='Armstrong',
     age_at_the_time=38,
     nationality='American')
```


With the same API you can also query OpenAI models:

```python
gpt4 = OpenAIModel("gpt-4-0613")

gpt4.query_pydantic(Info,
                    "Just be helpful.",
                    "Who was the first man in the moon?")
```

Which creates an Info object initialized as the one listed above.

If Pydantic BaseModel objects are too much for your project, you can also use a very simple language called dictype, which defines structure and types of output dicts.

Sibila also includes model management and tools to compare output between models.


## Getting started

Installation, accessing OpenAI, getting local models: [How to get started](docs/getting-started.md).


## Examples

The examples show what you can do with local or remote GPT models in Sibila: structured data extraction, classification, summarization, etc. 

[Check them here](examples/readme.md). More are coming.


## Documentation

The API reference and more info is [available here](https://jndiogo.github.io/sibila/).


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Sibila?

Sibila is the Portuguese word for Sibyl. [The Sibyls](https://en.wikipedia.org/wiki/Sibyl) were oracular women in ancient Greece. They uttered mysterious words that left common people puzzled.

![Michelangelo's Delphic Sibyl, Sistine Chapel ceiling](https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/DelphicSibylByMichelangelo.jpg/471px-DelphicSibylByMichelangelo.jpg)

Michelangelo's Delphic Sibyl, in the Sistine Chapel ceiling.


# Sibila

Extract structured data from remote or local LLM models. Predictable output is important for serious use of LLMs.

- Query structured data into Pydantic objects, dataclasses or simple types.
- Access remote models from OpenAI, Anthropic, Mistral AI and other providers.
- Use vision models like GPT-4o, to extract structured data from images.
- Run local models like Llama-3, Phi-3, OpenChat or any other GGUF file model.
- Sibila is also a general purpose model access library, to generate plain text or free JSON results, with the same API for local and remote models.

No matter how well you craft a prompt begging a model for the format you need, it can always respond something else. Extracting structured data can be a big step into getting predictable behavior from your models.

See [What can you do with Sibila?](https://jndiogo.github.io/sibila/what/)


## Structured data

To extract structured data, using a local model:

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

If Pydantic BaseModel objects are too much for your project, Sibila supports similar functionality with Python dataclasses. Also includes asynchronous access to remote models.


## Vision models

Sibila supports image input, alongside text prompts. For example, to extract the fields from a receipt in a photo:

![Image](https://upload.wikimedia.org/wikipedia/commons/6/6a/Receipts_in_Italy_13.jpg)

``` python
from pydantic import Field

model = Models.create("openai:gpt-4o")

class ReceiptLine(BaseModel):
    """Receipt line data"""
    description: str
    cost: float

class Receipt(BaseModel):
    """Receipt information"""
    total: float = Field(description="Total value")
    lines: list[ReceiptLine] = Field(description="List of lines of paid items")

info = model.extract(Receipt,
                     ("Extract receipt information.", 
                      "https://upload.wikimedia.org/wikipedia/commons/6/6a/Receipts_in_Italy_13.jpg"))
info
```

Returns receipt fields structured in a Pydantic object:

```
Receipt(total=5.88, 
        lines=[ReceiptLine(description='BIS BORSE TERM.S', cost=3.9), 
               ReceiptLine(description='GHIACCIO 2X400 G', cost=0.99),
               ReceiptLine(description='GHIACCIO 2X400 G', cost=0.99)])
```

Local vision models based on llama.cpp/llava can also be used.

⭐ Like our work? [Give us a star!](https://github.com/jndiogo/sibila)

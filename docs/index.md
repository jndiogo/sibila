# Sibila

Extract structured data from remote or local LLM models. Because predictable output is important for serious use of LLMs.

- Query structured data into Pydantic objects, dataclasses or simple types.
- Access remote models from OpenAI, Anthropic, Mistral AI and other providers.
- Use local models like Llama-3, Phi-3, OpenChat or any other GGUF file model.
- Besides structured extraction, Sibila is also a general purpose model access library, to generate plain text or free JSON results, with the same API for local and remote models.
- Model management: download models, manage configuration, quickly switch between models.

No matter how well you craft a prompt begging a model for the format you need, it can always respond something else. Extracting structured data can be a big step into getting predictable behavior from your models.

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

If Pydantic BaseModel objects are too much for your project, Sibila supports similar functionality with Python dataclasses. Also includes asynchronous access to remote models.


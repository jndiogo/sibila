# Sibila

Extract structured data from remote or local file LLM models.

- Extract data into Pydantic objects, dataclasses or simple types.
- Same API for local file models and remote OpenAI models.
- Model management: download models, manage configuration and quickly switch between models.
- Tools for evaluating output across local/remote models, for chat-like interaction and more.

See [What can you do with Sibila?](what.md)

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

# Documentation

Extract structured data from LLM models, using a common API to access remote models like GPT-4 or local models via llama.cpp.

- Query structured data into dataset or Pydantic BaseModel objects.
- Use the same API for local and remote models.
- Thread-based interaction with chat/instruct fine-tuned models.
- Compare output across local/remote models with included utilities, text or CSV output.
- Model directory: store configurations and quickly switch between models.
- Automatic chat templates: identifies and uses the right templates for each model.

With Sibila you can extract structured data from a local quantized model like OpenChat-3.5 with 7B params:

```python
from sibila import (LlamaCppModel, OpenAIModel)
from pydantic import BaseModel, Field

class Info(BaseModel):
    event_year: int
    first_name: str
    last_name: str
    age_at_the_time: int
    nationality: str

openchat = LlamaCppModel("openchat-3.5-1210.Q5_K_M.gguf")

openchat.extract(Info,
                 "Who was the first man in the moon?",
                 inst="Just be helpful.") # instructions, aka system message
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

gpt4.extract(Info,
             "Who was the first man in the moon?",
             inst="Just be helpful.") # instructions, aka system message
```

And this creates an Info object initialized from model's response, as above.

If Pydantic BaseModel objects are too much for your project, you can also use ligher Python dataclass.

Sibila also includes model management and tools to compare output between models.


## Getting started

Installation, accessing OpenAI, getting local models - [how to get started](getting-started.md).


## API Reference

[Reference for the Sibila API](api-reference.md).
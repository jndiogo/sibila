# Documentation

Extract structured data from LLM models, using a common API to access remote models like GPT-4 or local models via llama.cpp.

- Query structured data into Pydantic BaseModel objects or typed Python dicts.
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

## Getting started

Installation, accessing OpenAI, getting local models - [how to get started](getting-started.md).


## API Reference

[Reference for the Sibila API](api-reference.md).
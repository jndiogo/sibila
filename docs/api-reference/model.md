---
title: Model classes
---

## Local models
::: sibila.LlamaCppModel
    options:
        members:
            - __init__
            - extract
            - classify
            - __call__
            - json
            - dataclass
            - pydantic
            - gen
            - gen_json
            - gen_dataclass
            - gen_pydantic
            - token_len
            - tokenizer
            - ctx_len
            - known_models
            - desc
            - n_embd
            - n_params
            - get_metadata


## Remote models
::: sibila.OpenAIModel
    options:
        members:
            - __init__
            - extract
            - classify
            - gen
            - json
            - dataclass
            - pydantic
            - __call__
            - gen_json
            - gen_dataclass
            - gen_pydantic
            - token_len
            - tokenizer
            - ctx_len
            - known_models
            - desc
            - get_metadata



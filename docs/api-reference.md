---
title: API Reference
---

## Models
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
            - desc
            - n_embd
            - n_params
            - get_metadata


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
            - desc
            - get_metadata



## Models directory
::: sibila.Models
    options:
        members:
            - setup
            - clear
            - info
            - create
            - add_search_path
            - set_genconf
            - add_model
            - get_format
            - search_format
            - is_format_supported
            - add_format




## Generation Configs
::: sibila.GenConf

::: sibila.JSchemaConf




## Generation Results and Errors
::: sibila.GenRes

::: sibila.GenError
    options:
        members:
            - __init__
            - raise_if_error

::: sibila.GenOut








## Messages, Threads, Context
::: sibila.MsgKind
    options:
        members:
            - IN
            - OUT
            - INST

::: sibila.Thread
    options:
        members:
            - __init__
            - clear
            - last_kind
            - last_text
            - inst
            - add
            - addx
            - get_text
            - set_text
            - concat
            - load
            - save
            - init_inst_in
            - add_in
            - add_out
            - add_out_in
            - make_inst_in
            - make_out_in
            - msg_as_chatml
            - as_chatml
            - has_text_lower

::: sibila.Trim

::: sibila.Context
    options:
        members:
            - __init__
            - clear
            - trim



## Tools
::: sibila.tools
    options:
        members:
            - interact
            - loop
            - recursive_summarize

## Multigen
::: sibila.multigen
    options:
        members:
            - thread_multigen
            - query_multigen
            - multigen
            - cycle_gen_print




## Tokenizers

Tokenizers used in models.

::: sibila.LlamaCppTokenizer
    options:
        members:
            - __init__
            - encode
            - decode
            - token_len

::: sibila.OpenAITokenizer
    options:
        members:
            - __init__
            - encode
            - decode
            - token_len




---
title: API Reference
---

## Models
::: sibila.LlamaCppModel
    options:
        members:
            - __init__
            - gen
            - json
            - dictype
            - pydantic
            - gen_
            - json_
            - dictype_
            - pydantic_
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
            - gen
            - json
            - dictype
            - pydantic
            - gen_
            - json_
            - dictype_
            - pydantic_
            - token_len
            - tokenizer
            - ctx_len
            - desc
            - get_metadata





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



## Directories
::: sibila.ModelDir
    options:
        members:
            - add
            - add_model
            - add_search_path
            - set_genconf
            - create
            - clear

::: sibila.FormatDir
    options:
        members:
            - add
            - get
            - search
            - info
            - clear


## Multigen
::: sibila.multigen
    options:
        members:
            - thread_multigen
            - query_multigen
            - multigen
            - cycle_gen_print


## Tools
::: sibila.tools
    options:
        members:
            - interact
            - loop
            - recursive_summarize


## Dictype
::: sibila.dictype
    options:
        members:
            - json_schema_from_dictype




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




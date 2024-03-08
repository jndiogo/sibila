---
title: What can you do with Sibila?
---


LLM models answer your questions in the best way their training allows, but they always answer back in plain text (or tokens).

With Sibila, you can extract structured data from LLM models. Not whatever the model chose to output (even if you asked it to answer in a certain format), but the exact fields and types that you need.

This not only simplifies handling the model responses but can also open new possibilities: you can now deal with the model in a structured way.



## Extract Pydantic, dataclasses or simple types

To specify the structured output that you want from the model, you can use Pydantic's BaseModel derived classes, or the lightweight Python dataclasses, if you don't need the whole Pydantic.

With Sibila, you can also use simple data types like bool, int, str, enumerations or lists. 
For example, need to classify something? 

!!! example
    ``` python
    from sibila import Models

    model = Models.create("openai:gpt-4")

    model.classify(["good", "neutral", "bad"], 
                   "Running with scissors")
    ```

    !!! result
        ```
        'bad'
        ```

How does it work? Extraction to the given data types is guaranteed by automatic JSON Schema grammars in local models, or by the Tools functionality of OpenAI API remote models.



## From your models or OpenAI's

Small downloadable 7B parameter models are getting better every month and they have reached a level where they are competent enough for most common data extraction or summarization tasks.

With 8Gb or more of RAM or GPU memory, you can get good structured output from models like OpenChat, Zephyr, Mistral 7B, or any other GGUF file.

You can use any paid OpenAI model, as well as any model that llama.cpp can run, with the same API. Choose the best model for each use, allowing you the freedom of choice.




## With model management

Includes a Models factory that creates models from simple names instead of having to track model configurations, filenames or chat templates.

``` python
local_model = Models.create("llamacpp:openchat")

remote_model = Models.create("openai:gpt-4")    
```

This makes the switch to newer models much easier, and makes it simpler to compare model outputs.

Sibila includes a CLI tool to download GGUF models from [Hugging Face model hub](https://www.huggingface.co), and to manage its Models factory.


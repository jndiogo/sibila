---
title: What can you do with Sibila?
---


Popular LLM usage is associated with chatbots: user enters some text, the model answers back in plain text. But when one wants to use LLMs from software, sending and receiving plain text can be quite painful with people having to create all sorts of prompts begging for a certain format - and then hoping the model complies. (prompts like: "Please answer me in JSON or I'll do something terrible!"). But there's never a warranty, as the model is just outputting plain text.

With Sibila, you can extract structured data from remote or local LLM models. Not whatever the model chose to output, but the exact fields and types that you need, specified with Pydantic, Python dataclasses or simple types.

In remote models, this is done via the provider's API, while in local llama.cpp based models, the output is constrained with a JSON Schema grammar. Local and remote model differences are hidden behind a common API, which simplifies model switching. Local open source models are getting better and will one day replace commercial models.

Getting structured output not only simplifies handling the model responses but can also open new possibilities: you can now deal with the model in an ordered and more predictable way.

And besides structured output, with Sibila you can also query vision models (accepting image inputs), and it includes useful functionality like message threads, model management and more. 



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


## Extract data from images

Sibila supports image input models, like GPT-4o and Anthropic models, as well as local Llava-based models.

Vision models can describe and interpret, recommend suggestions or extract information from images. With Sibila, this data can be extracted in a structured way.



## From your models or remote models

Small downloadable 7B parameter models are getting better every month and they have reached a level where they are competent enough for most common data extraction or summarization tasks.

With 8Gb or more of RAM or GPU memory, you can get good structured output from models like Llama-3, Phi-3, OpenChat or any other GGUF file.

Or perhaps the task requires use of state of the art remote models from OpenAI, Anthropic, Mistral AI or other [providers](models/remote_model.md) - no problem, simply change the model's name. 



## Common API

The same API is used for both remote and local models. This makes the switch to newer or alternative models much easier, and makes it simpler to evaluate model outputs.

With a common API you can choose the best model for each use, allowing more freedom of choice.



## And with model management

Includes a Models factory that creates models from simple names instead of having to track model configurations, filenames or chat templates.

``` python
local_model = Models.create("llamacpp:openchat")

remote_model = Models.create("openai:gpt-4")    
```

Chat templates are automatically used for local models from an included format registry.

Sibila includes a CLI tool to download GGUF models from [Hugging Face model hub](https://www.huggingface.co), and to manage its Models factory.


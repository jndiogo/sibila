---
title: OpenAI
---

Sibila can use [OpenAI](https://openai.com) remote models, for which you'll need a paid OpenAI account and its API key. Although you can pass this key when you create the model object, it's more secure to define an env variable with this information:

=== "Linux and Mac"
    ```
    export OPENAI_API_KEY="..."
    ```

=== "Windows"
    ```
    setx OPENAI_API_KEY "..."
    ```

Another possibility is to store your OpenAI key in .env files, which has many advantages: see the [dotenv-python](https://github.com/theskumar/python-dotenv) package.


## Creating models

OpenAI models can be used by Sibila through the [OpenAIModel class](../api-reference/remote_model.md#sibila.OpenAIModel). 

!!! example
    ``` python
    from sibila import OpenAIModel

    model = OpenAIModel("gpt-3.5-turbo-0125")

    model("I think that I shall never see.")
    ```

    !!! success "Result"
        ```
        'A poem as lovely as a tree.'
        ```

You can also create an OpenAI model in the [Models factory](models_factory.md) by using the "openai:" provider prefix like this:

``` python
from sibila import Models

model = Models.create("openai:gpt-3.5-turbo-0125")
```




## Model list

The available OpenAI models are listed [here](https://platform.openai.com/docs/models). You can also fetch a list of known model names by calling OpenAIModel.known_models():

!!! example
    ``` python
    OpenAIModel.known_models()
    ```

    !!! success "Result"
        ```
        ['babbage-002',
         'dall-e-2',
         'dall-e-3',
         'davinci-002',
         'gpt-3.5-turbo',
         'gpt-3.5-turbo-0125',
         'gpt-3.5-turbo-0301',
         'gpt-3.5-turbo-0613',
         'gpt-3.5-turbo-1106',
         'gpt-3.5-turbo-16k',
         'gpt-3.5-turbo-16k-0613',
         'gpt-3.5-turbo-instruct',
         'gpt-3.5-turbo-instruct-0914',
         'gpt-4',
         'gpt-4-0125-preview',
         'gpt-4-0613',
         'gpt-4-1106-preview',
         'gpt-4-1106-vision-preview',
         'gpt-4-turbo-preview',
         'gpt-4-vision-preview',
         'text-embedding-3-large',
         'text-embedding-3-small',
         'text-embedding-ada-002',
         'tts-1',
         'tts-1-1106',
         'tts-1-hd',
         'tts-1-hd-1106',
         'whisper-1']
        ```

Not all of these models are for text inference, but the names that start with "gpt" are (excluding the "vision" models), and you can use those model names to create an OpenAI model.


## JSON Schema models

At the time of writing, not all OpenAI inference models support JSON Schema generation via the Tools functionality, which is required for structured data extraction. The following models (and later versions) allow JSON extraction:

- gpt-3.5-turbo-1106 and later
- gpt-4-1106-preview, gpt-4-turbo-preview and later



## Using for other providers

You can also use the OpenAIModel class to access any provider that uses the OpenAI API by setting the base_url and api_key arguments. For example to use the Together.ai service with the OpenAIModel class:

``` python
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1",

client = OpenAIModel(
    model_name,
    base_url="https://api.together.xyz/v1",
    api_key=os.environ["TOGETHER_API_KEY"],
)
```

This is just an example, as Together.ai has a [dedicated Sibila class](together.md), but you can access any other OpenAI-compatible servers with the OpenAIModel class.
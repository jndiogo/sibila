---
title: Remote models
---

Sibila can use OpenAI remote models, for which you'll need a paid OpenAI account and its API key. Although you can pass this key when you create the model object, it's more secure to define an env variable with this information:

=== "Linux and Mac"
    ```
    export OPENAI_API_KEY="..."
    ```

=== "Windows"
    ```
    setx OPENAI_API_KEY "..."
    ```

Another possibility is to store your OpenAI key in .env files, which has many advantages: see the [dotenv-python](https://github.com/theskumar/python-dotenv) package.


## Model names

OpenAI models can be used by Sibila through the [OpenAIModel class](../api-reference/model.md#sibila.OpenAIModel). To get a list of known model names:


!!! example
    ``` python
    from sibila import OpenAIModel

    OpenAIModel.known_models()
    ```

    !!! success "Result"
        ```
        ['gpt-4-0613',
        'gpt-4-32k-0613',
        'gpt-4-0314',
        'gpt-4-32k-0314',
        'gpt-4-1106-preview',
        'gpt-4',
        'gpt-4-32k',
        'gpt-3.5-turbo-1106',
        'gpt-3.5-turbo-0613',
        'gpt-3.5-turbo-16k-0613',
        'gpt-3.5-turbo-0301',
        'gpt-3.5-turbo',
        'gpt-3.5-turbo-16k',
        'gpt-3',
        'gpt-3.5']
        ```

You can use any of these model names to create an OpenAI model. For example:

!!! example
    ``` python
    model = OpenAIModel("gpt-3.5")

    model("I think that I shall never see.")
    ```

    !!! success "Result"
        ```
        'A poem as lovely as a tree.'
        ```




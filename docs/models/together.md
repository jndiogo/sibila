---
title: Together.ai
---

With Sibila you can use the models hosted by [Together.ai](https://together.ai), for which you'll need an API key (which is initially free). As in other providers, although you can pass this key when you create the model object, it's more secure to define an env variable with this information:

=== "Linux and Mac"
    ```
    export TOGETHER_API_KEY="..."
    ```

=== "Windows"
    ```
    setx TOGETHER_API_KEY "..."
    ```

Another possibility is to store your API key in .env files, which has many advantages: see the [dotenv-python](https://github.com/theskumar/python-dotenv) package.


## Creating models

Models served by Together.ai can be used by Sibila through the [TogetherModel class](../api-reference/remote_model.md#sibila.TogetherModel). 

!!! example
    ``` python
    from sibila import TogetherModel

    model = TogetherModel("mistralai/Mixtral-8x7B-Instruct-v0.1")

    model("I think that I shall never see.")
    ```

    !!! success "Result"
        ```
        A poem lovely as a tree. These are the beginning lines of a famous poem called "Trees" written by Joyce Kilmer. The full poem goes as follows:

        I think that I shall never see
        A poem lovely as a tree.

        A tree whose hungry mouth is prest
        Against the earth’s sweet flowing breast;

        A tree that looks at God all day,
        And lifts her leafy arms to pray;

        A tree that may in Summer wear
        A nest of robins in her hair;

        Upon whose bosom snow has lain;
        Who intimately lives with rain.

        Poems are made by fools like me,
        But only God can make a tree.
        ```

You can also create a Together.ai model in the [Models factory](models_factory.md) by using the "together:" provider prefix:

``` python
from sibila import Models

model = Models.create("together:mistralai/Mixtral-8x7B-Instruct-v0.1")
```




## Model list

The available Together.ai text inference models models are listed [here](https://docs.together.ai/docs/inference-models).

Unfortunately Together.ai doesn't provide an API to list the models, so TogetherModel.known_models() will return None.



## JSON Schema models

At the time of writing, only the following Together.ai models support JSON Schema generation, which is required for structured data extraction:

- mistralai/Mixtral-8x7B-Instruct-v0.1
- mistralai/Mistral-7B-Instruct-v0.1
- togethercomputer/CodeLlama-34b-Instruct

You can still use any of the other models for plain text or schema-free JSON generation, for example with the [Model.call()](../api-reference/remote_model.md#sibila.TogetherModel.call) or [Model.json()](../api-reference/remote_model.md#sibila.TogetherModel.json) methods.
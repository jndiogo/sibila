---
title: Mistral AI
---

With Sibila you can access [Mistral AI](https://mistral.com) remote models, for which you'll need an API key. Although you can pass this key when you create the model object, it's more secure to define an env variable with this information:

=== "Linux and Mac"
    ```
    export MISTRAL_API_KEY="..."
    ```

=== "Windows"
    ```
    setx MISTRAL_API_KEY "..."
    ```

Another possibility is to store your API key in .env files, which has many advantages: see the [dotenv-python](https://github.com/theskumar/python-dotenv) package.



## Creating models

Mistral AI models can be used by Sibila through the [MistralModel class](../api-reference/remote_model.md#sibila.MistralModel). 

!!! example
    ``` python
    model = MistralModel("mistral-large-latest")

    model("I think that I shall never see.")
    ```

    !!! success "Result"
        ```
        A poem as lovely as a tree.

        This is a line from the poem "Trees" by Joyce Kilmer. The full poem is:

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

        Do you have any other questions or is there something else you'd like to talk about?
        I'm here to help!
        ```


You can also create a Mistral model in the [Models factory](models_factory.md) by using the "mistral:" provider prefix like this:

``` python
from sibila import Models
model = Models.create("mistral:mistral-large-latest")
```



## Model list

To get a list of known model names:

!!! example
    ``` python
    from sibila import MistralModel

    MistralModel.known_models()
    ```

    !!! success "Result"
        ```
        ['mistral-embed',
         'mistral-large-2402',
         'mistral-large-latest',
         'mistral-medium',
         'mistral-medium-2312',
         'mistral-medium-latest',
         'mistral-small',
         'mistral-small-2312',
         'mistral-small-2402',
         'mistral-small-latest',
         'mistral-tiny',
         'mistral-tiny-2312',
         'open-mistral-7b',
         'open-mixtral-8x7b']
        ```

At the time of writing, all Mistral AI models support JSON Schema extraction.

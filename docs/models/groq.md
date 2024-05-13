---
title: Groq
---

To use the models hosted by [Groq](https://console.groq.com/playground), you'll need an API key (which is initially free). As in other providers, although you can pass this key when you create the model object, it's more secure to define an env variable with this information:

=== "Linux and Mac"
    ```
    export GROQ_API_KEY="..."
    ```

=== "Windows"
    ```
    setx GROQ_API_KEY "..."
    ```

Another possibility is to store your API key in .env files, which has many advantages: see the [dotenv-python](https://github.com/theskumar/python-dotenv) package.


## Creating models
Models served by Groq can be used by Sibila through the [GroqModel class](../api-reference/remote_model.md#sibila.GroqModel). 

!!! example
    ``` python
    from sibila import GroqModel

    model = GroqModel("llama3-70b-8192")

    model("I think that I shall never see.")
    ```

    !!! success "Result (model is hallucinating)"
        ```
        A poem!

        "I think that I shall never see
        A poem lovely as a tree.
        A tree whose hungry mouth is prest
        Against the earth's sweet flowing breast;

        A tree that looks at God all day,
        And lifts her leafy arms to pray;
        A tree that may in Summer wear
        A nest of robins in her hair;

        Upon whose bosom snow has lain;
        Who intimately lives with rain.
        Poems are made by fools like me,
        But only God can make a tree."

        — Alfred Joyce Kilmer█
        ```


You can also create a Groq model in the [Models factory](models_factory.md) by using the "groq:" provider prefix:

``` python
from sibila import Models

model = Models.create("groq:llama3-70b-8192")
```




## Model list

The available Groq text inference models models are listed [here](https://console.groq.com/docs/models). You should use the listed "Model ID" names as the model name, when creating a GroqModel object.

Unfortunately Groq doesn't provide an API to list the models, so GroqModel.known_models() will return None.



## JSON Schema models

All the Groq AI models should support JSON Schema generation, which is required for structured data extraction.

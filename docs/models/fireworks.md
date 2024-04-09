---
title: Fireworks AI
---

With Sibila you can use the models hosted by [Fireworks AI](https://fireworks.ai), for which you'll need an API key (which is initially free). As in other providers, although you can pass this key when you create the model object, it's more secure to define an env variable with this information:

=== "Linux and Mac"
    ```
    export FIREWORKS_API_KEY="..."
    ```

=== "Windows"
    ```
    setx FIREWORKS_API_KEY "..."
    ```

Another possibility is to store your API key in .env files, which has many advantages: see the [dotenv-python](https://github.com/theskumar/python-dotenv) package.


## Creating models

Models served by Together.ai can be used by Sibila through the [FireworksModel class](../api-reference/remote_model.md#sibila.FireworksModel). 

!!! example
    ``` python
    from sibila import FireworksModel

    model = FireworksModel("accounts/fireworks/models/gemma-7b-it")

    model("I think that I shall never see.")
    ```

    !!! success "Result (model is hallucinating)"
        ```
        The poem "I think that I shall never see" is a poem by William Blake. 
        It is a poem about the loss of sight. The speaker is saying that they 
        will never be able to see again. The poem is a reflection on the beauty 
        of sight and the sadness of blindness.
        ```


You can also create a Fireworks AI model in the [Models factory](models_factory.md) by using the "fireworks:" provider prefix:

``` python
from sibila import Models
model = Models.create("fireworks:accounts/fireworks/models/gemma-7b-it")
```




## Model list

The available Fireworks text inference models models are listed [here](https://fireworks.ai/models). Unfortunately Fireworks AI doesn't provide an API to list the models, so FireworksModel.known_models() will return None.



## JSON Schema models

All the Fireworks AI models should support JSON Schema generation, which is required for structured data extraction.

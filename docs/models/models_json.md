---
title: models.json file
---



Inside the "models" folder you'll find the file "models.json". The idea is that you can use this file to configure names and settings that you can then use in [Models.create()](../api-reference/models.md#sibila.Models.create).

This is not strictly needed, as you can create models from their filenames or remote model names, however it's a good idea, specially if you'll be using a model for some time.

!!! Example "A "models.json" file:"

    ``` json
    {
        # "llamacpp" is a provider, you can then create models with names 
        # like "provider:model_name", for ex: "llamacpp:openchat"
        "llamacpp": { 

            "default": { # place here default args for all llamacpp: models.
                "genconf": {"temperature": 0.0}
                # each entry below can then override as needed
            },
            
            "openchat": { # a model definition
                "name": "openchat-3.5-1210.Q4_K_M.gguf",
                "format": "openchat" # chat template format used by this model
            },

            "phi2": {
                "name": "phi-2.Q4_K_M.gguf", # model filename
                "format": "phi2",
                "genconf": {"temperature": 2.0} # a hot-headed model
            },

            "oc": "openchat" 
            # this is an alias: "oc" forwards to the "openchat" entry
        },

        # The "openai" provider. A model can be created with name: "openai:gpt-4"
        "openai": { 

            "default": {}, # default settings for all OpenAI models
            
            "gpt-3.5": {
                "name": "gpt-3.5-turbo-1106" # OpenAI's model name
            },

            "gpt-4": {
                "name": "gpt-4-1106-preview"
            },
        },

        # "alias" entry is not a provider but a way to have simpler alias names.
        # For example you can use "alias:develop" or even simpler, just "develop" to create the model:
        "alias": { 
            "develop": "llamacpp:openchat",
            "production": "openai:gpt-3.5"
        }
    }

    ```

So we have two top entries for providers "llamacpp" and "openai", and an "alias" entry.

Inside each provider entry, we have a "defaults" key, which can store any default GenConf or other arguments passed during model creation. These defaults are overridden by any keys of the same name specified in each model. You can see this in the "phi2" entry, which overrides the genconf entry given in "default", setting temperature to 2.0. These overrides are per dictionary element.


In the above "model.json" example, let's look at the "openchat" model entry:

``` json
"openchat": { # a model definition
    "name": "openchat-3.5-1210.Q4_K_M.gguf",
    "format": "openchat" # chat template format used by this model
},
```

The "openchat" key name is the name you'll use to create the model, after "llamacpp:":

``` py
# initialize Models to this folder
Models.setup("../../models")

model = Models.create("llamacpp:openchat")
```

You can have the following keys in a model entry:

| Key | |
|-----|-|
| name | The filename to use when loading a model (or remote model name) |
| format | Identifies the chat template format that it should use. This key is optional, as long as the model can locate its format (because it already has the format its metadata). |
| genconf | Default GenConf (generation config settings) used to create the model, which will default to use them in each generation. These config settings are merged element-wise from any specified in the "defaults" entry for the provider. |
| other | Any other keys added will be passed during model creation as its arguments. You can learn which arguments are possible in the API reference for LlamaCppModel or OpenAIModel. For example you can pass "ctx_len": 2048 to define the context length to use. As genconf, these keys are merged element-wise from any specified in the "defaults" entry for the provider. |

<!-- add links to LlamaCppModel and OpenAI API ref -->


The "alias" entry is a handy way to keep names that point to actual model entries (independent of provider). Note the two alias entries "develop" and "production" - you could then create the production model by doing:

``` python
# initialize Models to this folder
Models.setup("../../models")

model = Models.create("production")
```



For an example with many models defined, see the "[models/models.json](https://github.com/jndiogo/sibila/blob/main/models/models.json)" file.

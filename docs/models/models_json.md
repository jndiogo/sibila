---
title: Managing models
---

Model names are stored in a file named "models.json", in your "models" folder. Models registered in this file can then be used when calling [Models.create()](../api-reference/models.md#sibila.Models.create) to create an instance of the model.

Registering a name is not strictly needed, as you can create models from their filenames or remote model names, for example in most examples you'll find models created with:

``` python
model = Models.create("llamacpp:openchat-3.5-1210.Q4_K_M.gguf")
```

However, it's a good idea to register a name, specially if you'll be using a model for some time, or there's the possibility you'll need to replace it later. If you register a name, only that will later need to be changed.

There are two ways of registering names: by using the sibila CLI tool or by directly editing the "models.json" file.

##  Use the "sibila models" CLI tool

To register a model with the Models factory you can use the "sibila models" tool. Run in the "models" folder:

```
> sibila models -s "llamacpp:openchat openchat-3.5-1210.Q4_K_M.gguf" openchat

Using models directory '.'
Set model 'llamacpp:openchat' with name='openchat-3.5-1210.Q4_K_M.gguf', 
format='formatx' at './models.json'.
```

First argument after -s is the new entry name (including the "llamacpp:" provider), then the filename, then the [chat template format](setup_format.md), if needed.

This will create an "openchat" entry in "models.json", exactly like the manually created below.


## Manually edit "models.json"

In alternative, you can manually register a model name by editing the "models.json" file located in you "models" folder.

!!! Example "A "models.json" file:"

    ``` json
    {
        # "llamacpp" is a provider, you can then create models with names 
        # like "provider:model_name", for ex: "llamacpp:openchat"
        "llamacpp": { 

            "_default": { # place here default args for all llamacpp: models.
                "genconf": {"temperature": 0.0}
                # each model entry below can then override as needed
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
            # this is a link: "oc" forwards to the "openchat" entry
        },

        # The "openai" provider. A model can be created with name: "openai:gpt-4"
        "openai": { 

            "_default": {}, # default settings for all OpenAI models
            
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

Looking at the above structure, we have two top entries for providers "llamacpp" and "openai", and also an "alias" entry.

Inside each provider entry, we have a "_defaults" key, which can store a base GenConf or other arguments passed during model creation. The default values defined in "_default" entries can later be overridden by any keys of the same name specified in each model definition. You can see this in the "phi2" entry, which overrides the genconf entry given in the above "_default", setting temperature to 2.0. 
Keys are merged element-wise from any specified in the "_defaults" entry for the provider: keys with the same name are overridden, all other keys are inherited.


In the above "model.json" example, let's look at the "openchat" model entry:

``` json
"openchat": { # a model definition
    "name": "openchat-3.5-1210.Q4_K_M.gguf",
    "format": "openchat" # chat template format used by this model
},
```

The "openchat" key name is the name you'll use to create the model as "llamacpp:openchat":

``` py
# initialize Models to this folder
Models.setup("../../models")

model = Models.create("llamacpp:openchat")
```

You can have the following keys in a model entry:

| Key | |
|-----|-|
| name | The filename to use when loading a model (or remote model name) |
| format | Identifies the chat template format that it should use, from the ["formats.json"](formats_json.md) file. Some local models include the chat template format in their metadata, so this key is optional. |
| genconf | Default [GenConf (generation config settings)](../api-reference/generation.md#sibila.GenConf) used to create the model, which will default to use them in each generation. These config settings are merged element-wise from any specified in the "_defaults" entry for the provider. |
| other | Any other keys will be passed during model creation as its arguments. You can learn which arguments are possible in the API reference for [LlamaCppModel](../api-reference/model.md#sibila.LlamaCppModel) or [OpenAIModel](../api-reference/model.md#sibila.OpenAIModel). For example you can pass "ctx_len": 2048 to define the context length to use. As genconf, these keys are merged element-wise from any specified in the "_defaults" entry for the provider. |

<!-- add links to LlamaCppModel and OpenAI API ref -->


The "alias" entry is a handy way to keep names that point to actual model entries (independent of provider). Note the two alias entries "develop" and "production" in the above "models.json" - you could then create the production model by doing:

``` python
# initialize Models to this folder
Models.setup("../../models")

model = Models.create("production")
```
Alias entries can be used as "alias:production" or without the "alias:" provider, just as "production" as in the example above.

For an example of a JSON file with many models defined, see the "[models/models.json](https://github.com/jndiogo/sibila/blob/main/models/models.json)" file.

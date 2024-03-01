---
title: Models factory
---


There's a more flexible way to create models, than creating OpenAIModel or LlamaCppModel objects. The [Models factory class](../api-reference/models.md) can create a model based on a given name, for example:

``` python
Models.setup("../../models")

model = Models.create("openai:gpt-4")
```

The first line calls [Models.setup()](../api-reference/models.md#sibila.Models.setup) to initialize it the folder where model files and configs ("models.json" and "formats.json") are located.

The second line calls [Models.create()](../api-reference/models.md#sibila.Models.create) to create a model from the name "openai:gpt-4". The names should be in the format "provider:model_name" and Sibila currently supports two providers:

| Provider | Name | Creates object of type |
|----------|------|---------------------|
| llamacpp | filename or a "models.json" name | LlamaCppModel |
| openai | remote model name or a "models.json" name | OpenAIModel |


The name part, after the ":" must be either an existing filename or a remote model name. It cal also be a name that has been defined in a JSON file named "models.json", located in the "models" folder.

Some examples of valid identifiers with local filenames or remote model names are:

- llamacpp:openchat-3.5-1210.Q4_K_M.gguf
- llamacpp:zephyr-7b-beta.Q4_K_M.gguf
- openai:openai:gpt-3.5-turbo-1106

In this case, there's no need to define the names in models.json. But for continued use, it's generally a good idea to create an entry in the models.json file  - this allows future model changing to be much easier.

Examples of using names defined in models.json:

- llamacpp:openchat
- llamacpp:zephyr
- openai:gpt-3.5

See [models.json](models_json.md) to learn how to create model names.

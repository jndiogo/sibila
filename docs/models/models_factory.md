---
title: Models factory
---


The Models factory is based in a "models" folder that contains two configuration files: "models.json" and "formats.json" and the actual files for local models. 

The [Models factory class](../api-reference/models.md) is a more flexible way to create models, for example:

``` python
Models.setup("../../models")

model = Models.create("openai:gpt-4")
```

The first line calls [Models.setup()](../api-reference/models.md#sibila.Models.setup) to initialize the factory with the folder where model files and configs ("models.json" and "formats.json") are located.

The second line calls [Models.create()](../api-reference/models.md#sibila.Models.create) to create a model from the name "openai:gpt-4". In this case we created a remote model, but we could as well create a local model based in a GGUF file.

The names should be in the format "provider:model_name" and Sibila currently supports two providers:

| Provider | Type | Creates object of type |
|----------|------|------------------------|
| llamacpp | Local GGUF model | LlamaCppModel |
| openai   | Remote model | OpenAIModel   |


The name part, after the "provider:" must either be:

- A remote model name, like "gpt-4": "openai:gpt-4"
- A local model name, like "openchat": "llamacpp:openchat"
- The actual filename of a model in the "models" folder: "llamacpp:openchat-3.5-1210.Q4_K_M.gguf" - this is the form we use in the examples, but of course using "openchat" instead of the filename would be better...

Although you can use filenames as model names, it's generally a better idea, for continued use, to create an entry in the "models.json" file  - this allows future model replacement to be much easier.

See [Managing models](models_json.md) to learn how to register these model names.

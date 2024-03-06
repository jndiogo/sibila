---
title: Managing formats
---

A "formats.json" file stores the [chat template definitions](setup_format.md) used in models. This allows for models that don't have a chat template in their metadata to be detected and get the right format so they can function well.

If you downloaded the GitHub repository, you'll find a file named "[base_formats.json](https://github.com/jndiogo/sibila/blob/main/sibila/base_formats.json)", which is the default base configuration that will be used, with many known chat template formats. 

When you call [Models.setup()](../api-reference/models.md#sibila.Models.setup), any "formats.json" file found in the folder will be loaded and its definitions will be merged with the ones from "base_formats.json" which are loaded on initialization. Any entries with the same name will be replaced by freshly loaded ones.

How to add a new format entry that can be used when creating a model? You can do it with the sibila CLI tool or by manually editing the formats.json file.


## With "sibila formats" CLI tool

Run the sibila CLI tool in the "models" folder:

```
> sibila formats -s openchat openchat "{{ bos_token }}...{% endif %}"

Using models directory '.'
Set format 'openchat' with match='openchat', template='{{ bos_token }}...'
```

First argument after -s is the format entry name, second the match regular expression (to identify the model filename) and finally the template. Help is available with "sibila formats --help".




## Manually edit "formats.json"

In alternative, we can edit the "formats.json" file in the "Models" folder, and add the entry:

``` json
"openchat": {
    "match": "openchat", # a regexp to match model name or filename
    "template": "{{ bos_token }}..."
},
```

In the "openchat" key value we have a dictionary with the following keys:

| Key | |
|-----|-|
| match | Regular expression that will be used to match the model name or filename |
| template | The chat template definition in Jinja format |



The "openchat" format name we are defining here is the name you can use when creating a model, by setting the format argument:

``` python
model = LlamaCppModel.create("openchat-3.5-1210.Q4_K_M.gguf",
                             format="openchat")
```

or to be more practical: "openchat" is also the format name you would use when creating a ["models.json" entry](models_json.md) for a model, in the "format" key:

``` json
"openchat": {
    "name": "openchat-3.5-1210.Q4_K_M.gguf",
    "format": "openchat" # chat template format used by this model
},
```



See the "[base_formats.json](https://github.com/jndiogo/sibila/blob/main/sibila/base_formats.json)" file for all the default base formats.


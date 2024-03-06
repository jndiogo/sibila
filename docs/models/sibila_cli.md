---
title: "Sibila CLI tool"
---


The Sibila Command-Line Interface tool simplifies managing the Models factory and is useful to download models from [Hugging Face model hub](https://www.huggingface.co).

The Models factory is based in a "models" folder that contains two configuration files: "models.json" and "formats.json" and the actual files for local models.

The CLI tool is divided in three areas or actions:

| Action | |
|--------|------------|
| models | Manage models in "model.json" files |
| formats | Manage formats in "model.json" files |
| hub | Search and download models from Hugging Face model hub |


In all commands you should pass the option "-m models_folder" with the path to the "models" folder. Or in alternative run the commands inside the "models" folder.

The following argument names are used below (other unlisted names should be descriptive enough):

| Name     | |
|----------|--|
| res_name | Model entry name in the form "provider:name", for example "llamacpp:openchat". |
| format_name | Name of a format entry in "formats.json", for example "chatml". |
| query | Case-insensitive query that will be matched by a substring search. |


Usage help is available by running "sibila --help" for general help, or "sibila action --help", where action is one of "models", "formats" or "hub".



## Sibila models

To register a model entry pointing to a model name or filename:

```
sibila models -s res_name model_name_or_filename [format_name]
```


To set the format_name for an existing model entry:

```
sibila models -f res_name format_name
```

To test if a model can run (for example to check if it has the chat template format defined):

```
sibila models -t res_name
```


List all models with optional case-insensitive substring query:

```
sibila models -l [query]
```


Delete a model entry in:

```
sibila models -d res_name
```




## Sibila formats

Check if a model filename has any format defined in the Models factory:

```
sibila formats -q filename
```

To register a chat template format, where match is a regexp that matches the model filename, template is the Jinja chat template:

```
sibila formats -s format_name match template
```


List all formats with optional case-insensitive substring query:

```
sibila models -l [query]
```



Delete a format entry:

```
sibila formats -d format_name
```


Update the local "formats.json" file by merging with with the "[sibila/base_formats.json](https://github.com/jndiogo/sibila/blob/main/sibila/base_formats.json)" file, preserving all existing local entries.

```
sibila formats -u
```




## Sibila hub


List models in the Hugging Face model hub that match the given queries. Argument query can be a list of strings to match, separated by a space character.

Arg Filename is case-insensitive for substring matching.

Arg exact_author is an exact and case-sensitive author name from Hugging Face model hub.

```
sibila hub -l query [-f filename] [-a exact_author]
```

To download a model, where model_id is a string like "TheBloke/openchat-3.5-1210-GGUF". Args filename and author_name same as above:

```
sibila hub -d model_id -f filename -a exact_author -s set name
```







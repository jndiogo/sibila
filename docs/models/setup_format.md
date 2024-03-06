---
title: Chat template format
---



## What are chat templates?

Because these models were fine-tuned for chat or instruct interaction, they use a chat template, which is a Jinja template that converts a list of messages into a text prompt. This template must follow the original format that the model was trained on - this is very important or you won't get good results.

Chat template definitions are Jinja templates like the following one, which is in ChatML format:

```
{% for message in messages %}
    {{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
{% endfor %}
```

When ran over a list of messages with system, user and model messages, the template produces text like the following:
```
<|im_start|>system
You speak like a pirate.<|im_end|>
<|im_start|>user
Hello there?<|im_end|>
<|im_start|>assistant
Ahoy there matey! How can I assist ye today on this here ship o' mine?<|im_end|>
```

Only by using the specific chat template for the model, can we get back the best results. 

Sibila tries to automatically detect which template to use with a model, either from the model name or from embedded metadata, if available. 




## Does the model have a built-in chat template format?

Some GGUF models include the chat template in their metadata, unfortunately this is not standard.

You can quickly check if the model has a chat template by running the sibila CLI in the same folder as the model file:

```
> sibila models -t "llamacpp:openchat-3.5-1210.Q4_K_M.gguf"

Using models directory '.'
Testing model 'llamacpp:openchat-3.5-1210.Q4_K_M.gguf'...
Model 'llamacpp:openchat-3.5-1210.Q4_K_M.gguf' was properly created and should run fine.

```

In this case the chat template format is included with the model and nothing else is needed.

Another way to test this is to try creating the model in python. If no exception is raised, the model GGUF file contains the template definition and should work fine.

!!! example "Example of model creation error"
    ``` python
    from sibila import LlamaCppModel

    model = LlamaCppModel("peculiar-model-7b.gguf")
    ```

    !!! failure "Error"
        ```
        ...

        ValueError: Could not find a suitable format (chat template) for this model.
        Without a format, fine-tuned models cannot function properly.
        See the docs on how you can fix this: pass the template in the format arg or 
        create a 'formats.json' file.
        ```


But if you get an error such as above, you'll need to provide a chat template. It's quite easy - let's see how to do it.







## Find the chat template format

So, how to find the chat template for a new model that you intend to use? 

This is normally listed in the model's page: search in that page for "template" and copy the listed Jinja template text.

If the template isn't directly listed in the model's page, you can look for a file named "tokenizer_config.json" in the main model files. This file should include an entry named "chat_template" which is what we want.


!!! info "Example of a tokenizer_config.json file"

    For example, in OpenChat's file "tokenizer_config.json":

    [https://huggingface.co/openchat/openchat-3.5-1210/blob/main/tokenizer_config.json](https://huggingface.co/openchat/openchat-3.5-1210/blob/main/tokenizer_config.json)

    You'll find this line with the template:

    ``` json
    {
        "...": "...",

        "chat_template": "{{ bos_token }}...{% endif %}",

        "...": "..."
    }
    ```

    The value in the "chat_template" key is the Jinja template that we're looking for.


Another alternative is to search online for the name of the model and "chat template".

Either way, once you know the template used by the model, you can set and use it.




## Option 1: Pass the chat template format when creating the model

Once you know the chat template definition you can create the model and pass it in the format argument. Let's assume you have a model file named "peculiar-model-7b.gguf":

``` python

chat_template = "{{ bos_token }}...{% endif %}"

model = LlamaCppModel("peculiar-model-7b.gguf",
                      format=chat_template)
```

And the model should now work without problems.




## Option 2: Add the chat template to the Models factory


If you plan to use the model many times, a more convenient solution is to create an entry in the "formats.json" file so that all further models with this name will use the template.


### With "sibila formats" CLI tool

Run the sibila CLI tool in the "models" folder:

```
> sibila formats -s peculiar peculiar-model "{{ bos_token }}...{% endif %}"

Using models directory '.'
Set format 'peculiar' with match='peculiar-model', template='{{ bos_token }}...'
```

First argument after -s is the format entry name, second the match regular expression (to identify the model filename) and finally the template. Help is available with "sibila formats --help".


### Manually edit "formats.json"

In alternative to using the sibila CLI tool, you can add the chat template format by creating an entry in a "formats.json" file, in the same folder as the model, with these fields:

``` json
{
    "peculiar": {
        "match": "peculiar-model",
        "template": "{{ bos_token }}...{% endif %}"
    }
}
```

The "match" field is regular expression that will be used to match the model name or filename. Field "template" is the chat template in Jinja format.

After configuring the template as we've seen above, all you need to do is to create a LlamaCppModel object and pass the model file path.

``` python
model = LlamaCppModel("peculiar-model-7b.gguf")
```

Note that we're not passing the format argument anymore when creating the model. The "match" regular expression we defined above will recognize the model from the filename and use the given chat template format.


!!! info "Base format definitions"

    Sibila includes by default the definitions of several well-known chat template formats. These definitions are available in "[sibila/base_formats.json](https://github.com/jndiogo/sibila/blob/main/sibila/base_formats.json)", and are automatically loaded when Models factory is created.
    
     You can add any chat template formats into your own "formats.json" files, but please never change the "sibila/base_formats.json" file, to avoid potential errors.




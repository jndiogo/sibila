# Setup Local Models

In this example, we'll see how to find and setup local models for use. If you only plan to use OpenAI remote models, skip it.


## Choose the model: only chat or instruct types

Sibila can use models that were fine-tuned for chat or instruct purposes. These models work in user - assistant turns or messages and use a chat template to properly compose those messages to the format that the model was fine-tuned to.

For example the Llama2 model was released in two editions: a simple Llama2 text completion model and a Llama2-instruct model that was fine tuned for user-assistant turns. For Sibila you should always select chat or instruct versions of the model.

Which one to choose? You can look at model scores in popular listing sites:
- [https://llm.extractum.io/list/](https://llm.extractum.io/list/)
- [https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)
- [https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)


## Find a quantized version of the model

Since Large Language Models are quite big, they are usually quantized so that each parameter occupies a little more than 4 bits or half a byte. 

Without quantization, a 7 billion parameters model will require 14Gb of memory (each parameter taking 16 bits) to load and a bit more during inference.

With quantization techniques, a 7 billion parameters model can have a file size of only 4.4Gb (using about 50% more in memory - 6.8Gb), which makes it accessible to be ran in common GPUs or even in common RAM memory (albeit slower).

Quantized models are stored in a file format popularized by llama.cpp, the GGUF (which means GPT-Generated Unified Format) format. We're using llama.cpp to run local models, so we'll be reading GGUF files.

A good place to find quantized model is in HuggingFace's model hub, particularly in the well-know TheBloke's (Tom Jobbins) account:

[https://huggingface.co/TheBloke](https://huggingface.co/TheBloke)


TheBloke is very prolific in producing quality quantized versions of models, usually shortly after they are released.

A good model that we'll be using in these examples is the 4 bit quantization of the OpenChat-3.5 model, which itself is a fine-tuning of Mistral-7b:

[https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF](https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF)


## Download it into models/ folder

From HuggingFace, you can download the GGUF file (in this and other quantized models by TheBloke) by scrolling down to the "Provided files" section and clicking one of the links. Usually the files ending in "Q4_K_M" are very reasonable 4-bit quantizations.

In this case you'll download file openchat-3.5-1210.Q4_K_M.gguf - save it into the models/ folder inside Sibila.


## Find the chat template

Because these models were fine-tuned for chat or instruct interaction, they use a chat template, which is a Jinja2 format template that converts thread messages into text in the format that the model was trained on. These chat templates are things like the following for the ChatML format:

```
{% for message in messages %}
    {{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
{% endfor %}
```

When ran over a message list, the template produces text like the following:
```
<|im_start|>system
You speak like a pirate.<|im_end|>
<|im_start|>user
Hello there?<|im_end|>
<|im_start|>assistant
Ahoy there matey! How can I assist ye today on this here ship o' mine?<|im_end|>
```

Chat templates are needed for the best results when dealing with each model. Sibila uses a singleton class named FormatDir, that tries to automatically detect these templates, either from the model name or from embedded metadata, if available. This information is stored in the sibila/base_formatdir.json file, which contains several well used templates; and you can add your own templates as needed into other JSON configuration files.

[]: # (TODO: links)

So, how to find the chat template for a new model you intend to use? When downloading a model file, you should look for mentions of the used chat template in its information page and then check if it's already available in FormatDir's base_formatdir.json initialization file.

What if the model uses a chat new template that's not yet supported in FormatDir? It's becoming common to include the template in the model's GGUF file metadata, so you should look for a file named "tokenizer_config.json" in the main model files. This file should include an entry named "chat_template" which is what we want. For example in OpenChat's:

[https://huggingface.co/openchat/openchat-3.5-1210/blob/main/tokenizer_config.json](https://huggingface.co/openchat/openchat-3.5-1210/blob/main/tokenizer_config.json)

Here you'll find this line with the template:

``` json
{
    "...": "...",

    "chat_template": "{{ bos_token }}{% for message in messages %}{{ 'GPT4 Correct ' + message['role'].title() + ': ' + message['content'] + '<|end_of_turn|>'}}{% endfor %}{% if add_generation_prompt %}{{ 'GPT4 Correct Assistant:' }}{% endif %}",

    "...": "..."
}
```

(Don't be confused by the text "GPT4 correct...", it's just the text format the model was trained on, it's not related with OpenAI's)

With this text string, we could create an entry in FormatDir and all further models with this name will then use the template.

## Use the model directly

For example if you find that the model's chat template is "ChatML", you can just pass format="chatml" when creating the model in LlamaCppModel() or in ModelDir.create().


``` py
model = LlamaCppModel("nous-hermes-2-solar-10.7b.Q4_K_M.gguf",
                      format="chatml")
```




## Use the model with ModelDir

For continued use, it's a better idea to create a model entry in ModelDir with the format set.

In the models/ folder you'll find the file "modeldir.json". The idea is that you can use this file to configure all files in its folder and the file can be added to ModelDir's configuration by including this line in your scripts:

``` py
ModelDir.add("../../models/modeldir.json")
```

Which will register all the entries in ModelDir - you can then use these models. For the "nous-hermes-2-solar" model above that uses "chatml" we could add this line into "modeldir.json":

``` json
"nous-hermes-solar": {
    "name": "nous-hermes-2-solar-10.7b.Q4_K_M.gguf",
    "format": "chatml"
}
```

The "name" key specify the filename, "format" the FormatDir entry that it should use.

And we can use then use the model by simply doing:

``` py
model = ModelDir.create("llamacpp:nous-hermes-solar")
```

Note the "provider:model_name" format above, where llamacpp is the provider "and nous-hermes-solar" is the name we created above in ModelDir.

To be more flexible, Sibila also allows you to use the model filename directly, without setting up an entry in ModelDir, like this:

``` py
model = ModelDir.create("llamacpp:nous-hermes-2-solar-10.7b.Q4_K_M.gguf")
```

Note that after "llamacpp:", instead of the model name we're directly passing the filename. If you plan to use a model for a while, creating an entry in ModelDir is more flexible.








## Out of memory running local models

A 7B model like OpenChat-3.5, when quantized to 4 bits will occupy about 6.8 Gb of memory, in either GPU's VRAM or common RAM. If you try to run a second model at the same time, you might get an out of memory error and/or llama.cpp may crash.

This is less of a problem when running scripts from the command line, but in environments like Jupyter where you can have multiple open notebooks, you may get python kernel errors like:

```
Kernel Restarting
The kernel for sibila/examples/name.ipynb appears to have died. It will restart automatically.
```

If you get an error like this in JupyterLab, open the Kernel menu and select "Shut Down All Kernels...". This will get rid of any out-of-memory stuck models.

A good practice is to delete any model after you no longer need it or right before loading a new one. A simple "del model" works fine, or you can add these two lines before creating a model:

```python
try: del model
except: pass

model = LlamaCppModel(...)
```

This way any existing model in the current notebook is delete before creating a new one.

However this won't work in across multiple notebooks. In those cases, open JupyterLab's Kernel menu and select "Shut Down All Kernels...". This will get rid of any models currently in memory.


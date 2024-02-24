# How to Setup Local Models

Most current 7B quantized models are pretty capable for common data extraction tasks. Below we'll see how to find and setup local models for use with Sibila. If you only plan to use OpenAI remote models, this is not for you.


## Default model used in the examples: OpenChat

By default, most of the examples included with Sibila use OpenChat, a quantized 7B parameters model, that you can download from:

[https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF/blob/main/openchat-3.5-1210.Q4_K_M.gguf](https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF/blob/main/openchat-3.5-1210.Q4_K_M.gguf)

In this page, click "download" and save it into the "models" folder inside the Sibila project. It's a 4.4Gb download and can take some time.

Once the file "openchat-3.5-1210.Q4_K_M.gguf" is placed in the "models" folder, you should be able to run the examples with this local model.

But you can also search for and use other local models - keep reading to learn more.



## Choose the model: chat or instruct types

Sibila can use models that were fine-tuned for chat or instruct purposes. These models work in user - assistant turns or messages and use a chat template to properly compose those messages to the format that the model was fine-tuned to.

For example, the Llama2 model was released in two editions: a simple Llama2 text completion model and a Llama2-instruct model that was fine tuned for user-assistant turns. For Sibila you should always select chat or instruct versions of a model.

But which model to choose? You can look at model benchmark scores in popular listing sites:

- [https://llm.extractum.io/list/](https://llm.extractum.io/list/)
- [https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)
- [https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)




## Find a quantized version of the model

Since Large Language Models are quite big, they are usually quantized so that each parameter occupies a little more than 4 bits or half a byte. 

Without quantization, a 7 billion parameters model would require 14Gb of memory (each parameter taking 16 bits) to load and a bit more during inference.

With quantization techniques, a 7 billion parameters model can have a file size of only 4.4Gb (using about 50% more in memory - 6.8Gb), which makes it accessible to be ran in common GPUs or even in common RAM memory (albeit slower).

Quantized models are stored in a file format popularized by llama.cpp, the GGUF format (which means GPT-Generated Unified Format). We're using llama.cpp to run local models, so we'll be needing GGUF files.

A good place to find quantized models is in HuggingFace's model hub, particularly in the well-know TheBloke's (Tom Jobbins) area:

[https://huggingface.co/TheBloke](https://huggingface.co/TheBloke)


TheBloke is very prolific in producing quality quantized versions of models, usually shortly after they are released.

A good model that we'll be using for the examples is the 4 bit quantization of the OpenChat-3.5 model, which itself is a fine-tuning of Mistral-7b:

[https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF](https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF)




## Download it into models/ folder

From HuggingFace, you can download the GGUF file (in this and other quantized models by TheBloke) by scrolling down to the "Provided files" section and clicking one of the links. Usually the files ending in "Q4_K_M" are very reasonable 4-bit quantizations.

In this case you'll download the file "openchat-3.5-1210.Q4_K_M.gguf" - save it into the "models" folder inside Sibila.




## Find the chat template

Because these models were fine-tuned for chat or instruct interaction, they use a chat template, which is a Jinja2 format template that converts thread messages into text in the format that the model was trained on. These chat templates are similar to the following one for the ChatML format:

```
{% for message in messages %}
    {{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
{% endfor %}
```

When ran over a message list with system, user and model messages, the template produces text like the following:
```
<|im_start|>system
You speak like a pirate.<|im_end|>
<|im_start|>user
Hello there?<|im_end|>
<|im_start|>assistant
Ahoy there matey! How can I assist ye today on this here ship o' mine?<|im_end|>
```

Specific chat templates are needed for the best results when dealing with each model. Sibila uses a singleton class named FormatDir, that tries to automatically detect which template to use with a model, either from the model name or from embedded metadata, if available. This information is stored in the sibila/base_formatdir.json file, which contains several well templates for well-known models; and you can add your own templates as needed into other JSON configuration files.

So, how to find the chat template for a new model you intend to use? When downloading a model file, you should look for mentions of the used chat template in its information page and then check if it's already available in FormatDir's base_formatdir.json initialization file.

What if the model uses a new chat template that's not yet supported in FormatDir? It's becoming common to include the template in the model's GGUF file metadata, so you should look for a file named "tokenizer_config.json" in the main model files. This file should include an entry named "chat_template" which is what we want. For example in OpenChat's tokenizer_config.json:

[https://huggingface.co/openchat/openchat-3.5-1210/blob/main/tokenizer_config.json](https://huggingface.co/openchat/openchat-3.5-1210/blob/main/tokenizer_config.json)

You'll find this line with the template:

``` json
{
    "...": "...",

    "chat_template": "{{ bos_token }}{% for message in messages %}...{% endif %}",

    "...": "..."
}
```

(Don't be confused by the text "GPT4 correct...", it's just the text format the model was trained on, and it's not related with OpenAI's)

With this text string, we could create an entry in FormatDir and all further models with this name will then use the template.





## Use the model directly

You can create the model by passing its filename to LlamaCppModel. Suppose we wanted to use the [Nous Hermes 2 Solar 10](https://huggingface.co/TheBloke/Nous-Hermes-2-SOLAR-10.7B-GGUF), we would download the file and:

``` py
model = LlamaCppModel("nous-hermes-2-solar-10.7b.Q4_K_M.gguf")
```

If automatic detection doesn't work and you receive an error that the chat template format is unknown: if you know the proper format name ("chatml" in this case), you can pass it in the format parameter:

``` py
model = LlamaCppModel("nous-hermes-2-solar-10.7b.Q4_K_M.gguf",
                      format="chatml")
```

Or if you know the chat template definition, you can also pass it in the format argument:

``` py

chat_template = "{{ bos_token }}{% for message in messages %}{{ 'GPT4 Correct ' + message['role'].title() + ': ' + message['content'] + '<|end_of_turn|>'}}{% endfor %}{% if add_generation_prompt %}{{ 'GPT4 Correct Assistant:' }}{% endif %}"

model = LlamaCppModel("nous-hermes-2-solar-10.7b.Q4_K_M.gguf",
                      format=chat_template)
```

But most of the time, Sibila should automatically detect the used format from the model's filename.



## Use the model with the Models class

For continued use, it's a better idea for to create an entry for the model in the Models singleton, instead of manually creating a LlamaCppModel object that loads a model file - this allows future model changing to be much easier.

Inside the "models" folder you'll find the file "models.json". The idea is that you can use this file to configure all files in its folder and the file can be added to Models' configuration by including this line in your scripts:

``` py
Models.setup("../../models")
```

This will register all the defined entries in Models. For the "nous-hermes-2-solar" model above that uses "chatml" we could add this line to "models.json":

``` json
"nous-hermes-solar": {
    "name": "nous-hermes-2-solar-10.7b.Q4_K_M.gguf",
    "format": "chatml"
}
```

The "name" key specify the filename, "format" the FormatDir entry that it should use.

And we can use then use the model by simply doing:

``` py
model = Models.create("llamacpp:nous-hermes-solar")
```

Note the "provider:model_name" format above, where llamacpp is the provider and "nous-hermes-solar" is the entry name we created above in Models.

To be more flexible, Sibila also allows you to use the model filename directly, without setting up an entry in Models, like this:

``` py
model = Models.create("llamacpp:nous-hermes-2-solar-10.7b.Q4_K_M.gguf")
```

Note that after "llamacpp:", instead of the model name we're directly passing the filename. If you plan to use a model for a while, creating an entry in Models is more flexible.





## Out of memory running local models

An important thing to know if you'll be using local models is about "Out of memory" errors.

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
except: ...

model = LlamaCppModel(...)
```

This way, any existing model in the current notebook is deleted before creating a new one.

However this won't work in across multiple notebooks. In those cases, open JupyterLab's Kernel menu and select "Shut Down All Kernels...". This will get rid of any models currently in memory.


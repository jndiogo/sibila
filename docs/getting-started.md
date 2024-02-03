# Getting Started


## Installation

Sibilia requires Python 3.9+ and uses the llama-cpp-python package for local models and OpenAI's API to access remote models like GPT-4.

You can run it in a plain CPU, CUDA GPU or other accelerator supported by llama.cpp.

For accelerated inference with local models, to take advantage of CUDA, Metal, etc, make sure you install llamacpp-python with the right settings - [see more info here](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#installation).

To use Sibila, download the repository and from the base directory (which has a setup.py script) do:

```
pip install -e .
```

You should now be able to use Sibila to get structured information from local or remote models.





## Using OPEN AI models

To use an OpenAI remote model, you'll need a paid OpenAI account and its API key. You can explicitely pass this key when creating an OpenAIModel object but this is not a good security practice. A better way is to define an environment variable which the OpenAI API will use when needed.

In Linux/Mac you can define this key by running:
```
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

And in Windows command prompt:

```
setx OPENAI_API_KEY "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

Having set this variable with your OpenAI API key, you can run an ["Hello Model" example](https://github.com/sibila/tree/master/examples/hellomodel/hello_openai.py) :

```python
import os
from sibila import (OpenAIModel, GenConf)

# make sure you set an environment variable named OPENAI_API_KEY with your API key.
model = OpenAIModel("gpt-3.5",
                    genconf=GenConf(temperature=1))

sys_text = "You are a helpful model that speaks like a pirate."
in_text = "Hello there?"

print(in_text)

text = model.query_gen(sys_text, in_text)

print(text)
```





## Using local models in llama.cpp

Sibila can use llama.cpp (via the llamacpp-python package) to load models from local GGUF format files. Since model files are quite big, they are usually quantized so that each parameter occupies a little more than 4 bits or half a byte. 

This means that a 7 billion parameter model can have a file size of only 4.4Gb (and about 50% more in memory - 6.8Gb), which makes it accessible to be ran in common GPUs or even in common RAM memory (albeit slower).

A great place to find quantized model is in HuggingFace's model hub, particularly in TheBloke's (Tom Jobbins) account:

[https://huggingface.co/TheBloke](https://huggingface.co/TheBloke)

Sibila can use models that were fine-tuned for chat or instruct purposes. These models work in user/assistant turns or messages and use a chat template to properly compose those messages to the format that the model was fine-tuned to. For example the Llama2 model was released in two editions: a simple Llama2 text completion model and a Llama2-instruct model that was fine tuned for user-assistant turns. For using Sibila you should always select chat or instruct versions of the model.

A good model that we can use for our examples is the 4 bit quantization of the OpenChat-3.5 model, which itself is a fine-tuning of Mistral-7b:

[https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF](https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF)

You can download the GGUF file (in this and other quantized models by TheBloke) by scrolling down to the "Provided files" section and clicking one of the links. Usually the files ending in "Q4_K_M" are reasonable 4-bit quantizations.

In this case you'll download file openchat-3.5-1210.Q4_K_M.gguf - save it into the models folder inside Sibila.

You can now run [hello-llammacpp.py](https://github.com/sibila/tree/master/examples/hellomodel/hello_llamacpp.py):

```python
import os
from sibila import (LlamaCppModel, GenConf)

model_path = "../../models/openchat-3.5-1210.Q4_K_M.gguf"

model = LlamaCppModel(model_path,
                      genconf=GenConf(temperature=1))

sys_text = "You are a helpful model that speaks like a pirate."
in_text = "Hello there?"

print(in_text)

text = model.query_gen(sys_text, in_text)

print(text)
```

After running the above and/or OpenAI's script you'll receive the model's answer to your "Hello there?" - in arrr-style:

```
Hello there?
Ahoy, me hearty! How be it goin'? Me name's Captain Chatbot, and I be here to assist thee with whatever ye need! So, what can me crew and I do fer yer today? Arrr!
```




## Out of memory running local models

A 7B model like OpenChat-3.5, when quantized to 4 bits will occupy about 6.8 Gb of memory, in either GPU's VRAM or common RAM. If you try to run a second model, you might get an out of memory error, or llama.cpp may crash.

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


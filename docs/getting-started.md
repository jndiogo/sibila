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

To use an OpenAI remote model, you'll need a paid OpenAI account and its API key. You can explicitly pass this key when creating an OpenAIModel object but this is not a good security practice. A better way is to define an environment variable which the OpenAI API will use when needed.

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
from sibila import OpenAIModel, GenConf

# make sure you set the environment variable named OPENAI_API_KEY with your API key.
# create an OpenAI model with generation temperature=1
model = OpenAIModel("gpt-4",
                    genconf=GenConf(temperature=1))

# the instructions or system command: speak like a pirate!
inst_text = "You speak like a pirate."

# the in prompt
in_text = "Hello there?"
print(in_text)

# query the model with instructions and in text
text = model.query_gen(inst_text, in_text)
print(text)
```

This will display 



## Using local models in llama.cpp

Sibila can use llama.cpp (via the llamacpp-python package) to load models from local GGUF format files. Since model files are quite big, they are usually quantized so that each parameter occupies less than a byte. 

See the Setup local models example to learn how where to find these models and how to use them in Sibilia, then return here to run the following script:

``` py
from sibila import LlamaCppModel, GenConf

# model file from the models folder
model_path = "../../models/openchat-3.5-1210.Q4_K_M.gguf"

# create an OpenAI model with generation temperature=1
model = LlamaCppModel(model_path,
                      genconf=GenConf(temperature=1))

# the instructions or system command: speak like a pirate!
inst_text = "You speak like a pirate."

# the in prompt
in_text = "Hello there?"
print(in_text)

# query the model with instructions and in text
text = model.query_gen(inst_text, in_text)
print(text)
```

The script is available here: [hello-llammacpp.py](https://github.com/sibila/tree/master/examples/hellomodel/hello_llamacpp.py)




# Arrrr-answer

After running the above and/or OpenAI's script you'll receive the model's answer to your "Hello there?" - in arrr-style:

```
Hello there?
Ahoy, me hearty! How be it goin'? Me name's Captain Chatbot, and I be here to assist thee with whatever ye need! So, what can me crew and I do fer yer today? Arrr!
```

Which means Sibila is working.



# Getting Started


## Installation

Sibila requires Python 3.9+ and uses the llama-cpp-python package for local models and OpenAI's API to access remote models like GPT-4.

You can run local models in a plain CPU, CUDA GPU or other accelerator supported by llama.cpp.

For local hardware accelerated inference, to take advantage of CUDA, Metal, etc, make sure you install llamacpp-python with the right settings - [see more info here](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#installation).

Install Sibila from PyPI by running:

```
pip install sibila
```

Alternatively you can install Sibila in edit mode by downloading the GitHub repository and running the following in the base folder of the repository:

```
pip install -e .
```

Either way you should now be able to use Sibila.



## Using OPEN AI models

To use an OpenAI remote model, you'll need a paid OpenAI account and its API key. You can explicitly pass this key when creating an OpenAIModel object but this is not a good security practice. A better way is to define an environment variable which the OpenAI API will use when needed.

In Linux/Mac you can define this key by running:
```
export OPENAI_API_KEY="..."
```

And in Windows command prompt:

```
setx OPENAI_API_KEY "..."
```

Having set this variable with your OpenAI API key, you can run an ["Hello Model" example](https://github.com/jndiogo/sibila/tree/main/examples/hello_model) :

```python
from sibila import OpenAIModel, GenConf

# model file from the models folder
model_path = "../../models/openchat-3.5-1210.Q4_K_M.gguf"

# make sure you set the environment variable named OPENAI_API_KEY with your API key.
# create an OpenAI model with generation temperature=1
model = OpenAIModel("gpt-4",
                    genconf=GenConf(temperature=1))

# the instructions or system command: speak like a pirate!
inst_text = "You speak like a pirate."

# the in prompt
in_text = "Hello there?"
print("User:", in_text)

# query the model with instructions and input text
text = model(in_text,
                inst=inst_text)
print("Model:", text)
```

This will generate a pirate response as seen below.




## Using local models in llama.cpp

Sibila can use llama.cpp (via the llamacpp-python package) to load models from local GGUF format files. Since LLM model files are quite big, they are usually quantized so that each parameter occupies less than a byte. 

See [Setup local models](setup-local-models.md) to learn how where to find these models and how to use them in Sibila, then return here to run the following script:

``` py
from sibila import LlamaCppModel, GenConf

# model file from the models folder
model_path = "../../models/openchat-3.5-1210.Q4_K_M.gguf"

# create a LlamaCpp model
model = LlamaCppModel(model_path,
                        genconf=GenConf(temperature=1))

# the instructions or system command: speak like a pirate!
inst_text = "You speak like a pirate."

# the in prompt
in_text = "Hello there?"
print("User:", in_text)

# query the model with instructions and input text
text = model(in_text,
                inst=inst_text)
print("Model:", text)
```

The script is available here: [hello_llamacpp.py](https://github.com/jndiogo/sibila/blob/main/examples/hello_model/hello_llamacpp.py)




# Arrr-answer!

After running the above and/or OpenAI's script you'll receive the model's answer to your "Hello there?" - in arrr-style:

```
User: Hello there?
Model: Ahoy, me hearty! How be it goin'? Me name's Captain Chatbot, and I be here to assist thee with whatever ye need! So, what can me crew and I do fer yer today? Arrr!
```

Which means Sibila is working. [Check the examples](https://github.com/jndiogo/sibila/tree/main/examples).

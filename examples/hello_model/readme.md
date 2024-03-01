# Hello model

In this example we see how to directly create local or remote model objects and later to do that more easily with the Models class. 


## Using a local model

To use a local model, make sure you download its GGUF format file and save it into the "../../models" folder.

In these examples, we'll use a [4-bit quantization of the OpenChat-3.5 7 billion parameters model](https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF), which at the current time is quite a good model for its size. 

The file is named "openchat-3.5-1210.Q4_K_M.gguf" and was downloaded from the above link. Make sure to save it into the "../../models" folder.

[See here for more information](https://jndiogo.github.io/sibila/models/local_model/) about setting up your local models.


With the model file in the "../../models" folder, we can run the following script:

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

Run the script above and after a few seconds (it has to load the model from disk), the good model answers back something like:

```
User: Hello there?
Model: Ahoy there matey! How can I assist ye today on this here ship o' mine?
Is it be treasure you seek or maybe some tales from the sea?
Let me know, and we'll set sail together!
```


## Using an OpenAI model

To use a remote model like GPT-4 you'll need a paid OpenAI account: https://openai.com/pricing

With an OpenAI account, you'll be able to generate an access token that you should [set into the OPENAI_API_KEY env variable](https://jndiogo.github.io/sibila/models/remote_model/). 

(An even better way is to use .env files with your variables, and use the [dotenv library](https://pypi.org/project/python-dotenv/) to read them.)

Once a valid OPENAI_API_KEY env variable is set, you can run this script:


``` py
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


We get back the usual funny pirate answer:

```
User: Hello there?
Model: Ahoy there, matey! What can this old sea dog do fer ye today?
```


## Using the Models directory

In these two scripts we created different objects to access the LLM model: LlamaCppModel and OpenAIModel. 

This was done to simplify, but a better way is to use the Models class directory.

Models is a singleton class that implements a directory of models where you can store file locations, configurations, aliases, etc.

After setting up a JSON configuration file you can have the Models class create models by using names like "llamacpp:openchat" or "openai:gpt-4" together with their predefined settings. This permits easy model change, comparing model outputs, etc.

In the scripts above, instead on instancing different classes for different models, we could use Models class to create the model from a name, by setting the model_name variable:

``` py
from sibila import Models, GenConf

# Using a local llama.cpp model: we first setup the ../../models directory:
# Models.setup("../../models")
# model_name = "llamacpp:openchat"

# OpenAI: make sure you set the environment variable named OPENAI_API_KEY with your API key.
model_name = "openai:gpt-4"

model = Models.create(model_name,
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

The magic happens in the line: 

``` py
model = Models.create(model_name, ...)
```

The Models class will take care of initializing the model based on the name you provide.

<!--TODO: Add link to Models example -->

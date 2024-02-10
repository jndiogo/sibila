# Hello Model!

In this example we see how to directly create the Model objects and then have ModelDir do that for us. 


## Using a local model

To use a local model, make sure you download its GGUF format file and save it into the "../../models" folder.

In this example, we use a [4-bit quantization of the OpenChat-3.5 7 billion parameters model](https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF), which at the current time is a good model for its size. 

The file is named "openchat-3.5-1210.Q4_K_M.gguf" and was downloaded from the above link. Make sure to save it into the "../../models" folder.

[See here for more information](https://jndiogo.github.io/sibila/setup-local-models/#default-model-used-in-the-examples-openchat) about setting up your local models.


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
print(in_text)

# query the model with instructions and in text
text = model.query_gen(inst_text, in_text)
print(text)
```

Run the script above and after a few seconds (it has to load the model from disk), the good model answers back something like:

```
Hello there?
Ahoy there matey! How can I assist ye today on this here ship o' mine? Is it be treasure you seek or maybe some tales from the sea? Let me know, and we'll set sail together!
```


## Using an OpenAI model

To use a remote model like GPT-4 you'll need a paid OpenAI account: https://openai.com/pricing

With an OpenAI account, you'll be able to generate an access token that you should [set into the OPENAI_API_KEY env variable](https://jndiogo.github.io/sibila/getting-started/#using-open-ai-models). 

(An even better way is to use .env files with your variables, and use the dotenv library.)

Once a valid OPENAI_API_KEY env variable is set, you can run this script:


``` py
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


We get back the funny pirate answer:

```
Hello there?
Ahoy there, matey! What can this old sea dog do fer ye today?
```


## Or using the model directory

In these two scripts we created different objects to access the LLM model: LlamaCppModel and OpenAIModel. 

This was done to simplify, but a better way is to use ModelDir, the model directory.

ModelDir is a singleton class that implements a directory of models where you can store file locations, configurations, aliases, etc.

After setting up the model in a JSON configuration file you can create models with names like "llamacpp:openchat" or "openai:gpt-4" together with their predefined settings. This permits easy model change, comparing model outputs, etc.

In the scripts above, instead on instancing different classes for different models, we could use ModelDir to create the model from a name, by setting the model_name variable:

``` py
from sibila import ModelDir, GenConf

# Using a llama.cpp model: we first add its JSON configuration file:
# ModelDir.add("../../models/modeldir.json")
# model_name = "llamacpp:openchat"

# OpenAI: make sure you set the environment variable named OPENAI_API_KEY with your API key.
model_name = "openai:gpt-4"

model = ModelDir.create(model_name)

# the instructions or system command: speak like a pirate!
inst_text = "You speak like a pirate."

# the in prompt
in_text = "Hello there?"
print(in_text)

# query the model with instructions and in text
text = model.query_gen(inst_text, in_text, genconf=GenConf(temperature=1))
print(text)
```

The magic happens in the line: 

``` py
model = ModelDir.create(model_name)
```

ModelDir will take care of initializing the model from a name in the model_name string.

<!--TODO: Add link to ModelDir example -->

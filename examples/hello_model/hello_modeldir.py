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
text = model.query_gen(inst_text, in_text)
print(text)

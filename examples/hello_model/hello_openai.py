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

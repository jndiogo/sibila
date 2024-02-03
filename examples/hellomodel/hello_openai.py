import os
from sibila import (OpenAIModel, GenConf)

# make sure you set an environment variable named OPENAI_API_KEY with your API key.
# using a .env file has many benefits - loading a .env from base folder if any:
env_path = "../../.env"
if os.path.isfile(env_path):
    from dotenv import load_dotenv
    assert load_dotenv(env_path, override=True, verbose=True)


model = OpenAIModel("gpt-3.5",
                    genconf=GenConf(temperature=1))

sys_text = "You are a helpful model that speaks like a pirate."
in_text = "Hello there?"

print(in_text)

text = model.query_gen(sys_text, in_text)

print(text)

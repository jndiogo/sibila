# load env variables from a .env if available:
env_path = "../../.env"
import os
if os.path.isfile(env_path):
    from dotenv import load_dotenv
    assert load_dotenv(env_path, override=True, verbose=True)


from sibila import ModelDir, GenConf
from sibila.tools import (
    interact
)


# delete any previous model
try: del model
except: ...

# to use a local model, assuming it's in ../../models/:
# add models folder config which also adds to ModelDir path
ModelDir.add("../../models/modeldir.json")
# set the model's filename - change to your own model
name = "llamacpp:openchat-3.5-1210.Q4_K_M.gguf"
model = ModelDir.create(name)

# to use an OpenAI model:
# model = ModelDir.create("openai:gpt-4")


print("Help available by typing '!'. Enter an empty line to quit.")

interact(model,
         inst_text="Be helpful.", # model instructions text, also known as system message
         genconf=GenConf(temperature=0.9))
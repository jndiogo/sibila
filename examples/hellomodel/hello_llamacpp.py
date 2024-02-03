import os
from sibila import (LlamaCppModel, GenConf)

# using a .env file has many benefits - loading a .env from base folder if any:
env_path = "../../.env"
if os.path.isfile(env_path):
    from dotenv import load_dotenv
    assert load_dotenv(env_path, override=True, verbose=True)


# 
model_path = "../../models/openchat-3.5-1210.Q4_K_M.gguf"

if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Could not find model file at '{model_path}'. Please download openchat-3.5-1210-GGUF model from:\n\n https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF/resolve/main/openchat-3.5-1210.Q4_K_M.gguf \n\nAfter downloading, save it into the ../../models folder and run this script again.")

    
model = LlamaCppModel(model_path,
                      genconf=GenConf(temperature=1))

sys_text = "You are a helpful model that speaks like a pirate."
in_text = "Hello there?"

print(in_text)

text = model.query_gen(sys_text, in_text)

print(text)

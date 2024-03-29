# load env variables like OPENAI_API_KEY from a .env file (if available)
try: from dotenv import load_dotenv; load_dotenv()
except: ...

if __name__ == "__main__":

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
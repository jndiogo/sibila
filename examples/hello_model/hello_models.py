# load env variables like OPENAI_API_KEY from a .env file (if available)
try: from dotenv import load_dotenv; load_dotenv()
except: ...

if __name__ == "__main__":

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

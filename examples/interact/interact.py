# load env variables like OPENAI_API_KEY from a .env file (if available)
try: from dotenv import load_dotenv; load_dotenv()
except: ...

if __name__ == "__main__":

    from sibila import Models, GenConf
    from sibila.tools import interact

    # delete any previous model
    try: del model
    except: ...

    # to use a local model, assuming it's in ../../models:
    # setup models folder:
    Models.setup("../../models")
    # set the model's filename - change to your own model
    model = Models.create("llamacpp:openchat-3.5-1210.Q4_K_M.gguf")

    # to use an OpenAI model:
    # model = Models.create("openai:gpt-4")

    print("Help available by typing '!'. Enter an empty line to quit.")

    interact(model,
            inst_text="Be helpful.", # model instructions text, also known as system message
            genconf=GenConf(temperature=0.9))
    
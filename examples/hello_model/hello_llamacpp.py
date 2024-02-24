if __name__ == "__main__":

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
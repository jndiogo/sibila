---
title: First run
---

## With a remote model

Sibila can use remote models from OpenAI, Anthropic, Mistral and other providers. For example, to use an OpenAI remote model, you'll need a paid OpenAI account and its API key. You can explicitly pass this key in your script but this is a poor security practice. 

A better way is to define an environment variable which the OpenAI API will use when needed:

=== "Linux and Mac"
    ```
    export OPENAI_API_KEY="..."
    ```

=== "Windows"
    ```
    setx OPENAI_API_KEY "..."
    ```

Having set this variable with your OpenAI API key, you can run a "Hello Model" like this:


!!! example
    ``` python
    from sibila import OpenAIModel, GenConf

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
    ```

    !!! success "Result"
        ```
        User: Hello there?
        Model: Ahoy there, matey! What can this old sea dog do fer ye today?
        ```


You're all set if you only plan to use remote OpenAI models.



## With a local model

Local models run from files in GGUF format which are loaded run by the llama.cpp component.

You'll need to download a GGUF model file: we suggest OpenChat 3.5 - an excellent 7B parameters quantized model that will run in less thant 7Gb of memory. 

To download the OpenChat model file, please see [Download OpenChat model](models/local_model.md#openchat-model).

After downloading the file, you can run this "Hello Model" script:



!!! example
    ``` python
    from sibila import LlamaCppModel, GenConf

    # model file from the models folder - change if different:
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
    ```

    !!! success "Result"
        ```
        User: Hello there?
        Model: Ahoy there matey! How can I assist ye today on this here ship o' mine?
        Is it be treasure you seek or maybe some tales from the sea?
        Let me know, and we'll set sail together!
        ```

If the above scripts output similar pirate talk, Sibila should be working fine.

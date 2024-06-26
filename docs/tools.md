---
title: Tools
---

The tools module includes some utilities to simplify common tasks.

## Interact

The [interact()](api-reference/tools.md#sibila.tools.interact) function allows a back-and-forth chat session. The user enters messages in an input() prompt and can use some special "!" commands for more functionality. The model answers back after each user message.

In a chat interaction, the model has to "remember" the previous messages exchanged. For this to work, a persistent context with the previous messages has to be provided to the model in each turn. This is done inside interact(), which can manage thread messages and delete older ones when the context maximum length is reached.

!!! example
    ``` python
    from sibila import Models, GenConf
    from sibila.tools import interact

    Models.setup("../models")
    model = Models.create("llamacpp:openchat")

    print("Help available by typing '!'. Enter an empty line to quit.")

    interact(model,
            inst_text="Be helpful.", # model instructions text, also known as system message
            genconf=GenConf(temperature=0.9))
    ```

    !!! success "Result"
        ```
        Help available by typing '!'. Enter an empty line to quit.

        > Hello there!

        Hello! How can I help you today?


        > I'm doing an example of this LLM access package called Sibila - have you heard about it?
        
        Yes, I am familiar with the AI language model named Sibila. It is a large language 
        model developed by a Spanish company called Wall Street Horus. Sibila is designed to 
        understand and generate text in multiple languages, particularly those spoken in 
        Spain and Latin America. It is similar to other language models like OpenAI's GPT-3 
        and Google's BERT, but it has been specifically fine-tuned for the Spanish language.
        If you have any questions about Sibila or need assistance related to it, feel free to ask!

        (NOTE: ignore the above model-generated paragraph, it's total hallucination)


        > No way! I've just searched and there's no such company nor model. 
        You are hallucinating, Mr. Model!

        Apologies for the confusion in my previous response. It appears that Sibila is not 
        an existing AI language model or a known company. I may have mixed up information 
        from different sources. If you have any other questions or need assistance with a 
        different topic, please feel free to ask!


        > No problem. See, I'm using an function of package Sibila that allows for maintaining 
        a chat with you. It stores all user and model messages in a sibila.Thread class.

        Got it! So, you are using a function from the Sibila package to have a conversation 
        with me, and it's storing our messages in a sibila.Thread object. If you need any help 
        or information during our chat, please don't hesitate to ask. I'm here to assist you!

        (Continues...)
        ```

Check the [Interact example](examples/interact.md) to see a chat created with this function.



## Multigen

The multigen module includes functions that build a table of answers to a list of questions, as generated by multiple models. This can be very helpful to compare how two or more models react to the same input. 

The [query_multigen()](api-reference/multigen.md#sibila.multigen.query_multigen) function generates a 2-D table of [ input , model ], where each row is the output from different models to the same question or input. Such table can be printed or saved as a CSV file.

See the [Compare example](examples/compare.md) for a side-by-side comparison of a local and a remote model, answering to the same queries.




<!-- TODO links -->
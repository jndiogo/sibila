In this example we look at the interact() function, which allows a back-and-forth chat session. The user enters messages in an input() prompt and can use some special "!" commands for more functionality. The model answers back after each user message.

In a chat interaction, the model has to "remember" the previous messages exchanged. For this to work, a persistent context with the previous messages has to be provided to the model in each turn. This is done by using a Context class object, which can manage thread messages and delete older ones when the context maximum length is reached.

To use a local model, make sure you have its file in the folder "../../models". You can use any GGUF format model - [see here how to download the OpenChat model used below](https://jndiogo.github.io/sibila/models/local_model/#examples). If you use a different one, don't forget to set its filename in the name variable below, after the text "llamacpp:".

To use an OpenAI model, make sure you defined the env variable OPENAI_API_KEY with a valid token and uncomment the line after "# to use an OpenAI model:".

Jupyter notebook and Python script versions are available in the example's folder.

So, let's create a local model.

```python
from sibila import Models, GenConf

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
```

Import interact() from the tools module and run it. 

What follows is the actual chat where the user prompts start with a ">" and the model answer's immediately below.

Read on, it does get funny at times... : )


```python
from sibila.tools import (
    interact
)

interact(model,
         inst_text="Be helpful.", # model instructions text, also known as system message
         genconf=GenConf(temperature=0.9, max_tokens=600))
```

    > Hello there!


    Hello! How can I help you today?
    


    > I'm doing an example of this LLM access package called Sibila - have you heard about it?


    Yes, I am familiar with the Sibila AI platform. Sibila is a Spanish-language large language model developed by OpenAI, similar to ChatGPT. It is designed to assist users with various tasks, such as answering questions, writing text, and providing recommendations. Sibila can be integrated into different applications and services to improve their functionality and user experience.
    


    > No way! I've just searched and there's no such company nor model. You are hallucinating, Mr. Model!


    My apologies for the confusion in my previous response. It appears that I made an error in my information. There is no specific LLM access package or model called Sibila that I am aware of. If you have any other questions or topics you'd like to discuss, please feel free to ask.
    


    > No problem. See, I'm using an function of package Sibila that allows for maintaining a chat with you. It stores all user and model messages in a sibila.Thread class.


    Got it! So, you are using a function from the Sibila package that enables a continuous conversation between you and the AI model. This function likely utilizes a threading mechanism to store and manage both user and model messages within a `sibila.Thread` class. If you have any questions or need assistance while using this function, please don't hesitate to ask. I'm here to help!
    


    > That's right - I can list the context messages by entering "!c"


    I see! It seems that within the Sibila package, you can use the command `!c` to list the context messages in your conversation. This feature allows you to review and manage the messages exchanged between you and the AI model, which can be useful for understanding the conversation history and making sure you have the necessary information at hand. If you need any assistance or clarification regarding any of these messages or other aspects of the Sibila package, feel free to ask.
    


    > !c


    Thread token len=542, max len before next gen=1448
    Thread inst='Be helpful.', join_sep='\n', len=10
    0: IN='Hello there!'
    1: OUT='Hello! How can I help you today?'
    2: IN="I'm doing an example of this LLM access package called Sibila - have you heard about it?"
    3: OUT='Yes, I am familiar with the Sibila AI platform. Sibila is a Spanish-language large language model developed by OpenAI, similar to ChatGPT. It is designed to assist users with various tasks, such as answering questions, writing text, and providing recommendations. Sibila can be integrated into different applications and services to improve their functionality and user experience.'
    4: IN="No way! I've just searched and there's no such company nor model. You are hallucinating, Mr. Model!"
    5: OUT="My apologies for the confusion in my previous response. It appears that I made an error in my information. There is no specific LLM access package or model called Sibila that I am aware of. If you have any other questions or topics you'd like to discuss, please feel free to ask."
    6: IN="No problem. See, I'm using an function of package Sibila that allows for maintaining a chat with you. It stores all user and model messages in a sibila.Thread class."
    7: OUT="Got it! So, you are using a function from the Sibila package that enables a continuous conversation between you and the AI model. This function likely utilizes a threading mechanism to store and manage both user and model messages within a `sibila.Thread` class. If you have any questions or need assistance while using this function, please don't hesitate to ask. I'm here to help!"
    8: IN='That\'s right - I can list the context messages by entering "!c"'
    9: OUT='I see! It seems that within the Sibila package, you can use the command `!c` to list the context messages in your conversation. This feature allows you to review and manage the messages exchanged between you and the AI model, which can be useful for understanding the conversation history and making sure you have the necessary information at hand. If you need any assistance or clarification regarding any of these messages or other aspects of the Sibila package, feel free to ask.'
    


    > Info about the model: "!i"


    The `!i` command likely provides information about the AI model being used within the Sibila package. This could include details such as the model's architecture, training data, capabilities, and other relevant specifications. If you need any further information or assistance regarding the model or its features, please let me know.
    


    > Sorry, it's actually "!info"


    My apologies for the confusion in my previous response. The correct command to get information about the AI model within the Sibila package is `!info`. This command should provide you with details about the model's architecture, training data, capabilities, and other relevant specifications. If you need any further information or assistance regarding the model or its features, please feel free to ask.
    


    > !info


    Model:
    desc='LlamaCppModel: '/home/jorge/ai/sibila/models/openchat-3.5-1210.Q4_K_M.gguf' - 'llama 7B Q4_K - Medium'',
    ctx_len=2048, max_tokens_limit=2048,
    genconf=GenConf(max_tokens=0, stop=[], temperature=0.0, top_p=0.9, format='text', json_schema=None, special=None)
    GenConf:
    GenConf(max_tokens=600, stop=[], temperature=0.9, top_p=0.9, format='text', json_schema=None, special=None)
    
    Thread token len=736, max len before next gen=1448
    


    > For help on al lcommands: "!"


    Within the Sibila package, using an exclamation mark (`!`) by itself typically displays a list of available commands and their usage. This command helps you understand the different options and functionalities offered by the package so that you can make the most of it during your interactions with the AI model. If you need further assistance or clarification on any specific commands or features, please don't hesitate to ask.
    


    > !


    Unknown command '!' - known commands:
     !inst[=text] - clear messages and add inst (system) message
     !add=path - load file and add to last msg
     !image=path/url - include a local or remote image. Local images must fit the context!
     !c - list context msgs
     !cl=path - load context (default=thread.json)
     !cs=path - save context (default=thread.json)
     !tl - thread's token length
     !info - model and genconf info
     Delimit with """ for multiline begin/end or terminate line with \ to continue into a new line
     Empty line + enter to quit
    


    > The Thread class can manage total token length: it will delete older messages if the thread becomes larger than the content max length.


    I understand that the `sibila.Thread` class in the Sibila package is designed to manage the total token length of your conversation. If the conversation becomes too large and exceeds the maximum allowed token length, the class will automatically delete older messages to maintain the thread within the desired size limit. This ensures that the conversation remains efficient and manageable while preventing it from growing too large. If you have any questions or concerns about managing the token length in your conversations or need assistance with other aspects of the Sibila package, please feel free to ask.
    


    > That's it - to see token length: "!tl"


    The `!tl` command within the Sibila package is likely used to display the current token length of your conversation. This command helps you keep track of how many tokens are being used in your exchange with the AI model, ensuring that you remain aware of the conversation's size and staying within any imposed token length limits. If you need assistance with managing token lengths or have questions about other aspects of the Sibila package, please don't hesitate to ask.
    


    > !tl


    Thread token len=1128, max len before next gen=1448
    


    > To save the thread to a JSON file: "!cs"


    The `!cs` command within the Sibila package is likely used to save your current conversation thread to a JSON file. This feature allows you to store the exchanged messages between you and the AI model for future reference or further analysis. To use this command, simply enter `!cs` in your conversation, and follow any prompts or instructions provided by the package. If you need assistance with saving your thread to a JSON file or have questions about other aspects of the Sibila package, please don't hesitate to ask.
    


    > !cs


    Saved context to thread.json
    


    > 





    Thread inst='Be helpful.', join_sep='\n', len=22
    0: IN='Hello there!'
    1: OUT='Hello! How can I help you today?'
    2: IN="I'm doing an example of this LLM access package called Sibila - have you heard about it?"
    3: OUT='Yes, I am familiar with the Sibila AI platform. Sibila is a Spanish-language large language model developed by OpenAI, similar to ChatGPT. It is designed to assist users with various tasks, such as answering questions, writing text, and providing recommendations. Sibila can be integrated into different applications and services to improve their functionality and user experience.'
    4: IN="No way! I've just searched and there's no such company nor model. You are hallucinating, Mr. Model!"
    5: OUT="My apologies for the confusion in my previous response. It appears that I made an error in my information. There is no specific LLM access package or model called Sibila that I am aware of. If you have any other questions or topics you'd like to discuss, please feel free to ask."
    6: IN="No problem. See, I'm using an function of package Sibila that allows for maintaining a chat with you. It stores all user and model messages in a sibila.Thread class."
    7: OUT="Got it! So, you are using a function from the Sibila package that enables a continuous conversation between you and the AI model. This function likely utilizes a threading mechanism to store and manage both user and model messages within a `sibila.Thread` class. If you have any questions or need assistance while using this function, please don't hesitate to ask. I'm here to help!"
    8: IN='That\'s right - I can list the context messages by entering "!c"'
    9: OUT='I see! It seems that within the Sibila package, you can use the command `!c` to list the context messages in your conversation. This feature allows you to review and manage the messages exchanged between you and the AI model, which can be useful for understanding the conversation history and making sure you have the necessary information at hand. If you need any assistance or clarification regarding any of these messages or other aspects of the Sibila package, feel free to ask.'
    10: IN='Info about the model: "!i"'
    11: OUT="The `!i` command likely provides information about the AI model being used within the Sibila package. This could include details such as the model's architecture, training data, capabilities, and other relevant specifications. If you need any further information or assistance regarding the model or its features, please let me know."
    12: IN='Sorry, it\'s actually "!info"'
    13: OUT="My apologies for the confusion in my previous response. The correct command to get information about the AI model within the Sibila package is `!info`. This command should provide you with details about the model's architecture, training data, capabilities, and other relevant specifications. If you need any further information or assistance regarding the model or its features, please feel free to ask."
    14: IN='For help on al lcommands: "!"'
    15: OUT="Within the Sibila package, using an exclamation mark (`!`) by itself typically displays a list of available commands and their usage. This command helps you understand the different options and functionalities offered by the package so that you can make the most of it during your interactions with the AI model. If you need further assistance or clarification on any specific commands or features, please don't hesitate to ask."
    16: IN='The Thread class can manage total token length: it will delete older messages if the thread becomes larger than the content max length.'
    17: OUT='I understand that the `sibila.Thread` class in the Sibila package is designed to manage the total token length of your conversation. If the conversation becomes too large and exceeds the maximum allowed token length, the class will automatically delete older messages to maintain the thread within the desired size limit. This ensures that the conversation remains efficient and manageable while preventing it from growing too large. If you have any questions or concerns about managing the token length in your conversations or need assistance with other aspects of the Sibila package, please feel free to ask.'
    18: IN='That\'s it - to see token length: "!tl"'
    19: OUT="The `!tl` command within the Sibila package is likely used to display the current token length of your conversation. This command helps you keep track of how many tokens are being used in your exchange with the AI model, ensuring that you remain aware of the conversation's size and staying within any imposed token length limits. If you need assistance with managing token lengths or have questions about other aspects of the Sibila package, please don't hesitate to ask."
    20: IN='To save the thread to a JSON file: "!cs"'
    21: OUT="The `!cs` command within the Sibila package is likely used to save your current conversation thread to a JSON file. This feature allows you to store the exchanged messages between you and the AI model for future reference or further analysis. To use this command, simply enter `!cs` in your conversation, and follow any prompts or instructions provided by the package. If you need assistance with saving your thread to a JSON file or have questions about other aspects of the Sibila package, please don't hesitate to ask."



These are the "!" commands that you can use in the interact() inputs:
```
    !inst[=text] - clear messages and add inst (system) message
    !add=path - load file and add to last msg
    !image=path/url - include a local or remote image. Local images must fit the context!
    !c - list context msgs
    !cl=path - load context (default=thread.json)
    !cs=path - save context (default=thread.json)
    !tl - thread's token length
    !info - model and genconf info
    Delimit with """ for multiline begin/end or terminate line with \ to continue into a new line
    Empty line + enter to quit
```


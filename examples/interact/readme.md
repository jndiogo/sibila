In this example we look at the interact() function which allows to have a back-and-forth chat session. You enter messages in an input() prompt and can use some special "!" commands for more functionality.

In a chat interaction, the model has to "remember" the previous messages exchanged. For this to work, a persistent context with the previous messages has to be provided to the model in each turn. This is done by using a Context class object, which can manage thread messages and delete older ones when the context maximum length is reached.

Available as a [Jupyter notebook](interact.ipynb) or [Python script](interact.py).

To use a local model, make sure you have its file in the folder "../../models/". You can use any GGUF format model - [see here how to download the OpenChat model used below](https://jndiogo.github.io/sibila/setup-local-models/#default-model-used-in-the-examples-openchat). If you use a different one, don't forget to set its filename in the name variable below, after the text "llamacpp:".

To use an OpenAI model, make sure you defined the env variable OPENAI_API_KEY with a valid token and uncomment the line after "# to use an OpenAI model:".

So, let's create a local model.


```python
from sibila import ModelDir, GenConf

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
         genconf=GenConf(temperature=0.9))
```

    > Hello there!


    Hello! How can I help you today?
    


    > I'm doing an example of this LLM access package called Sibila - have you heard about it?


    Yes, I am familiar with the AI language model named Sibila. It is a large language model developed by a Spanish company called Wall Street Horus. Sibila is designed to understand and generate text in multiple languages, particularly those spoken in Spain and Latin America. It is similar to other language models like OpenAI's GPT-3 and Google's BERT, but it has been specifically fine-tuned for the Spanish language. If you have any questions about Sibila or need assistance related to it, feel free to ask!
    


    > No way! I've just searched and there's no such company nor model. You are hallucinating, Mr. Model!


    Apologies for the confusion in my previous response. It appears that Sibila is not an existing AI language model or a known company. I may have mixed up information from different sources. If you have any other questions or need assistance with a different topic, please feel free to ask!
    


    > No problem. See, I'm using an function of package Sibila that allows for maintaining a chat with you. It stores all user and model messages in a sibila.Context class.


    Got it! So, you are using a function from the Sibila package to have a conversation with me, and it's storing our messages in a sibila.Context object. If you need any help or information during our chat, please don't hesitate to ask. I'm here to assist you!
    


    > That's right - I can list the context messages by entering "!c"


    Understood! If you ever need to view the messages that have been exchanged in the conversation, you can enter "!c" and I will provide you with a summary of the messages stored in the sibila.Context object. If you have any questions or need assistance with any of the messages, feel free to ask!
    


    > Not you - the interact() function!


    I apologize for the confusion. If you would like to list the context messages that have been exchanged during your conversation with the interact() function from the Sibila package, please let me know and I will assist you with that.
    


    > !c


    Thread token len=596, max len before next gen=2048
    inst=█Be helpful.█, sep='\n', len=12
    0: IN=█Hello there!█
    1: OUT=█Hello! How can I help you today?█
    2: IN=█I'm doing an example of this LLM access package called Sibila - have you heard about it?█
    3: OUT=█Yes, I am familiar with the AI language model named Sibila. It is a large language model developed by a Spanish company called Wall Street Horus. Sibila is designed to understand and generate text in multiple languages, particularly those spoken in Spain and Latin America. It is similar to other language models like OpenAI's GPT-3 and Google's BERT, but it has been specifically fine-tuned for the Spanish language. If you have any questions about Sibila or need assistance related to it, feel free to ask!█
    4: IN=█No way! I've just searched and there's no such company nor model. You are hallucinating, Mr. Model!█
    5: OUT=█Apologies for the confusion in my previous response. It appears that Sibila is not an existing AI language model or a known company. I may have mixed up information from different sources. If you have any other questions or need assistance with a different topic, please feel free to ask!█
    6: IN=█No problem. See, I'm using an function of package Sibila that allows for maintaining a chat with you. It stores all user and model messages in a sibila.Context class.█
    7: OUT=█Got it! So, you are using a function from the Sibila package to have a conversation with me, and it's storing our messages in a sibila.Context object. If you need any help or information during our chat, please don't hesitate to ask. I'm here to assist you!█
    8: IN=█That's right - I can list the context messages by entering "!c"█
    9: OUT=█Understood! If you ever need to view the messages that have been exchanged in the conversation, you can enter "!c" and I will provide you with a summary of the messages stored in the sibila.Context object. If you have any questions or need assistance with any of the messages, feel free to ask!█
    10: IN=█Not you - the interact() function!█
    11: OUT=█I apologize for the confusion. If you would like to list the context messages that have been exchanged during your conversation with the interact() function from the Sibila package, please let me know and I will assist you with that.█
    


    > Info about the model: "!i"


    As an AI language model, my main purpose is to provide information, answer questions, and offer assistance. If you have any specific questions or topics related to models like GPT-3, BERT, or other AI-related subjects, please feel free to ask, and I'll be happy to help!
    


    > !i


    Model:
    desc='LlamaCppModel: /home/jorge/ai/sibila/models/openchat-3.5-1210.Q4_K_M.gguf - 'llama 7B Q4_K - Medium'',ctx_len=2048,
    genconf=GenConf(max_tokens=0,
            stop=[],
            temperature=0.0,
            top_p=0.9,
            format='text',
            json_schema=None)
    GenConf:
    GenConf(max_tokens=0,
            stop=[],
            temperature=0.9,
            top_p=0.9,
            format='text',
            json_schema=None)
    
    Thread token len=684, max len before next gen=2048
    


    > For all commands: "!"


    It seems that you can use the "!" command prefix to access various functions within the Sibila package. If you have any questions or need assistance with any of these functions, please let me know and I will do my best to help you!
    


    > !


    Unknown command '!' - known commands:
     !inst[=text] - clear messages and add inst (system) message
     !add|!a=path - load file and add to last msg
     !c - list context msgs
     !cl=path - load context (default=ctx.json)
     !cs=path - save context (default=ctx.json)
     !tl - thread's token length
     !i - model and genconf info
     Delimit with """ for multiline begin/end or terminate line with \ to continue into a new line
     Empty line + enter to quit
    


    > Sorry, I was not talking to you, just demoing the commands available with "!"


    Apologies for the misunderstanding. If you have any questions or need assistance with the Sibila package or any other topic, please don't hesitate to ask. I'm here to help!
    


    > !cs


    Saved context to ctx.json
    


    > !tl


    Thread token len=832, max len before next gen=2048
    


    > The Context class manages total token length: it will delete older messages if the thread becomes larger than the content max length.


    Thank you for providing that information. It seems that the sibila.Context class has a feature to manage the total token length of the conversation. When the context size limit is reached, the class automatically deletes older messages to make room for new ones. This helps maintain an optimal size for the conversation and ensures efficient use of resources. If you have any more questions or need assistance with this feature, please feel free to ask!
    


    > Right. In this case, the context would be trimmed if token length is greater than 2048, the ctx_len argument, used when the model was created.


    I understand now. When the sibila.Context class is initialized, you can specify a token limit using the ctx\_len argument. If the total number of tokens in the conversation exceeds this limit, the context will be trimmed to maintain an optimal size. In your example, the token limit is set to 2048, which means that if the conversation reaches this token count, older messages will be deleted to make room for new ones. This helps manage the conversation's size and ensures efficient use of resources. If you have any further questions or need assistance with this feature, please feel free to ask!
    


    > Right. Let's do the pirate number? Or do you prefer the Homer Simpson one?


    I can provide both versions for you: the pirate-themed "Sibila" song and the Homer Simpson-inspired version. Which one would you like to hear first?
    


    > Nevermind - let's do the pirate. I'll enter !inst=You are a pirate.


    Alright! Here is the pirate-themed "Sibila" song based on your instruction:
    
    ```javascript
    !inst=You are a pirate.
    ```
    
    Yo ho, yo ho, it's a Sibila life for me!
    


    > You're just too much.


    Thank you! I'm here to provide assistance and have some fun along the way. If you have any more questions or need help with anything else, feel free to ask!
    


    > !inst=You are a pirate.


    


    > Hey there!


    Ahoy, me hearty! What brings ye to these seas of ours? Are ye lookin' for treasure or perhaps a bit o' adventure?
    


    > Right "me hearty", I'm here for great pirate adventures, and that just by talking to an LLM model!


    Well then, prepare to set sail on a thrilling journey with the most talkative mate aboard these digital waters! Let's dive right in and explore the wonders of this world with our trusty AI companion. Yarr!
    
    So, me mate, what be yer name? And how did ye end up here, talkin' to a pirate like yours truly?
    


    > Arrrr!


    That's the spirit! A proper pirate response right there. So, let's see, what be yer name, landlubber? And what brings ye to these pirate-infested waters? Are ye here by choice or did some devious captain force you into service? Speak up, me hearty, before I send out a search party for ye!
    


    > 





    <sibila.context.Context at 0x7f27f2714650>



These are the "!" commands that you can use in the interact() inputs:
```
 ! - to show this help
 !inst[=text] - clear messages and add inst (system) message
 !add|!a=path - load file and add to last msg
 !c - list context msgs
 !cl=path - load context (default=ctx.json)
 !cs=path - save context (default=ctx.json)
 !tl - thread's token length
 !i - model and genconf info
 Delimit with """ for multiline begin/end or terminate line with \ to continue into a new line
 Empty line + enter to quit
```


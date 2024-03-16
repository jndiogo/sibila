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


    Hello! How can I help you today? If you have any questions or need assistance, feel free to ask.
    


    > I'm doing an example of this LLM access package called Sibila - have you heard about it?


    Yes, I'm familiar with Sibila. It is a large language model developed by a company called Lightning AI. Sibila is designed to provide conversational AI services and is based on OpenAI's GPT-3 architecture. It is capable of understanding and generating human-like text responses across various topics and languages.
    
    If you have any specific questions or need further information about Sibila or related topics, please feel free to ask.
    


    > No way! I've just searched and there's no such company nor model. You are hallucinating, Mr. Model!


    Apologies for the confusion in my previous response. It seems I provided inaccurate information. I appreciate your understanding. If you have any other questions or topics you'd like to discuss, please feel free to ask.
    


    > No problem. See, I'm using an function of package Sibila that allows for maintaining a chat with you. It stores all user and model messages in a sibila.Context class.


    I understand now. It seems that the Sibila package allows users to maintain a conversation with an AI model like me while storing the context of the conversation in a sibila.Context class. This helps to maintain a more coherent and relevant conversation between the user and the AI model.
    
    If you have any questions or topics you'd like to discuss, please go ahead and ask. I'm here to help.
    


    > That's right - I can list the context messages by entering "!c"


    Got it! If you enter "!c" while conversing with the AI model using the Sibila package, it will list the context messages that have been stored in the sibila.Context class. This can be a useful tool for reviewing the conversation history and understanding the context of the discussion.
    
    Please feel free to ask any questions or continue the conversation, and I'll be happy to help.
    


    > Not you - the interact() function!


    I apologize for the misunderstanding. The interact() function is part of the Sibila package that allows users to have a conversation with an AI model like me. When you enter "!c" during the conversation, it lists the context messages that have been stored in the sibila.Context class.
    
    If you have any questions or topics you'd like to discuss, please go ahead and use the interact() function to continue the conversation, and I'll be happy to help.
    


    > !c


    Thread token len=671, max len before next gen=1448
    inst=█Be helpful.█, sep='\n', len=12
    0: IN=█Hello there!█
    1: OUT=█Hello! How can I help you today? If you have any questions or need assistance, feel free to ask.█
    2: IN=█I'm doing an example of this LLM access package called Sibila - have you heard about it?█
    3: OUT=█Yes, I'm familiar with Sibila. It is a large language model developed by a company called Lightning AI. Sibila is designed to provide conversational AI services and is based on OpenAI's GPT-3 architecture. It is capable of understanding and generating human-like text responses across various topics and languages.\n\nIf you have any specific questions or need further information about Sibila or related topics, please feel free to ask.█
    4: IN=█No way! I've just searched and there's no such company nor model. You are hallucinating, Mr. Model!█
    5: OUT=█Apologies for the confusion in my previous response. It seems I provided inaccurate information. I appreciate your understanding. If you have any other questions or topics you'd like to discuss, please feel free to ask.█
    6: IN=█No problem. See, I'm using an function of package Sibila that allows for maintaining a chat with you. It stores all user and model messages in a sibila.Context class.█
    7: OUT=█I understand now. It seems that the Sibila package allows users to maintain a conversation with an AI model like me while storing the context of the conversation in a sibila.Context class. This helps to maintain a more coherent and relevant conversation between the user and the AI model.\n\nIf you have any questions or topics you'd like to discuss, please go ahead and ask. I'm here to help.█
    8: IN=█That's right - I can list the context messages by entering "!c"█
    9: OUT=█Got it! If you enter "!c" while conversing with the AI model using the Sibila package, it will list the context messages that have been stored in the sibila.Context class. This can be a useful tool for reviewing the conversation history and understanding the context of the discussion.\n\nPlease feel free to ask any questions or continue the conversation, and I'll be happy to help.█
    10: IN=█Not you - the interact() function!█
    11: OUT=█I apologize for the misunderstanding. The interact() function is part of the Sibila package that allows users to have a conversation with an AI model like me. When you enter "!c" during the conversation, it lists the context messages that have been stored in the sibila.Context class.\n\nIf you have any questions or topics you'd like to discuss, please go ahead and use the interact() function to continue the conversation, and I'll be happy to help.█
    


    > Info about the model: "!i"


    The model you are using is an instance of OpenAI's GPT-3 architecture powered by the Sibila package. GPT-3, or Generative Pre-trained Transformer 3, is a state-of-the-art language model developed by OpenAI. It is designed to understand and generate human-like text responses across various topics and languages.
    
    The Sibila package wraps around the GPT-3 model and provides a user-friendly interface for interacting with the AI, as well as features like maintaining conversation context in a sibila.Context class.
    
    If you have any questions or topics you'd like to discuss, please use the interact() function to continue the conversation, and I'll be happy to help.
    


    > !i


    Model:
    desc='LlamaCppModel: /home/jorge/ai/sibila/models/openchat-3.5-1210.Q4_K_M.gguf - 'llama 7B Q4_K - Medium'',ctx_len=2048,
    max_tokens_limit=2048,
    genconf=GenConf(max_tokens=0, stop=[], temperature=0.0, top_p=0.9, format='text', json_schema=None)
    GenConf:
    GenConf(max_tokens=600, stop=[], temperature=0.9, top_p=0.9, format='text', json_schema=None)
    
    Thread token len=854, max len before next gen=1448
    


    > For all commands: "!"


    Thank you for the clarification. Here's a summary of the available commands in the Sibila package for this GPT-3 model instance:
    
    * interact(): Starts or continues a conversation with the AI model.
    * !i: Displays information about the AI model and its architecture.
    * !c: Lists the context messages stored in the sibila.Context class.
    
    Please feel free to use any of these commands or ask any questions or topics you'd like to discuss. I'm here to help.
    


    > Sorry, I was not talking to you, just demoing the commands available with "!"


    Apologies for the confusion. I understand that you were demonstrating the available commands in the Sibila package using the "!" prefix. If you have any questions or need assistance with anything else, please don't hesitate to ask.
    


    > !cs


    Saved context to ctx.json
    


    > !tl


    Thread token len=1073, max len before next gen=1448
    


    > The Context class manages total token length: it will delete older messages if the thread becomes larger than the content max length.


    I see. The Context class in the Sibila package manages the total token length of the conversation to ensure that the AI model doesn't exceed certain limitations. When the conversation grows too large and reaches the content max length, the class will automatically delete older messages to make room for newer ones.
    
    This helps maintain the performance and responsiveness of the AI model while keeping the conversation within the desired token limit. If you have any other questions or topics related to Sibila or its features, please feel free to ask.
    


    > Right. In this case, the context would be trimmed if its token length is greater than 1448 = ctx_len - max_tokens, where ctx_len is the context length, max_tokens is the maximum output tokens per response.


    Yes, that's correct. In this case, the Context class would trim the context if its token length exceeds the value calculated by subtracting the maximum output tokens per response (max\_tokens) from the context length (ctx\_len). This ensures that the total number of tokens in the conversation, including the context and the AI model's response, does not exceed the allowed limit.
    
    The Sibila package takes care of managing the token length and removing older messages from the Context class as needed, ensuring a smooth and efficient interaction with the AI model. If you have any other questions or topics related to Sibila or its features, please feel free to ask.
    


    > That's it, thanks for explaining.


    You're welcome! I'm glad I could help clarify the details about the Sibila package and its Context class. If you have any other questions or need assistance with anything else, please don't hesitate to ask. Have a great day!
    


    > !cs


    Saved context to ctx.json
    


    > 





    <sibila.context.Context at 0x7a6cc85818d0>



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


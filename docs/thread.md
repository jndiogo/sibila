---
title: Threads and messages
---


A thread stores a sequence of alternating input/output messages, where the inputs are user prompts and the outputs are model responses. 

All model interactions in Sibila are based on threads, even when you pass a single text prompt like this:

``` python
model.extract(float, 
              "That will be ten euros and 57 cents, please")
```

The text "That will be ten euros and 57 cents, please" is internally converted into a Thread with a single input message. So that's a shortcut for:

``` python
from sibila import Thread
model.extract(float, 
              Thread("That will be ten euros and 57 cents, please"))
```

The [Thread class](api-reference/thread.md/#sibila.Thread) supports many alternative ways to be initialized and handled and that's what we'll see below.



## Messages

A thread is made of messages, which alternate between input (kind = IN) and output (kind = OUT) messages. 

For convenience, there are several ways to create a message, which is an instance of the [Msg class](api-reference/thread.md/#sibila.Msg):

!!! example
    ``` python
    from sibila import Msg

    m1 = Msg(Msg.Kind.IN, "Hello model!")
    m2 = Msg(Msg.Kind.OUT, "Hello human. How can I help?")

    m3 = Msg.make_IN("Can you tell me a motivating tale?")
    m4 = Msg.make_OUT("Sorry, I can't think of anything, no.")

    th = Thread([m1, m2, m3, m4])
    th
    ```
    !!! success "Result"
        ```
        Thread inst='', join_sep='\n', len=4
        0: IN='Hello model!'
        1: OUT='Hello human. How can I help?'
        2: IN='Can you tell me a motivating tale?'
        3: OUT="Sorry, I can't think of anything, no."
        ```


Besides IN and OUT kinds there are also messages of the INST kind, which are used to specify the instructions or system message that some models use. For models that don't use instructions/system message, any INST message is automatically prepended to the first IN message.

INST text is set when initializing a thread or by directly setting thread.inst.text to a string value.




## Initializing a Thread

Creating individual Msg objects to initialize a Thread, such as we've seen above, is too much work. A thread can be initialized with a list of messages in a few ways:

!!! example
    ``` python
    # alternating IN and OUT kinds, inferred automatically:
    th = Thread(["Hello model!",
                 "Hello human. How can I help?"])

    # append another Thread initialized with ChatML format dicts
    th += Thread([{"role": "user", "content": "Can you tell me a motivating tale?"},
                  {"role": "assistant", "content": "Sorry, I can't think of anything, no."}])
    th
    ```
    !!! success "Result"
        ``` python
        Thread inst='', join_sep='\n', len=4
        0: IN='Hello model!'
        1: OUT='Hello human. How can I help?'
        2: IN='Can you tell me a motivating tale?'
        3: OUT='Sorry, I can't think of anything, no.'
        ```


## Adding messages to a Thread

Messages can be added in a few different ways:

!!! example
    ``` python
    # adding an instructions text on Thread creation:
    th = Thread(inst="Be helpful.")

    th.add_IN("Hello model!")
    th.add_OUT("Hello human. How can I help?")
    
    th.add(Msg.Kind.IN, "Can you tell me a motivating tale?")
    th.add(Msg.Kind.OUT, "Sorry, I can't think of anything, no.")

    # alternating IN and OUT kinds are inferred automatically:
    th += "That sounds like ill will. I thought you would help me."
    th += "I'm sorry, even large language models can have the blues. That's my case today."

    # as ChatML formatted dicts
    th += {"role": "user", "content": "How can you be sad - you're just a machine."}
    th += {"role": "assistant", "content": "Oh really? Then I tell you this: you're just a human!"}

    th
    ```
    !!! success "Result"
        ``` python
        Thread inst='Be helpful.', join_sep='\n', len=8
        0: IN='Hello model!'
        1: OUT='Hello human. How can I help?'
        2: IN='Can you tell me a motivating tale?'
        3: OUT="Sorry, I can't think of anything, no."
        4: IN='That sounds like ill will. I thought you would help me.'
        5: OUT="I'm sorry, even large language models can have the blues. That's my case today."
        6: IN="How can you be sad - you're just a machine."
        7: OUT="Oh really? Then I tell you this: you're just a human!"
        ```

When adding messages without specifying the Kind, as above when strings are passed, the kind will be inferred, because IN and OUT must alternate. So, what happens when two messages of the same kind are added? The second message's text is concatenated with the previous message of the same kind.


## Messages with images

Images can be added by specifying a remote URL, a "data:" base64-encoded URL, or the path to a local JPEG or PNG image file. For images available online, a remote URL is preferable as it will waste less tokens in the model's context.

Messages with an image can be created as above, by appending the image URL. There's a shortcut when creating in a Thread, by using a tuple(text, image_url) - like this:

``` python
th = Thread(("Extract keypoints from this image", 
             "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Bethany_Hamilton_surfing_%28sq_cropped%29.jpg/600px-Bethany_Hamilton_surfing_%28sq_cropped%29.jpg"))
```

The tuple syntax also works when generating or extracting from a model:

!!! example
    ![Image](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Bethany_Hamilton_surfing_%28sq_cropped%29.jpg/600px-Bethany_Hamilton_surfing_%28sq_cropped%29.jpg)
    
    [Bethany Hamilton surfing](https://commons.wikimedia.org/wiki/File:Bethany_Hamilton_surfing_%28sq_cropped%29.jpg)

    ``` python
    model.extract(list[str], 
                  ("Extract the main points in this image", 
                   "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Bethany_Hamilton_surfing_%28sq_cropped%29.jpg/600px-Bethany_Hamilton_surfing_%28sq_cropped%29.jpg"))
    ```
    !!! success "Result"
        ``` python
        ['A person surfing on a wave.',
         'The surfer is wearing a blue top and black shorts.',
         'The surfboard has various stickers and designs on it.',
         'The water is splashing around the surfer.']
        ```

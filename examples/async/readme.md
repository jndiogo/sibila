In this example we'll look at how to do multiple parallel requests to remote models by using Python's asyncio capabilities.

Generating from local llama.cpp models does not benefit from async functionality, because the local models must already be loaded in memory and can't benefit from asynchronous IO loading. When the async methods are used with a LlamaCppModel, inference will end up being made sequentially.

So we'll be using a remote OpenAI model. Make sure you defined the env variable OPENAI_API_KEY with a valid token.

This example is available as a Jupyter notebook or a Python script in this folder.

As usual, let's start by creating the model:


```python
# load env variables like OPENAI_API_KEY from a .env file (if available)
try: from dotenv import load_dotenv; load_dotenv()
except: ...

import time, asyncio

from sibila import Models

# delete any previous model
try: del model
except: ...

# to use a local model, assuming it's in ../../models:
# setup models folder:
# Models.setup("../../models")
# model = Models.create("llamacpp:openchat-3.5-1210.Q4_K_M.gguf", ctx_len=3072)

# to use an OpenAI model:
model = Models.create("openai:gpt-4")

# convenience time-counting functions:
start_time = None
def start_secs():
    global start_time
    start_time = time.time()
def secs(): 
    return f"{time.time() - start_time:.1f}"
```

We'll create two tasks that will run in parallel:
1. Ask the model to generate 20 names
2. Classify a phrase as spam

This example is running in a Jupyter notebook, so we can directly call the function with an await. In a python script we'd use asyncio.run() instead.

Note that we're using the _async suffix methods: extract_async() and classify_async(), instead of the normal functions.

The first task, generate 20 names:


```python
async def extract_names():    
    print("extract_names begin...", secs())
    
    names = await model.extract_async(list[str],
                                      "Generate 20 English names with first name and surname")
    
    print("...extract_names done", secs())
    
    return names

start_secs()
await extract_names()
```

    extract_names begin... 0.0
    ...extract_names done 4.4





    ['James Smith',
     'Michael Johnson',
     'Robert Williams',
     'Maria Garcia',
     'David Jones',
     'Jennifer Miller',
     'John Davis',
     'Patricia Wilson',
     'Daniel Anderson',
     'Elizabeth Taylor',
     'William Brown',
     'Barbara Moore',
     'Joseph Thompson',
     'Susan Martinez',
     'Charles Jackson',
     'Linda Harris',
     'Thomas Clark',
     'Jessica Lewis',
     'Christopher Walker',
     'Sarah Robinson']



The second task will classify a phrase as "spam"/"not spam":


```python
async def classify_spam():
    print("classify_spam begin...", secs())
    
    classification = await model.classify_async(["spam", "not spam"],
                                                "I am a Nigerian prince and will make you very rich!")
    
    print("...classify_spam done", secs())
    
    return classification

start_secs()
await classify_spam()
```

    classify_spam begin... 0.0
    ...classify_spam done 1.4





    'spam'



Let's use asyncio.as_completed(), to receive each task output, as soon as it's ready:


```python
async def run_tasks():
    print("as_complete begin---", secs())
    
    tasks = [extract_names(), classify_spam()]
    for task in asyncio.as_completed(tasks):
        res = await task
        print("Result:", res)
        
    print("---as_complete done", secs())

start_secs()
await run_tasks()
```

    as_complete begin--- 0.0
    extract_names begin... 0.0
    classify_spam begin... 0.0
    ...classify_spam done 1.0
    Result: spam
    ...extract_names done 5.8
    Result: ['James Smith', 'Emma Johnson', 'Olivia Williams', 'Liam Brown', 'Ava Jones', 'Noah Garcia', 'Sophia Miller', 'Mason Davis', 'Isabella Rodriguez', 'Ethan Martinez', 'Mia Hernandez', 'Logan Wilson', 'Charlotte Anderson', 'Aiden Moore', 'Harper Thomas', 'Lucas Jackson', 'Ella White', 'Benjamin Taylor', 'Amelia Harris', 'Alexander Clark']
    ---as_complete done 5.8


Follow the above begin/done print statements and the listed time in seconds, as they are printed.

Both tasks were started at the same time and classify_spam() terminated first (at the 1.0s mark), because it's a shorter task that simply outputs "spam"/"not spam".

On the meanwhile, the model worked on generating the 20 names that we requested with extract_names(), a longer operation which terminates later (at 4.8s).

In the same manner any other tasks could be run in parallel by using the *_async() methods of the model classes.

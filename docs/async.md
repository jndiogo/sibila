---
title: Asynchronous use
---

All the model calls like extract(), classify, json() etc, are also available in an asynchronous version of the same name but ending in _async, for example extract_async(). For example:

!!! example
    ``` python
    import asyncio

    from sibila import Models

    model = Models.create("openai:gpt-4")

    async def extract_names():    
        return await model.extract_async(list[str],
                                         "Generate 20 English names with first name and surname")

    async def classify_spam():
        return await model.classify_async(["spam", "not spam"],
                                          "I am a Nigerian prince and will make you very rich!")

    async def run_tasks():
        tasks = [extract_names(), classify_spam()]
        for task in asyncio.as_completed(tasks):
                res = await task
                print("Result:", res)

    asyncio.run(run_tasks()) # or in Jupyter: await run_tasks()
    ```

    !!! success "Result"
        ```
        Result: spam
        Result: ['John Smith', 'Emily Johnson', 'Michael Brown', 'Jessica Williams', 
        'David Jones', 'Sarah Davis', 'Daniel Miller', 'Laura Wilson', 'James Taylor', 
        'Sophia Anderson', 'Christopher Thomas', 'Emma Thompson', 'Joseph White', 
        'Olivia Lewis', 'Andrew Harris', 'Isabella Clark', 'Matthew Robinson', 
        'Ava Hall', 'Ethan Allen', 'Mia Wright']
        ```

The first result, with only one of two tokens generated is quickly fetched from the model, while the 20 generated names take a while and arrive later. See the [Async example](examples/async.md) to play with the above code.

Asynchronous access has many advantages when parallel requests are needed, allowing responses to be handled as soon as they are ready, instead of sequentially sending and waiting for the model responses.



## Local llama.cpp models

Using LlamaCppModel objects to generate locally does not benefit from async functionality, because the local models must already be loaded in memory and can't benefit from asynchronous IO loading. When the async class methods are used with LlamaCppModel, inference will end up being made sequentially.


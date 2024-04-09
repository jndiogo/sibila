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
# the transcript is large, so we'll create the model with a context length of 3072, which should be enough.
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


# the first task, generate 20 names:
async def extract_names():    
    print("extract_names begin...", secs())
    
    names = await model.extract_async(list[str],
                                      "Generate 20 English names with first name and surname")
    
    print("...extract_names done", secs())
    
    return names


# the second task will classify a phrase as "spam"/"not spam":
async def classify_spam():
    print("classify_spam begin...", secs())
    
    classification = await model.classify_async(["spam", "not spam"],
                                                "I am a Nigerian prince and will make you very rich!")
    
    print("...classify_spam done", secs())
    
    return classification


# run both tasks in parallel:
async def run_tasks():
    print("as_complete begin---", secs())
    
    tasks = [extract_names(), classify_spam()]
    for task in asyncio.as_completed(tasks):
        res = await task
        print("Result:", res)
        
    print("---as_complete done", secs())


if __name__ == "__main__":
    start_secs()
    asyncio.run(run_tasks())
    

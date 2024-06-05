"""
Requires a defined env variable TOGETHER_API_KEY with a valid API key.
See:
https://docs.together.ai/docs/inference-models
"""

import pytest

import os, subprocess, shutil, json, asyncio, time
from typing import Any, Optional, Union, Literal, Annotated, get_origin, get_args
from itertools import product

import logging
logger = logging.getLogger(__name__)

from dotenv import load_dotenv


from sibila import (
    Model,
    Models,
    LlamaCppModel,
    GenConf
)

from .utils import setup_env_models, teardown_env_models, setup_model, teardown_model
from .utils import run_cmd, run_text, run_json





models_dir = None

@pytest.fixture(autouse=True, scope="module")
def get_models_dir(pytestconfig):    
    global models_dir

    # print("PRE", Models.info())
    # Models.clear()
    # print("POST", Models.info())

    models_dir = pytestconfig.getoption("models_dir")
    if not models_dir:
        models_dir = "../../models"

    if not os.path.isabs(models_dir):
        base_dir = os.path.dirname(__file__)
        models_dir = os.path.normpath(os.path.join(base_dir, models_dir))




def create_model(filename: str) -> Model:

    path = os.path.join(models_dir, filename)
    model = LlamaCppModel(path, 
                          ctx_len=IN_CTX_LEN)
    return model





IN_CTX_LEN = 2048

models = [
    "Hermes-2-Pro-Mistral-7B.Q4_K_M.gguf",
    "llama-2-7b-chat.Q4_K_M.gguf",
    "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
    "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    "openchat-3.5-0106.Q4_K_M.gguf",
    "openchat-3.5-1210.Q4_K_M.gguf",
    "phi-2.Q5_K_M.gguf",
    "Phi-3-mini-4k-instruct-q4.gguf",
    "stablelm-2-12b-chat-Q4_K_M.gguf",
    "stablelm-zephyr-3b.Q4_K_M.gguf",
    "starling-lm-7b-alpha.Q5_K_M.gguf",
    "Starling-LM-7B-beta-Q4_K_M.gguf",
    "TinyDolphin-2.8-1.1b.Q4_K_M.gguf",
    "zephyr-7b-beta.Q4_K_M.gguf",
]

__models = [
]

"""
Models that don't pass these tests:
"dolphin-2_6-phi-2.Q4_K_M.gguf",
"gemma-2b-it-q4_k_m.gguf",
"gemma-2b-it-q8_0.gguf",
"kunoichi-dpo-v2-7b.Q4_K_M.gguf",
"mistral-7b-instruct-v0.2.Q5_K_M.gguf",
"openchat-3.6-8b-20240522-Q4_K_M.gguf",
"qwen1_5-0_5b-chat-q4_k_m.gguf",
"rocket-3b.Q4_K_M.gguf",
"stablelm-2-zephyr-1_6b-Q4_K_M.gguf",
"tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
"zephyr-7b-gemma-v0.1-Q4_K_M.gguf",
"""







@pytest.mark.parametrize("filename", models)
def test_all_for_model(filename):
    print("\n########### test_all_for_model", filename)

    model = create_model(filename)


    run_lower_in(model)

    run_extract(model)

    asyncio.run( run_async_extract1(model) )
    asyncio.run( run_async_extract2(model) )
    asyncio.run( run_async_extract3(model) )


    del model












# ============================================================================== lower_in_ops tests
# query, in response.lower()
lower_in_ops = [
    ("Tell me briefly about oranges", "orange"),
    ("Where are the Azores?", "atlantic ocean"),
]


def run_lower_in(model: Model):

    for op in lower_in_ops:
        text = model(op[0])
        print(text)
        assert op[1] in text.lower(), op






# ============================================================================== extract
# query, type, value
extract_ops = [
    ("Extract the number from the following text: there are twelve bananas", int, 12),
    ("The contrary of False is?", bool, True),
]


def run_extract(model: Model):

    for extract in extract_ops:
        res = model.extract(extract[1], extract[0])
        assert res == extract[2], extract





# =================================== extract async

async def run_async_extract1(model: Model):
    print("run_async begin")

    for extract in extract_ops:

        res = await model.extract_async(extract[1], extract[0])
        assert res == extract[2], extract

    print("run_async done")
        




async def _run_async(index: int,
                    model: Model):
    print(f"run {index} begin")

    extract = extract_ops[index]
    res = await model.extract_async(extract[1], extract[0])
    assert res == extract[2], extract
    
    print(f"run {index} done")


async def run_async_extract2(model: Model):

    print("gather begin")

    tasks = [_run_async(index, model) for index in range(len(extract_ops))]
    await asyncio.gather(*tasks)

    print("gather done")


async def run_async_extract3(model: Model):

    print("as_complete begin")

    tasks = [_run_async(index, model) for index in range(len(extract_ops))]

    for task in asyncio.as_completed(tasks):
        await task

    print("as_complete done")



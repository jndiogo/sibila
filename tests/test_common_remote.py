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
    AnthropicModel,
    GenConf
)

from .utils import setup_env_models, teardown_env_models, setup_model, teardown_model
from .utils import run_cmd, run_text, run_json



@pytest.fixture(autouse=True, scope="module")
def env_model():
    load_dotenv(override=True, verbose=True)




IN_CTX_LEN = 2048
OUT_MAX_TOKENS = 1024

models = [
    "anthropic:claude-3-haiku-20240307",
    "fireworks:accounts/fireworks/models/mixtral-8x7b-instruct",
    "groq:llama3-70b-8192",
    "mistral:mistral-small-latest",
    "openai:gpt-3.5",
    "openai:gpt-4",
    "together:mistralai/Mixtral-8x7B-Instruct-v0.1",
]

__models = [
    "groq:llama3-70b-8192",
]




def limit_rate(what: Union[str,Model]):
    if (isinstance(what, str) and "anthropic" in what or
        isinstance(what, AnthropicModel)):
        time.sleep(60/5)
        
def create_model(res_name: str) -> Model:
    model = Models.create(res_name, 
                          ctx_len=IN_CTX_LEN, 
                          max_tokens_limit=OUT_MAX_TOKENS)
    return model


# ============================================================================== lower_in_ops tests
# query, in response.lower()
lower_in_ops = [
    ("Tell me briefly about oranges", "orange"),
    ("Where are the Azores?", "atlantic ocean"),
]


@pytest.mark.parametrize("res_name, op", product(models, lower_in_ops))
def test_lower_in(res_name, op):

    model = create_model(res_name)

    limit_rate(res_name)

    text = model(op[0])
    print(text)
    assert op[1] in text.lower()








# ============================================================================== extract
# query, type, value
extract_ops = [
    ("Extract the number from the following text: there are twelve bananas", int, 12),
    ("Yes, that's true.", bool, True),
]


@pytest.mark.parametrize("res_name, extract", product(models, extract_ops))
def test_extract(res_name, extract):

    model = create_model(res_name)

    limit_rate(res_name)

    res = model.extract(extract[1], extract[0])

    assert res == extract[2]







# =================================== extract async

@pytest.mark.parametrize("res_name", models)
async def test_async_extract1(res_name):

    model = create_model(res_name)

    print("run_async begin")

    for extract in extract_ops:
        limit_rate(res_name)

        res = await model.extract_async(extract[1], extract[0])
        assert res == extract[2]

    print("run_async done")
        




async def _run_async(index: int,
                    model: Model):
    print(f"run {index} begin")

    limit_rate(model)

    extract = extract_ops[index]
    res = await model.extract_async(extract[1], extract[0])
    assert res == extract[2]
    
    print(f"run {index} done")



@pytest.mark.parametrize("res_name", models)
async def test_async_extract2(res_name):

    model = create_model(res_name)

    print("gather begin")

    tasks = [_run_async(index, model) for index in range(len(extract_ops))]
    await asyncio.gather(*tasks)

    print("gather done")



@pytest.mark.parametrize("res_name", models)
async def test_async_extract3(res_name):

    model = create_model(res_name)

    print("as_complete begin")

    tasks = [_run_async(index, model) for index in range(len(extract_ops))]

    for task in asyncio.as_completed(tasks):
        await task

    print("as_complete done")






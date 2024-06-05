"""
Requires a defined env variable OPENAI_API_KEY with a valid OpenAI API key.
"""

import pytest

import os, subprocess, shutil, json, asyncio
from typing import Any, Optional, Union, Literal, Annotated, get_origin, get_args

import logging
logger = logging.getLogger(__name__)
# pytest --log-cli-level=DEBUG

from dotenv import load_dotenv

from sibila import (
    Models,
    OpenAIModel,
    GenConf,
    Thread,
    Msg
)



from .utils import setup_env_models, teardown_env_models, setup_model, teardown_model
from .utils import run_cmd, run_text, run_json


MODEL_NAME = "gpt-4o"


DO_TEARDOWN = True

@pytest.fixture(autouse=True, scope="module")
def env_model():

    load_dotenv()

    base_dir, models_dir = setup_env_models("openai-" + MODEL_NAME, 
                                            change_cwd=True,
                                            full_clean=False)

    ret = base_dir, models_dir
    print("---> setup", ret)
    
    yield ret

    # --------------------------- teardown
    if DO_TEARDOWN == False:
        return

    print("---> teardown", ret)

    teardown_env_models(base_dir)









describe_ops = [
    ("../res/coypu.jpg", "boat"),
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/Singapura_Cats.jpg/320px-Singapura_Cats.jpg", "cats")
]

def test_describe(env_model):

    model = OpenAIModel(MODEL_NAME)

    for op in describe_ops:
        th = Thread(Msg.make_IN("Describe this image", op[0]))
        res = model(th)
        print(res)
        assert op[1] in res.lower()


count_ops = [
    ("How many elements in this image?", "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/Singapura_Cats.jpg/320px-Singapura_Cats.jpg", 2),
    ("How many persons in this image?", "https://upload.wikimedia.org/wikipedia/en/6/6e/VUMillennium.jpg", 4)
]

def test_count(env_model):

    model = OpenAIModel(MODEL_NAME)

    for op in count_ops:
        th = Thread(Msg.make_IN(op[0], op[1]))
        res = model.extract(int, th)
        print(res)
        assert res == op[2]



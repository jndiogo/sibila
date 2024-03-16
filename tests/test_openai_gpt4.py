import pytest

import os, subprocess, shutil, json
from typing import Any, Optional, Union, Literal, Annotated, get_origin, get_args

import logging
logger = logging.getLogger(__name__)
# pytest --log-cli-level=DEBUG

from dotenv import load_dotenv

from sibila import (
    Models,
    OpenAIModel,
    GenConf
)



from .utils import setup_env_models, teardown_env_models, setup_model, teardown_model
from .utils import run_cmd, run_text, run_json


MODEL_NAME = "gpt-4"
CTX_LEN = 8192


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











def test_create_openai(env_model):

    model = OpenAIModel(MODEL_NAME)
    del model

    with pytest.raises(NameError):
        model = OpenAIModel(MODEL_NAME + "NOT_THERE")
        del model





def test_models(env_model):

    with pytest.raises(NameError):
        res_name = MODEL_NAME
        model = Models.create(res_name)
        del model


    with pytest.raises(NameError):
        res_name = "openai:NOT_THERE"
        model = Models.create(res_name)
        del model

    with pytest.raises(NameError):
        res_name = "NOT_THERE"
        model = Models.create(res_name)
        del model



    Models.setup("models")

    res_name = "openai:" + MODEL_NAME
    model = Models.create(res_name)
    del model


    Models.setup("models", clear=True)

    res_name = "openai:" + MODEL_NAME
    model = Models.create(res_name)
    del model






def test_ctx_len(env_model):
    
    model = OpenAIModel(MODEL_NAME)
    # print(model.ctx_len, model.max_tokens_limit)
    assert model.ctx_len == CTX_LEN
    assert model.max_tokens_limit == CTX_LEN
    del model


    model = OpenAIModel(MODEL_NAME,
                        ctx_len=0)
    assert model.ctx_len == CTX_LEN
    assert model.max_tokens_limit == CTX_LEN
    del model

    model = OpenAIModel(MODEL_NAME,
                        ctx_len=1024)
    assert model.ctx_len == 1024
    assert model.max_tokens_limit == 1024
    del model






def test_max_tokens(env_model):
    
    model = OpenAIModel(MODEL_NAME)

    assert model.calc_max_max_tokens(0) == CTX_LEN
    assert model.calc_max_max_tokens(500) == CTX_LEN-500
    assert model.calc_max_max_tokens(3000) == CTX_LEN-3000
    assert model.calc_max_max_tokens(16000) == CTX_LEN-16000

    genconf=GenConf(max_tokens=1000)    
    assert genconf.resolve_max_tokens(model.ctx_len, model.max_tokens_limit) == 1000
    assert model.resolve_genconf_max_tokens(100, genconf) == 1000
    assert model.resolve_genconf_max_tokens(1500, genconf) == 1000

    genconf=GenConf(max_tokens=-20)
    assert genconf.resolve_max_tokens(model.ctx_len, model.max_tokens_limit) == int(CTX_LEN * 20/100)
    assert model.resolve_genconf_max_tokens(100, genconf) == int(model.ctx_len * 20/100)
    assert model.resolve_genconf_max_tokens(1900, genconf) == int(CTX_LEN * 20/100)

    genconf=GenConf(max_tokens=0)    
    assert genconf.resolve_max_tokens(model.ctx_len, model.max_tokens_limit) == CTX_LEN
    assert model.resolve_genconf_max_tokens(100, genconf) == CTX_LEN - 100
    assert model.resolve_genconf_max_tokens(1900, genconf) == CTX_LEN - 1900
    
    with pytest.raises(ValueError):
        assert model.resolve_genconf_max_tokens(16000, genconf) == CTX_LEN - 16000

    del model






def test_prompt(env_model):

    PROMPT = "Tell me briefly about oranges"
    HAS = "orange"

    model = OpenAIModel(MODEL_NAME)
    text = model(PROMPT)
    print(text)
    assert HAS in text.lower()
    del model





def test_extract(env_model):

    model = OpenAIModel(MODEL_NAME)

    res = model.extract(int, "twelve bananas")
    assert res == 12

    res = model.extract(bool, "Yes I got it!")
    assert res == True

    del model


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
    GenConf
)



from .utils import setup_env_models, teardown_env_models, setup_model, teardown_model
from .utils import run_cmd, run_text, run_json


MODEL_NAME = "gpt-4"
IN_CTX_LEN = 128000
OUT_MAX_TOKENS = 4096


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


    # no longer raises: inner OpenAI object is created in first call:
    # with pytest.raises(NameError):
    #    model = OpenAIModel(MODEL_NAME + "NOT_THERE")
    #    del model




def test_models(env_model):

    with pytest.raises(NameError):
        res_name = MODEL_NAME
        model = Models.create(res_name)
        del model

    # no longer raises: inner OpenAI object is created in first call:
    # with pytest.raises(NameError):
    #    res_name = "openai:NOT_THERE"
    #    model = Models.create(res_name)
    #    del model

    # no longer raises: inner OpenAI object is created in first call:
    # with pytest.raises(NameError):
    #    res_name = "NOT_THERE"
    #    model = Models.create(res_name)
    #    del model


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
    assert model.ctx_len == IN_CTX_LEN
    assert model.max_tokens_limit == OUT_MAX_TOKENS
    del model


    model = OpenAIModel(MODEL_NAME,
                        ctx_len=0)
    assert model.ctx_len == IN_CTX_LEN
    assert model.max_tokens_limit == OUT_MAX_TOKENS
    del model

    model = OpenAIModel(MODEL_NAME,
                        ctx_len=1024)
    assert model.ctx_len == 1024
    assert model.max_tokens_limit == 1024
    del model






def test_max_tokens(env_model):
    
    model = OpenAIModel(MODEL_NAME)

    assert model.calc_max_max_tokens(0) == OUT_MAX_TOKENS
    assert model.calc_max_max_tokens(500) == OUT_MAX_TOKENS
    assert model.calc_max_max_tokens(3000) == OUT_MAX_TOKENS
    assert model.calc_max_max_tokens(16000) == OUT_MAX_TOKENS

    genconf=GenConf(max_tokens=1000)    
    assert genconf.resolve_max_tokens(model.ctx_len, model.max_tokens_limit) == 1000
    assert model.resolve_genconf_max_tokens(100, genconf) == 1000
    assert model.resolve_genconf_max_tokens(1500, genconf) == 1000

    genconf=GenConf(max_tokens=-20)
    assert genconf.resolve_max_tokens(model.ctx_len, model.max_tokens_limit) == OUT_MAX_TOKENS
    assert model.resolve_genconf_max_tokens(100, genconf) == OUT_MAX_TOKENS
    assert model.resolve_genconf_max_tokens(1900, genconf) == OUT_MAX_TOKENS

    genconf=GenConf(max_tokens=0)    
    assert genconf.resolve_max_tokens(model.ctx_len, model.max_tokens_limit) == OUT_MAX_TOKENS
    assert model.resolve_genconf_max_tokens(100, genconf) == OUT_MAX_TOKENS
    assert model.resolve_genconf_max_tokens(1900, genconf) == OUT_MAX_TOKENS
    
    with pytest.raises(ValueError):
        assert model.resolve_genconf_max_tokens(999*1000, genconf)

    del model




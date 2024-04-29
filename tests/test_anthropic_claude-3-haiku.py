"""
Requires a defined env variable ANTHROPIC_API_KEY with a valid key.
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
    AnthropicModel,
    GenConf
)



from .utils import setup_env_models, teardown_env_models, setup_model, teardown_model
from .utils import run_cmd, run_text, run_json


MODEL_NAME = "claude-3-haiku-20240307"
IN_CTX_LEN = 200000
OUT_MAX_TOKENS = 4096


DO_TEARDOWN = True

@pytest.fixture(autouse=True, scope="module")
def env_model():

    load_dotenv(override=True, verbose=True)

    base_dir, models_dir = setup_env_models("mistral-" + MODEL_NAME, 
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







def test_create_anthropic(env_model):

    model = AnthropicModel(MODEL_NAME)
    del model

    # models are only checked when used, so no NameError
    # with pytest.raises(NameError):
    model = AnthropicModel(MODEL_NAME + "NOT_THERE")
    del model





def test_models(env_model):

    with pytest.raises(NameError):
        res_name = MODEL_NAME
        model = Models.create(res_name)
        del model

    # models are only checked when used, so no NameError
    # with pytest.raises(NameError):
    res_name = "anthropic:NOT_THERE"
    model = Models.create(res_name)
    del model

    with pytest.raises(NameError):
        res_name = "NOT_THERE"
        model = Models.create(res_name)
        del model



    Models.setup("models")

    res_name = "anthropic:" + MODEL_NAME
    model = Models.create(res_name)
    del model


    Models.setup("models", clear=True)

    res_name = "anthropic:" + MODEL_NAME
    model = Models.create(res_name)
    del model






def test_ctx_len(env_model):
    
    model = AnthropicModel(MODEL_NAME)
    # print(model.ctx_len, model.max_tokens_limit)
    assert model.ctx_len == IN_CTX_LEN
    assert model.max_tokens_limit == OUT_MAX_TOKENS
    del model


    model = AnthropicModel(MODEL_NAME,
                           ctx_len=0)
    assert model.ctx_len == IN_CTX_LEN
    assert model.max_tokens_limit == OUT_MAX_TOKENS
    del model

    model = AnthropicModel(MODEL_NAME,
                           ctx_len=1024)
    assert model.ctx_len == 1024
    assert model.max_tokens_limit == 1024
    del model






def test_max_tokens(env_model):
    
    model = AnthropicModel(MODEL_NAME)

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






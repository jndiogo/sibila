"""
Requires a defined env variable TOGETHER_API_KEY with a valid API key.
See:
https://docs.together.ai/docs/inference-models
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
    FireworksModel,
    GenConf,
    GenError
)



from .utils import setup_env_models, teardown_env_models, setup_model, teardown_model
from .utils import run_cmd, run_text, run_json


MODEL_NAME = "accounts/fireworks/models/mixtral-8x7b-instruct"
IN_CTX_LEN = 32768
OUT_MAX_TOKENS = IN_CTX_LEN


DO_TEARDOWN = True

@pytest.fixture(autouse=True, scope="module")
def env_model():

    load_dotenv(override=True, verbose=True)

    base_dir, models_dir = setup_env_models("fireworks-mixtral", 
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





def create_model(ctx_len: int = IN_CTX_LEN):
    model = FireworksModel(MODEL_NAME, 
                          ctx_len=ctx_len)
    return model





def test_create_together(env_model):

    model = create_model()
    del model

    # models are only checked when used, so no NameError
    # with pytest.raises(NameError):
    model = FireworksModel(MODEL_NAME + "NOT_THERE")
    del model





def test_models(env_model):

    with pytest.raises(NameError):
        res_name = MODEL_NAME
        model = Models.create(res_name)
        del model


    # models are only checked when used, so no NameError
    # with pytest.raises(NameError):
    res_name = "together:NOT_THERE"
    model = Models.create(res_name)
    del model

    with pytest.raises(NameError):
        res_name = "NOT_THERE"
        model = Models.create(res_name)
        del model



    Models.setup("models")

    res_name = "together:" + MODEL_NAME
    model = Models.create(res_name)
    del model


    Models.setup("models", clear=True)

    res_name = "together:" + MODEL_NAME
    model = Models.create(res_name)
    del model






def test_ctx_len(env_model):
    
    model = create_model()
    # print(model.ctx_len, model.max_tokens_limit)
    assert model.ctx_len == IN_CTX_LEN
    assert model.max_tokens_limit == OUT_MAX_TOKENS
    del model

    """ defaults depend on base_models.json: don't test
    model = FireworksModel(MODEL_NAME,
                           ctx_len=0)
    assert model.ctx_len == IN_CTX_LEN
    assert model.max_tokens_limit == OUT_MAX_TOKENS
    del model
    """

    model = FireworksModel(MODEL_NAME,
                           ctx_len=1024)
    assert model.ctx_len == 1024
    assert model.max_tokens_limit == 1024
    del model






def test_max_tokens(env_model):
    
    model = create_model()

    assert model.calc_max_max_tokens(0) == OUT_MAX_TOKENS
    assert model.calc_max_max_tokens(500) == OUT_MAX_TOKENS - 500
    assert model.calc_max_max_tokens(3000) == OUT_MAX_TOKENS - 3000
    assert model.calc_max_max_tokens(16000) == OUT_MAX_TOKENS - 16000

    genconf=GenConf(max_tokens=1000)    
    assert genconf.resolve_max_tokens(model.ctx_len, model.max_tokens_limit) == 1000
    assert model.resolve_genconf_max_tokens(100, genconf) == 1000
    assert model.resolve_genconf_max_tokens(1500, genconf) == 1000

    genconf=GenConf(max_tokens=-20)
    max_tokens = int(OUT_MAX_TOKENS * 20 / 100)
    assert genconf.resolve_max_tokens(model.ctx_len, model.max_tokens_limit) == max_tokens
    assert model.resolve_genconf_max_tokens(100, genconf) == max_tokens
    assert model.resolve_genconf_max_tokens(1900, genconf) == max_tokens

    genconf=GenConf(max_tokens=0)    
    assert genconf.resolve_max_tokens(model.ctx_len, model.max_tokens_limit) == OUT_MAX_TOKENS
    assert model.resolve_genconf_max_tokens(100, genconf) == OUT_MAX_TOKENS - 100
    assert model.resolve_genconf_max_tokens(1900, genconf) == OUT_MAX_TOKENS - 1900
    
    with pytest.raises(ValueError):
        assert model.resolve_genconf_max_tokens(160*1000, genconf)

    del model


import pytest

import os, subprocess, shutil, json, asyncio
from typing import Any, Optional, Union, Literal, Annotated, get_origin, get_args

import logging
logger = logging.getLogger(__name__)
# pytest --log-cli-level=DEBUG

from sibila import (
    Models,
    LlamaCppModel,
    GenConf
)


from .utils import setup_env_models, teardown_env_models, setup_model, teardown_model
from .utils import run_cmd, run_text, run_json




# https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
MODEL_FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

DO_TEARDOWN = True

@pytest.fixture(autouse=True, scope="module")
def env_model():

    base_dir, models_dir = setup_env_models("llamacpp", 
                                            change_cwd=True,
                                            full_clean=False)

    model_path = setup_model(MODEL_FILENAME, 
                             models_dir,
                             always_copy=False)

    ret = base_dir, model_path
    print("---> setup", ret)
    
    yield ret

    # --------------------------- teardown
    if DO_TEARDOWN == False:
        return

    print("---> teardown", ret)

    teardown_model(model_path)

    teardown_env_models(base_dir)










def test_create_llamacppmodel(env_model):

    full_model_path = env_model[1]
    rel_model_path = os.path.join("models", MODEL_FILENAME)

    # model file not found
    with pytest.raises(NameError):
        model = LlamaCppModel(full_model_path + "NOT_THERE")

    # relative path
    model = LlamaCppModel(rel_model_path, format="zephyr")
    del model

    # absolute path
    model = LlamaCppModel(full_model_path)
    del model





def test_models(env_model):

    base_dir = env_model[0]
    models_dir_path = os.path.join(base_dir, "models")
    models_json_path = os.path.join(models_dir_path, "models.json")

    full_model_path = env_model[1]
    rel_model_path = os.path.join("models", MODEL_FILENAME)

    name = "local-model"
    res_name = f"llamacpp:{name}"
    link_name = f"{name}-link"
    link_res_name = f"llamacpp:{link_name}"


    # avoid any existing models.json
    if os.path.isfile(models_json_path):
        os.remove(models_json_path)

    # create model entry
    cmd = f"sibila models -m models/ -s {res_name} {MODEL_FILENAME} zephyr"
    run_json(cmd, 
             path=models_json_path,
             expected_json={
                    "llamacpp": {
                        name: {
                            "name": MODEL_FILENAME,
                            "format": "zephyr"
                        }
                    }
                })

    # create model link
    cmd = f"sibila models -m models/ -sl {link_res_name} {name}"
    run_json(cmd, 
             path=models_json_path,
             expected_json={
                    "llamacpp": {
                        name: {
                            "name": MODEL_FILENAME,
                            "format": "zephyr"
                        },
                        link_name: name
                    }
                })
    



    # model file not found
    with pytest.raises(NameError):
        model = Models.create(res_name + "NOT_THIS")

    # Models.setup() not called ===============================================
    with pytest.raises(NameError):
        model = Models.create(res_name)

    # rel_path
    model = Models.create("llamacpp:" + rel_model_path)
    del model

    # absolute path
    model = Models.create("llamacpp:" + full_model_path)
    del model


    # Models.setup() called ===================================================
    Models.setup("models")

    model = Models.create(res_name)
    del model

    model = Models.create(link_res_name)
    del model







def test_ctx_len(env_model):
    
    rel_model_path = os.path.join("models", MODEL_FILENAME)

    model = LlamaCppModel(rel_model_path, format="zephyr")
    assert model.ctx_len == 2048
    assert model.max_tokens_limit == 2048
    del model


    model = LlamaCppModel(rel_model_path, format="zephyr",
                          ctx_len=0)
    assert model.ctx_len == 2048
    assert model.max_tokens_limit == 2048
    del model

    model = LlamaCppModel(rel_model_path, format="zephyr",
                          ctx_len=1024)
    assert model.ctx_len == 1024
    assert model.max_tokens_limit == 1024
    del model






def test_max_tokens(env_model):
    
    rel_model_path = os.path.join("models", MODEL_FILENAME)

    model = LlamaCppModel(rel_model_path, format="zephyr")
    CTX_LEN = 2048

    assert model.calc_max_max_tokens(0) == CTX_LEN-0
    assert model.calc_max_max_tokens(500) == CTX_LEN-500
    assert model.calc_max_max_tokens(3000) == CTX_LEN-3000


    genconf=GenConf(max_tokens=1000)    
    assert genconf.resolve_max_tokens(model.ctx_len, model.max_tokens_limit) == 1000
    assert model.resolve_genconf_max_tokens(100, genconf) == 1000
    assert model.resolve_genconf_max_tokens(1500, genconf) == CTX_LEN - 1500

    genconf=GenConf(max_tokens=-20)
    assert genconf.resolve_max_tokens(model.ctx_len, model.max_tokens_limit) == int(CTX_LEN * 20/100)
    assert model.resolve_genconf_max_tokens(100, genconf) == int(CTX_LEN * 20/100)
    assert model.resolve_genconf_max_tokens(1900, genconf) == CTX_LEN - 1900

    genconf=GenConf(max_tokens=0)    
    assert genconf.resolve_max_tokens(model.ctx_len, model.max_tokens_limit) == 2048
    assert model.resolve_genconf_max_tokens(100, genconf) == CTX_LEN - 100
    assert model.resolve_genconf_max_tokens(1900, genconf) == CTX_LEN - 1900

    del model





def test_prompt(env_model):

    rel_model_path = os.path.join("models", MODEL_FILENAME)

    PROMPT = "Tell me about oranges"
    HAS = "orange"

    model = LlamaCppModel(rel_model_path, format="zephyr")
    text = model(PROMPT)
    assert HAS in text.lower()
    del model




INT_PROMPT = "extract a number from the following text: there are 12 bananas"
TRUE_PROMPT = "extract a True/False value from the following text: Yes I got it!"



def test_extract(env_model):

    rel_model_path = os.path.join("models", MODEL_FILENAME)

    model = LlamaCppModel(rel_model_path, format="zephyr")


    res = model.extract(int, INT_PROMPT)
    assert res == 12

    res = model.extract(bool, TRUE_PROMPT)
    assert res == True

    del model







def test_extract_async1(env_model):

    rel_model_path = os.path.join("models", MODEL_FILENAME)

    model = LlamaCppModel(rel_model_path, format="zephyr")

    async def run_async():
        print("run_async begin")

        res = await model.extract_async(int, INT_PROMPT)
        assert res == 12

        res = await model.extract_async(bool, TRUE_PROMPT)
        assert res == True
        print("run_async done")
        
    asyncio.run(run_async())



    async def run1_async():        
        res = await model.extract_async(int, INT_PROMPT)
        assert res == 12
        print("run1 done")

    async def run2_async():
        res = await model.extract_async(bool, TRUE_PROMPT)
        assert res == True
        print("run2 done")


    async def as_completed():
        print("as_complete begin")
        tasks = [run1_async(), run2_async()]
        for task in asyncio.as_completed(tasks):
            await task
        print("as_complete done")
            
    asyncio.run(as_completed())    



    async def gather():
        print("gather begin")
        tasks = [run1_async(), run2_async()]
        await asyncio.gather(*tasks)
        print("gather done")
            
    asyncio.run(gather())    


    del model






async def test_extract_async2(env_model):

    rel_model_path = os.path.join("models", MODEL_FILENAME)

    model = LlamaCppModel(rel_model_path, format="zephyr")

    print("run_async begin")

    res = await model.extract_async(int, INT_PROMPT)
    assert res == 12

    res = await model.extract_async(bool, TRUE_PROMPT)
    assert res == True
    print("run_async done")
        

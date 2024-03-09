import pytest

import os, subprocess, shutil, json
from typing import Any, Optional, Union, Literal, Annotated, get_origin, get_args

import logging
logging.basicConfig(level=logging.DEBUG)

from sibila import (
    Models,
    LlamaCppModel,
    GenConf
)



FOLDER = "llamacpp"
MODEL_FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
DO_TEARDOWN = True

@pytest.fixture(autouse=True, scope="module")
def model_dir():

    base_dir = os.path.dirname(__file__)

    res_dir = os.path.join(base_dir, "res")
    if not os.path.isdir(res_dir):
        assert False, f"Resource dir missing: '{res_dir}'"

    src_model_path = os.path.join(res_dir, MODEL_FILENAME)
    if not os.path.isfile(src_model_path):
        assert False, f"Model missing at: '{src_model_path}'"
    

    base_dir = os.path.join(base_dir, FOLDER)

    if os.path.isdir(base_dir):
        assert base_dir.endswith(FOLDER) # the paranoid
        shutil.rmtree(base_dir)

    os.mkdir(base_dir)    

    models_dir = os.path.join(base_dir, "models")
    os.mkdir(models_dir)

    # copy model
    dest_mode_path = os.path.join(models_dir, MODEL_FILENAME)
    print(f"Copying model to '{dest_mode_path}'")
    shutil.copyfile(src_model_path, dest_mode_path)

    old_cwd = os.getcwd()
    os.chdir(base_dir)

    ret = base_dir, models_dir, dest_mode_path, os.path.join("models", MODEL_FILENAME)
    print("setup --->", ret)
    
    yield ret

    if DO_TEARDOWN == False:
        return

    print("teardown")

    if os.path.isdir(base_dir):
        assert base_dir.endswith(FOLDER) # the paranoid
        shutil.rmtree(base_dir)

    os.chdir(old_cwd)
    



def test_creation(model_dir):

    full_model_path = model_dir[2]
    rel_model_path = model_dir[3]


    PROMPT = "Tell me about oranges"
    # tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf includes bad chat_template metadata="llama"
    GOOD_ANSWER = """Orange is a type of citrus fruit that originated in the Mediterranean region."""


    with pytest.raises(ValueError):
        model = LlamaCppModel(full_model_path + "NOT_THERE")


    model = LlamaCppModel(full_model_path)
    text = model(PROMPT)
    assert not text.startswith(GOOD_ANSWER)
    del model


    model = LlamaCppModel(rel_model_path, format="zephyr")
    text = model(PROMPT)
    assert text.startswith(GOOD_ANSWER)
    del model






def test_extract(model_dir):

    rel_model_path = model_dir[3]

    model = LlamaCppModel(rel_model_path, format="zephyr")


    res = model.extract(int, "twelve bananas")
    assert res == 12

    res = model.extract(bool, "Yes I got it!")
    assert res == True



    del model


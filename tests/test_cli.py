import pytest

import os, subprocess, shutil, json
from typing import Any, Optional, Union, Literal, Annotated, get_origin, get_args

import logging
logging.basicConfig(level=logging.DEBUG)

from sibila import Models

from .utils import setup_env_models, teardown_env_models, setup_model, teardown_model
from .utils import run_cmd, run_text, run_json



DO_TEARDOWN = True

@pytest.fixture(autouse=True, scope="module")
def env():

    base_dir, models_dir = setup_env_models("cli", 
                                            change_cwd=True,
                                            full_clean=True)

    print("---> setup", base_dir)
    
    yield base_dir, models_dir

    # --------------------------- teardown
    if DO_TEARDOWN == False:
        return

    print("---> teardown", base_dir)

    teardown_env_models(base_dir)
    




def test_models(env):

    models_path = os.path.join(env[1], Models.MODELS_CONF_FILENAME)

    cmd = "sibila models -m models/ -s llamacpp:local-model local-model.gguf"
    run_json(cmd, 
             path=models_path,
             expected_json={
                    "llamacpp": {
                        "local-model": {
                            "name": "local-model.gguf"
                        }
                    }
                })


    cmd = "sibila models -m models/ -f llamacpp:local-model chatml"
    run_json(cmd, 
             path=models_path,
             expected_json={
                    "llamacpp": {
                        "local-model": {
                            "name": "local-model.gguf",
                            "format": "chatml"
                        }
                    }
                })


    cmd = "sibila models -m models/ -d llamacpp:local-model"
    run_cmd(cmd)


    cmd = "sibila models -m models/ -s llamacpp:local-model local-model.gguf chatml"
    run_json(cmd, 
             path=models_path,
             expected_json={
                    "llamacpp": {
                        "local-model": {
                            "name": "local-model.gguf",
                            "format": "chatml"
                        }
                    }
                })


    cmd = "sibila models -m models/ -sl llamacpp:local-model-link local-model"
    dic = run_json(cmd, 
                   path=models_path)
    assert ("llamacpp" in dic and 
            "local-model-link" in dic["llamacpp"] and 
            dic["llamacpp"]["local-model-link"] == "local-model")

    cmd = "sibila models -m models/ -l"
    res = run_text(cmd,
                   expected_text=[
                       "only local models",
                        "local-model",
                        "local-model.gguf",
                        "chatml",
                        "local-model-link"                       
                   ],
                   not_expected_text=["gpt-4"])


    cmd = "sibila models -m models/ -l local-model-link"
    res = run_text(cmd,
                   expected_text=[
                       "only local models",
                       "local-model-link",
                       "local-model"
                   ],
                   not_expected_text=["local-model.gguf",
                                      "chatml",
                                      "gpt-4"])


    cmd = "sibila models -m models/ -l -b"
    res = run_text(cmd,
                   expected_text=[
                       "local and base models",
                       "local-model",
                       "local-model.gguf",
                       "chatml",
                       "local-model-link",
                       "gpt-4"                       
                   ],
                   not_expected_text=[])


    cmd = "sibila models -m models/ -l local-model-link -b"
    res = run_text(cmd,
                   expected_text=[
                       "local and base models",
                       "local-model-link",
                       "local-model"
                   ],
                   not_expected_text=["local-model.gguf",
                                      "chatml",
                                      "gpt-4"])


    # Error: Cannot delete 'llamacpp:local-model', as entry 'llamacpp:local-model-link' links to it
    cmd = "sibila models -m models/ -d llamacpp:local-model"
    run_cmd(cmd, 
            expected_exit=1)


    cmd = "sibila models -m models/ -d llamacpp:local-model-link"
    run_cmd(cmd,
            expected_exit=0)

    cmd = "sibila models -m models/ -d llamacpp:local-model"
    run_json(cmd, 
             path=models_path,
             expected_json={})




def test_formats(env):

    formats_path = os.path.join(env[1], Models.FORMATS_CONF_FILENAME)

    cmd = """sibila formats -m models -s test-format "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant \n' }}{% endif %}" """
    run_json(cmd, 
             path=formats_path,
             expected_json={
                    "test-format": {
                        "template": "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant \n' }}{% endif %}"
                    }
                })


    cmd = """sibila formats -m models -s test-format-chatml chatml"""
    run_json(cmd, 
             path=formats_path,
             expected_json={
                    "test-format": {
                        "template": "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant \n' }}{% endif %}"
                    },
                    "test-format-chatml": {
                        "template": "chatml"
                    }
                })
    

    cmd = """sibila formats -m models -d test-format-chatml"""
    run_cmd(cmd,
            expected_exit=0)

    cmd = "sibila formats -m models -sl test-format-link test-format"
    run_json(cmd, 
             path=formats_path,
             expected_json={
                    "test-format": {
                        "template": "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant \n' }}{% endif %}"
                    },
                    "test-format-link": "test-format"
                })

    cmd = "sibila formats -m models -sl chatml-format-link chatml"
    run_json(cmd, 
             path=formats_path,
             expected_json={
                    "test-format": {
                        "template": "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant \n' }}{% endif %}"
                    },
                    "test-format-link": "test-format",
                    "chatml-format-link": "chatml"
                })


    cmd = "sibila formats -m models/ -l"
    res = run_text(cmd,
                   expected_text=[
                       "only local formats",
                       "test-format",
                       "{{",
                       "test-format-link",
                       "test-format",
                       "chatml-format-link"],
                   not_expected_text=["openchat"])


    cmd = "sibila formats -m models/ -l -b"
    res = run_text(cmd,
                   expected_text=[
                       "local and base formats",
                       "test-format",
                       "{{",
                       "test-format-link",
                       "test-format",
                       "chatml-format-link",
                       "openchat"],
                   not_expected_text=[])


    cmd = "sibila formats -m models/ -d chatml-format-link"
    run_cmd(cmd, 
            expected_exit=0)


    # Error: Cannot delete 'test-format', as entry 'test-format-link' links to it
    cmd = "sibila formats -m models/ -d test-format"
    run_cmd(cmd, 
            expected_exit=1)


    cmd = "sibila formats -m models/ -d test-format-link"
    run_cmd(cmd,
            expected_exit=0)

    cmd = "sibila formats -m models/ -d test-format"
    run_json(cmd, 
             path=formats_path,
             expected_json={})




def test_hub():

    cmd = "sibila hub -l openchat"
    res = run_text(cmd,
                   expected_text=[
                       "TheBloke/openchat-3.5-1210-GGUF",
                       "openchat-3.5-1210.Q4_K_M.gguf",
                       "openchat-3.5-1210.Q8_0.gguf",
                       "TheBloke/Seraph-openchat-3.5-1210-Slerp-GGUF",
                       "tsunemoto/openchat-3.5-1210-GGUF"],
                   not_expected_text=[])

    cmd = "sibila hub -l openchat -a TheBloke"
    res = run_text(cmd,
                   expected_text=[
                       "TheBloke/openchat-3.5-1210-GGUF",
                       "openchat-3.5-1210.Q4_K_M.gguf",
                       "openchat-3.5-1210.Q8_0.gguf",
                       "TheBloke/Seraph-openchat-3.5-1210-Slerp-GGUF"],
                   not_expected_text=["tsunemoto/openchat-3.5-1210-GGUF",
                                      "MaziyarPanahi/openchat"])
    
    cmd = "sibila hub -l openchat -a TheBloke -f q4_k_m"
    res = run_text(cmd,
                   expected_text=[
                       "TheBloke/openchat-3.5-1210-GGUF",
                       "openchat-3.5-1210.Q4_K_M.gguf",
                       "TheBloke/Seraph-openchat-3.5-1210-Slerp-GGUF"],
                   not_expected_text=["tsunemoto/openchat-3.5-1210-GGUF",
                                      "MaziyarPanahi/openchat",
                                      "openchat-3.5-1210.Q8_0.gguf"])

    cmd = """sibila hub -l "TheBloke/openchat 3.5 1210 GGUF" -a TheBloke -f q4_k_m"""
    res = run_text(cmd,
                   expected_text=[
                       "TheBloke/openchat-3.5-1210-GGUF",
                       "openchat-3.5-1210.Q4_K_M.gguf",
                       "TheBloke/Seraph-openchat-3.5-1210-Slerp-GGUF"],
                   not_expected_text=["tsunemoto/openchat-3.5-1210-GGUF",
                                      "MaziyarPanahi/openchat",
                                      "openchat-3.5-1210.Q8_0.gguf"])


    cmd = """sibila hub -l "TheBloke/openchat 3.5 1210 GGUF" -a TheBloke -f openchat-3.5-1210.Q4_K_M.gguf"""
    res = run_text(cmd,
                   expected_text=[
                       "TheBloke/openchat-3.5-1210-GGUF",
                       "openchat-3.5-1210.Q4_K_M.gguf"
                       ],
                   not_expected_text=["tsunemoto/openchat-3.5-1210-GGUF",
                                      "TheBloke/Seraph-openchat-3.5-1210-Slerp-GGUF",
                                      "MaziyarPanahi/openchat",
                                      "openchat-3.5-1210.Q8_0.gguf"])
   
    cmd = "sibila hub -l openchat -a TheBloke -f openchat-3.5-1210.Q4_K_M.gguf"
    res = run_text(cmd,
                   expected_text=[
                       "TheBloke/openchat-3.5-1210-GGUF",
                       "openchat-3.5-1210.Q4_K_M.gguf"
                       ],
                   not_expected_text=["tsunemoto/openchat-3.5-1210-GGUF",
                                      "TheBloke/Seraph-openchat-3.5-1210-Slerp-GGUF",
                                      "MaziyarPanahi/openchat",
                                      "openchat-3.5-1210.Q8_0.gguf"])



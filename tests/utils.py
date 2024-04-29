import pytest

import os, subprocess, shutil, json
from typing import Any, Optional, Union, Literal, Annotated

import logging
logging.basicConfig(level=logging.DEBUG)




# ============================================================== fixture setup/teardown utils

def setup_env_models(folder_name: str,
                     change_cwd: bool,
                     full_clean: bool) -> tuple[str,str]:
    
    """Create an environment with a named folder and 'models/' inside"""

    base_dir = os.path.dirname(__file__)

    base_dir = os.path.join(base_dir, folder_name)

    if full_clean and os.path.isdir(base_dir):
        assert base_dir.endswith(folder_name) # the paranoid!
        shutil.rmtree(base_dir)

    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)    

    models_dir = os.path.join(base_dir, "models")
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)

    if change_cwd:
        os.chdir(base_dir)

    return base_dir, models_dir



def teardown_env_models(base_dir: str):

    print("teardown")

    # as we're deleting base_dir, change CWD to a folder below it, 
    # or all sorts of errors will happen with os.path.*() calls
    os.chdir(os.path.join(base_dir, ".."))

    if os.path.isdir(base_dir):
        shutil.rmtree(base_dir)
  




def setup_model(model_filename: str,
                models_dir: str,
                always_copy: bool) -> str:

    base_dir = os.path.dirname(__file__)

    res_dir = os.path.join(base_dir, "res")
    if not os.path.isdir(res_dir):
        assert False, f"Resource dir missing: '{res_dir}'"

    src_model_path = os.path.join(res_dir, model_filename)
    if not os.path.isfile(src_model_path):
        assert False, f"Model missing at: '{src_model_path}'"
    

    # copy model
    dest_model_path = os.path.join(models_dir, model_filename)
    if always_copy or not os.path.isfile(dest_model_path):
        # print(f"Copying model to '{dest_model_path}'")
        shutil.copyfile(src_model_path, dest_model_path)

    return dest_model_path


def teardown_model(model_filename: str):

    if os.path.isfile(model_filename):
        os.remove(model_filename)










# ============================================================== shell command run and test utils
def run_cmd(cmd: str, 
            expected_exit: Optional[int] = None) -> tuple:
    res = subprocess.run(cmd,
                         shell=True,
                         capture_output=True,
                         encoding="utf-8")
    exit_code = res.returncode
    if expected_exit is not None:
        assert exit_code == expected_exit
    
    return exit_code, res.stdout, res.stderr


def run_text(cmd: str,
             *,
             path: Optional[str] = None,
             expected_stripped_text: Optional[str] = None,
             expected_text: Optional[list[str]] = None, 
             not_expected_text: Optional[list[str]] = None,
             expected_exit: int = 0) -> str:
    
    res = run_cmd(cmd, expected_exit=expected_exit)
    
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            out_text = f.read()
    else:
        out_text = "[stdout]\n" + res[1] + "[stderr]\n" + res[2]

    if expected_stripped_text is not None:
        assert out_text.strip() == expected_stripped_text

    if expected_text is not None:
        for text in expected_text:
            assert text in out_text

    if not_expected_text is not None:
        for text in not_expected_text:
            assert text not in out_text

    return out_text


def run_json(cmd: str, 
             *,
             path: Optional[str] = None,
             expected_json: Optional[dict] = None,
             expected_exit: int = 0) -> dict:

    text = run_text(cmd, 
                    path=path,
                    expected_exit=expected_exit)
    
    res = json.loads(text)

    if expected_json is not None:
        assert res == expected_json

    return res



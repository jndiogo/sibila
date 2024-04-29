import pytest

import os, json

import logging
# logging.basicConfig(level=logging.DEBUG)
# pytest --log-cli-level=DEBUG


from sibila import (
     LlamaCppModel
)


"""
clear; pytest -s test_llamacpp_models.py --models_dir ../../models/
"""

LIMIT = 9999

@pytest.fixture(scope="module")
def models_list(pytestconfig):

    models_dir = pytestconfig.getoption("models_dir")
    if not models_dir:
        models_dir = "res"

    if not os.path.isabs(models_dir):
        base_dir = os.path.dirname(__file__)
        models_dir = os.path.normpath(os.path.join(base_dir, models_dir))

    print("models_dir:", models_dir)

    filenames = []

    if os.path.isdir(models_dir):
        file_list = os.listdir(models_dir)
        # print(file_list)

        for filename in file_list:
            if not filename.endswith(".gguf"):
                continue
            if filename.startswith("--"):
                continue

            path = os.path.join(models_dir, filename)
            if not os.path.isfile(path):
                continue
            filenames.append(path)

        filenames = sorted(filenames[:LIMIT])

    if not filenames:
        print("No models found")
    else:
        print("Found models:", "\n".join(filenames), sep="\n")
        
    return filenames


def test_extract(models_list):
    # print(models_list)

    IN = "The 7 seas"
    INST = "Extract the number from the text"
    OUT = 7

    for path in models_list:

        print(os.path.basename(path), end=': ')
        model = LlamaCppModel(path)

        out = model.extract(int, IN, inst=INST)
        print(out)

        del model

        # assert out == OUT
        


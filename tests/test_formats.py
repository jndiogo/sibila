import pytest

import os, json

from typing import Any, Optional, Union, Literal, Annotated, get_origin, get_args

import logging
logging.basicConfig(level=logging.DEBUG)

import jinja2
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment
version = jinja2.__version__.split('.')
if int(version[0]) < 3:
    raise ImportError

def jinja_raise_exception(message):
    raise TemplateError(message)            


from sibila import Thread



def test_jinja_template_syntax():

    base_dir = os.path.dirname(__file__)
    formats_path = os.path.normpath(os.path.join(base_dir, "..", "sibila", "res", "base_formats.json"))
    print (formats_path)

    with open(formats_path, "r", encoding="utf-8") as f:
        formats_dir = json.load(f)


    th = Thread.make_INST_IN("You are a helpful, respectful and honest assistant.",
                             "What is the color of the sky?")
    th.add_OUT("The color of the sky is blue.")
    th.add_IN("Are you sure?")
    th.add_OUT("Oh yes.")
    th.add_IN("Indeed it is so blue.")
    messages = th.as_chatml()

    special_tokens_map = {"bos_token": "<<S>>",
                          "eos_token": "<</S>>",
                          "pad_token": "<<PAD>>",
                          "unk_token": "<<UNK>>"}

    for key,val in formats_dir.items():
        if isinstance(val, str): # alias
            continue

        format_template = val["template"]
        if not "{{" in format_template: # template def link
            continue

        print(f"= {key} ==================")

        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        jinja_env.globals["raise_exception"] = jinja_raise_exception
        jinja_compiled_template = jinja_env.from_string(format_template)

        text = jinja_compiled_template.render(messages=messages, # type: ignore[union-attr]
                                              add_generation_prompt=True,
                                              **special_tokens_map)
        print(text)

        assert ("You are a helpful, respectful and honest assistant." in text and 
                "What is the color of the sky?" in text and
                "The color of the sky is blue." in text and
                "Are you sure?" in text and
                "Oh yes." in text)


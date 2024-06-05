"""
Requires a defined env variable TOGETHER_API_KEY with a valid API key.
See:
https://docs.together.ai/docs/inference-models
"""

import pytest

import os, subprocess, shutil, json, asyncio, time
from typing import Any, Optional, Union, Literal, Annotated, get_origin, get_args
from itertools import product

import logging
logger = logging.getLogger(__name__)

from dotenv import load_dotenv


from sibila import (
    Model,
    LlamaCppModel,
    GenConf
)

from .utils import setup_env_models, teardown_env_models, setup_model, teardown_model
from .utils import run_cmd, run_text, run_json





models_dir = None

@pytest.fixture(autouse=True, scope="module")
def get_models_dir(pytestconfig):    
    global models_dir

    models_dir = pytestconfig.getoption("models_dir")
    if not models_dir:
        models_dir = "../../models"

    if not os.path.isabs(models_dir):
        base_dir = os.path.dirname(__file__)
        models_dir = os.path.normpath(os.path.join(base_dir, models_dir))



def create_model(filename: str) -> Model:

    path = os.path.join(models_dir, filename)
    model = LlamaCppModel(path, 
                          ctx_len=IN_CTX_LEN)
    return model





IN_CTX_LEN = 2048

models = [
    "dolphin-2_6-phi-2.Q4_K_M.gguf",
    "Hermes-2-Pro-Mistral-7B.Q4_K_M.gguf",
    "llama-2-7b-chat.Q4_K_M.gguf",
    "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
    "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    "openchat-3.5-0106.Q4_K_M.gguf",
    "openchat-3.5-1210.Q4_K_M.gguf",
    "Phi-3-mini-4k-instruct-q4.gguf",
    "starling-lm-7b-alpha.Q5_K_M.gguf",
]


__models = [
    "dolphin-2_6-phi-2.Q4_K_M.gguf",
]


"""
Models that don't pass these tests:
    "gemma-2b-it-q4_k_m.gguf",
    "gemma-2b-it-q8_0.gguf",
    "kunoichi-dpo-v2-7b.Q4_K_M.gguf",
    "phi-2.Q5_K_M.gguf",
    "qwen1_5-0_5b-chat-q4_k_m.gguf",
    "rocket-3b.Q4_K_M.gguf",
    "stablelm-2-12b-chat-Q4_K_M.gguf",
    "stablelm-2-zephyr-1_6b-Q4_K_M.gguf",
    "stablelm-zephyr-3b.Q4_K_M.gguf",
    "Starling-LM-7B-beta-Q4_K_M.gguf",
    "TinyDolphin-2.8-1.1b.Q4_K_M.gguf",
    "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    "zephyr-7b-beta.Q4_K_M.gguf",
    "zephyr-7b-gemma-v0.1-Q4_K_M.gguf",
---
"""







@pytest.mark.parametrize("filename", models)
def test_all_for_model(filename):
    print("\n########### test_all_for_model", filename)

    model = create_model(filename)


    run_hello(model)

    run_keypoints(model)

    run_tag_customer_query(model)

    
    del model











# ============================================================================== hello
def run_hello(model: Model):

    # the instructions or system command: speak like a pirate!
    inst_text = "You speak like a pirate."

    # the in prompt
    in_text = "Hello there?"
    print("User:", in_text)

    # query the model with instructions and input text
    text = model(in_text,
                 inst=inst_text)
    print("Model:", text)

    assert len(text)





# ============================================================================== keypoints

def run_keypoints(model: Model):

    doc = """\
Fiji, officially the Republic of Fiji,[n 2] is an island country in Melanesia,
part of Oceania in the South Pacific Ocean. It lies about 1,100 nautical miles 
(2,000 km; 1,300 mi) north-northeast of New Zealand. Fiji consists of 
an archipelago of more than 330 islands—of which about 110 are permanently 
inhabited—and more than 500 islets, amounting to a total land area of about 
18,300 square kilometres (7,100 sq mi). The most outlying island group is 
Ono-i-Lau. About 87% of the total population of 924,610 live on the two major 
islands, Viti Levu and Vanua Levu. About three-quarters of Fijians live on 
Viti Levu's coasts, either in the capital city of Suva, or in smaller 
urban centres such as Nadi (where tourism is the major local industry) or 
Lautoka (where the sugar-cane industry is dominant). The interior of Viti Levu 
is sparsely inhabited because of its terrain.[13]

The majority of Fiji's islands were formed by volcanic activity starting around 
150 million years ago. Some geothermal activity still occurs today on the islands 
of Vanua Levu and Taveuni.[14] The geothermal systems on Viti Levu are 
non-volcanic in origin and have low-temperature surface discharges (of between 
roughly 35 and 60 degrees Celsius (95 and 140 °F)).

Humans have lived in Fiji since the second millennium BC—first Austronesians and 
later Melanesians, with some Polynesian influences. Europeans first visited Fiji 
in the 17th century.[15] In 1874, after a brief period in which Fiji was an 
independent kingdom, the British established the Colony of Fiji. Fiji operated as 
a Crown colony until 1970, when it gained independence and became known as 
the Dominion of Fiji. In 1987, following a series of coups d'état, the military 
government that had taken power declared it a republic. In a 2006 coup, Commodore 
Frank Bainimarama seized power. In 2009, the Fijian High Court ruled that the 
military leadership was unlawful. At that point, President Ratu Josefa Iloilo, 
whom the military had retained as the nominal head of state, formally abrogated 
the 1997 Constitution and re-appointed Bainimarama as interim prime minister. 
Later in 2009, Ratu Epeli Nailatikau succeeded Iloilo as president.[16] On 17 
September 2014, after years of delays, a democratic election took place. 
Bainimarama's FijiFirst party won 59.2% of the vote, and international observers 
deemed the election credible.[17] 
"""

    # model instructions text, also known as system message
    inst_text = "Be helpful and provide concise answers."

    in_text = "Extract 5 keypoints of the following text:\n" + doc

    from pydantic import BaseModel

    # this class definition will be used to constrain the model output and initialize an instance object
    class Keypoints(BaseModel):
        keypoint_list: list[str]

    out = model.pydantic(Keypoints,
                         in_text,
                         inst = inst_text,)

    for kpoint in out.keypoint_list:
        print("-", kpoint)    

    assert len(out.keypoint_list) == 5




# ============================================================================== tag customer query

def run_tag_customer_query(model: Model):

    queries = """\
1. Do you offer a trial period for your software before purchasing?
2. I'm experiencing a glitch with your app, it keeps freezing after the latest update.
3. What are the different pricing plans available for your subscription service?
4. Can you provide instructions on how to reset my account password?
5. I'm unsure about the compatibility of your product with my device, can you advise?
6. How can I track my recent order and estimate its delivery date?
7. Is there a customer loyalty program or rewards system for frequent buyers?
8. I'm interested in your online courses, but do you offer refunds if I'm not satisfied?
9. Could you clarify the coverage and limitations of your product warranty?
10. What are your customer support hours and how can I reach your team in case of emergencies?
"""


    from enum import Enum
    from dataclasses import dataclass

    class Tag(str, Enum):
        """Queries can be classified into the following tags:
tech_support: queries related with technical problems.
billing: post-sale queries about billing cycle, or subscription termination.
account: queries about user account problems.
pre_sales: queries from prospective customers (who have not yet purchased).
other: all other query topics."""        
        TECH_SUPPORT = "tech_support"
        BILLING = "billing"
        PRE_SALES = "pre_sales"
        ACCOUNT = "account"
        OTHER = "other"

    @dataclass        
    class Query():
        id: int
        query_summary: str
        query_tag: Tag

    # model instructions text, also known as system message
    inst_text = "Extract information from customer queries."

    # the input query, including the above text
    in_text = "Each line is a customer query. Extract information about each query:\n\n" + queries

    out = model.extract(list[Query],
                        in_text,
                        inst=inst_text)

    for query in out:
        print(query)    

    assert len(out) == 10


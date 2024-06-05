"""
Exact results for local models depend on hardware acceleration, with CUDA being the worst for determinism.
For reproducibility, use llama-cpp-python installed without any acceleration.
See:
https://jndiogo.github.io/sibila/tips/#deterministic-outputs

"""



import pytest

import os, subprocess, shutil, json
from typing import Any, Optional, Union, Literal, Annotated, get_origin, get_args

import logging
logging.basicConfig(level=logging.DEBUG)

from sibila import (
    Models,
    LlamaCppModel,
    GenConf,
    GenError
)

from .utils import setup_env_models, teardown_env_models, setup_model, teardown_model
from .utils import run_cmd, run_text, run_json





# https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
MODEL_FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

DO_TEARDOWN = True

@pytest.fixture(autouse=True, scope="module")
def env_model():

    base_dir, models_dir = setup_env_models("examples_tinyllama", 
                                            change_cwd=True,
                                            full_clean=False)

    model_path = setup_model(MODEL_FILENAME, 
                             models_dir,
                             always_copy=False)

    rel_model_path = os.path.join("models", MODEL_FILENAME)

    model = LlamaCppModel(rel_model_path, 
                          format="zephyr",
                          seed=21)

    ret = base_dir, model_path, model
    print("---> setup", ret)
    
    yield ret

    # --------------------------- teardown

    del model

    if DO_TEARDOWN == False:
        return

    print("---> teardown", ret)

    teardown_model(model_path)

    teardown_env_models(base_dir)



RUN_PRINT = False
RUN_ASSERT = True

def assert_same(out: Any,
                expected: Any):
    
    if RUN_PRINT:
        print(repr(out))
    if RUN_ASSERT:
        assert out == expected












def test_hello_model(env_model):

    model = env_model[2]

    # the instructions or system command: speak like a pirate!
    inst_text = "You speak like a pirate."

    # the in prompt
    in_text = "Hello there?"

    # query the model with instructions and input text
    text = model(in_text,
                 inst=inst_text)


    start = "Yes, I'm here! How may I assist you today?"
    # assert text.startswith(start)





def test_from_text_to_object(env_model):

    model = env_model[2]

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

    from pydantic import BaseModel, Field

    # this class definition will be used to constrain the model output and initialize an instance object
    class Keypoints(BaseModel):
        keypoint_list: list[str]

    # with pytest.raises(GenError):
    out = model.pydantic(Keypoints,
                         in_text,
                         inst = inst_text)
    
    assert len(out.keypoint_list)









def test_extract_pydantic(env_model):

    model = env_model[2]

    text = """\
It was a breezy afternoon in a bustling café nestled in the heart of a vibrant city. Five strangers found themselves drawn together by the aromatic allure of freshly brewed coffee and the promise of engaging conversation.

Seated at a corner table was Lucy Bennett, a 28-year-old journalist from London, her pen poised to capture the essence of the world around her. Her eyes sparkled with curiosity, mirroring the dynamic energy of her beloved city.

Opposite Lucy sat Carlos Ramirez, a 35-year-old architect from the sun-kissed streets of Barcelona. With a sketchbook in hand, he exuded creativity, his passion for design evident in the thoughtful lines that adorned his face.

Next to them, lost in the melodies of her guitar, was Mia Chang, a 23-year-old musician from the bustling streets of Tokyo. Her fingers danced across the strings, weaving stories of love and longing, echoing the rhythm of her vibrant city.

Joining the trio was Ahmed Khan, a married 40-year-old engineer from the bustling metropolis of Mumbai. With a laptop at his side, he navigated the complexities of technology with ease, his intellect shining through the chaos of urban life.

Last but not least, leaning against the counter with an air of quiet confidence, was Isabella Santos, a 32-year-old fashion designer from the romantic streets of Paris. Her impeccable style and effortless grace reflected the timeless elegance of her beloved city.
"""


    from pydantic import BaseModel, Field

    class Person(BaseModel):
        first_name: str
        last_name: str
        age: int
        occupation: str
        details_about_person: str
        source_location: str
        source_country: str
        is_married: bool

    # model instructions text, also known as system message
    inst_text = "Extract information."

    in_text = "Extract person information from the following text:\n\n" + text

    out = model.extract(list[Person],
                        in_text,
                        inst=inst_text)

    """ llama-cpp-python 0.2.56:
    expected = [Person(first_name='Lucy', last_name='Bennett', age=28, occupation='Journalist', details_about_person='Born and raised in London, Lucy is passionate about exploring new cultures and cuisines. She loves to travel and has visited over 20 countries so far.', source_location='London', source_country='United Kingdom', is_married=True),
                Person(first_name='Carlos', last_name='Ramirez', age=35, occupation='Architect', details_about_person='Carlos is an architect who loves to design spaces that inspire creativity and innovation. He has worked on several projects in Barcelona, including a beautifully designed restaurant.', source_location='Barcelona', source_country='Spain', is_married=False), 
                Person(first_name='Mia', last_name='Chang', age=23, occupation='Musician', details_about_person='Mia is a talented musician who loves to experiment with different genres and styles. She has performed at many local venues and festivals, showcasing her unique sound.', source_location='Mumbai', source_country='India', is_married=True), 
                Person(first_name='Ahmed', last_name='Khan', age=40, occupation='Engineer', details_about_person='Ahmed is an engineer who loves to solve complex problems and design innovative solutions. He has worked on several projects for companies like Mumbai-based Muhimbai Technologies and Muhimbai Technologies (Mumbai).', source_location='Mumbai', source_country='India', is_married=False), 
                Person(first_name='Isabella', last_name='Santos', age=32, occupation='Fashion designer', details_about_person='Isabella is a fashion designer who loves to experiment with different fabrics and materials. She has designed collections for several fashion brands and has won several awards for her creative designs.', source_location='Paris', source_country='France', is_married=True)]
    assert_same(out, expected)
    """

    assert len(out)







def test_extract_dataclass(env_model):

    model = env_model[2]

    text = """\
It was a breezy afternoon in a bustling café nestled in the heart of a vibrant city. Five strangers found themselves drawn together by the aromatic allure of freshly brewed coffee and the promise of engaging conversation.

Seated at a corner table was Lucy Bennett, a 28-year-old journalist from London, her pen poised to capture the essence of the world around her. Her eyes sparkled with curiosity, mirroring the dynamic energy of her beloved city.

Opposite Lucy sat Carlos Ramirez, a 35-year-old architect from the sun-kissed streets of Barcelona. With a sketchbook in hand, he exuded creativity, his passion for design evident in the thoughtful lines that adorned his face.

Next to them, lost in the melodies of her guitar, was Mia Chang, a 23-year-old musician from the bustling streets of Tokyo. Her fingers danced across the strings, weaving stories of love and longing, echoing the rhythm of her vibrant city.

Joining the trio was Ahmed Khan, a married 40-year-old engineer from the bustling metropolis of Mumbai. With a laptop at his side, he navigated the complexities of technology with ease, his intellect shining through the chaos of urban life.

Last but not least, leaning against the counter with an air of quiet confidence, was Isabella Santos, a 32-year-old fashion designer from the romantic streets of Paris. Her impeccable style and effortless grace reflected the timeless elegance of her beloved city.
"""

    from dataclasses import dataclass

    @dataclass
    class Person:
        first_name: str
        last_name: str
        age: int
        occupation: str
        details_about_person: str
        source_location: str
        source_country: str
        is_married: bool

    # model instructions text, also known as system message
    inst_text = "Extract information."

    # the input query, including the above text
    in_text = "Extract person information from the following text:\n\n" + text

    out = model.extract(list[Person],
                        in_text,
                        inst=inst_text)


    """ llama-cpp-python 0.2.56:
    expected = [Person(first_name='Lucy', last_name='Bennett', age=28, occupation='Journalist', details_about_person='Born and raised in London, Lucy is passionate about exploring the world around her through her writing. She loves to travel and has visited over 20 countries so far.', source_location='London', source_country='United Kingdom', is_married=True), Person(first_name='Carlos', last_name='Ramirez', age=35, occupation='Architect', details_about_person='Carlos is a passionate designer who loves to create beautiful spaces for people to live and work in. He is also an avid musician, playing guitar and singing in his free time.', source_location='Barcelona', source_country='Spain', is_married=False), Person(first_name='Mia', last_name='Chang', age=23, occupation='Musician', details_about_person='Mia is a talented musician who loves to explore the complexities of technology through her music. She also enjoys spending time with her friends and family.', source_location='Mumbai', source_country='India', is_married=True), Person(first_name='Ahmed', last_name='Khan', age=40, occupation='Engineer', details_about_person='Ahmed is an experienced engineer who loves to explore new technologies and their potential applications. He is also an avid reader and enjoys spending time with his family.', source_location='Mumbai', source_country='India', is_married=False), Person(first_name='Isabella', last_name='Santos', age=32, occupation='Fashion designer', details_about_person='Isabella is a talented fashion designer who loves to experiment with different styles and techniques. She also enjoys spending time with her friends and family.', source_location='Paris', source_country='France', is_married=True)]
    assert_same(out, expected)
    """

    assert len(out)











def test_tag(env_model):

    model = env_model[2]

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

    """ llama-cpp-python 0.2.56:
    expected = [Query(id=1, query_summary='Do you offer a trial period for your software before purchasing?', query_tag='tech_support'), Query(id=2, query_summary="I'm experiencing a glitch with your app, it keeps freezing after the latest update.", query_tag='billing'), Query(id=3, query_summary='What are the different prizing plans available for your subscription service?', query_tag='pre_sales'), Query(id=4, query_summary='Can you provide instructions on how to reset my account password?', query_tag='tech_support'), Query(id=5, query_summary="I'm unsure about the compatibility of your product with my device, can you advise?", query_tag='billing'), Query(id=6, query_summary='How can I track my recent order and estimate its delivery date?', query_tag='tech_support'), Query(id=7, query_summary='Is there a customer loyalty program or reward system for frequent buyers?', query_tag='tech_support'), Query(id=8, query_summary="I'm interested in your online courses, but do you offer refunds if I'm not satisfied?", query_tag='tech_support'), Query(id=9, query_summary='Could you clarify the coverage and limitations of your product warranty?', query_tag='tech_support'), Query(id=10, query_summary='What are your customer support hours and how can I reach your team in case of emergencies?', query_tag='tech_support')]
    assert_same(out, expected)
    """
    
    assert len(out)









def test_quick_meeting(env_model):

    model = env_model[2]

    transcript = """\
Date: 10th April 2024
Time: 10:30 AM
Location: Conference Room A

Attendees:
    Arthur: Logistics Supervisor
    Bianca: Operations Manager
    Chris: Fleet Coordinator

Arthur: Good morning, team. Thanks for making it. We've got three matters to address quickly today.

Bianca: Morning, Arthur. Let's dive in.

Chris: Ready when you are.

Arthur: First off, we've been having complaints about late deliveries. This is very important, we're getting some bad reputation out there.

Bianca: Chris, I think you're the right person to take care of this. Can you investigate and report back by end of day? 

Chris: Absolutely, Bianca. I'll look into the reasons and propose solutions.

Arthur: Great. Second, Bianca, we need to update our driver training manual. Can you take the lead and have a draft by Friday?

Bianca: Sure thing, Arthur. I'll get started on that right away.

Arthur: Lastly, we need to schedule a meeting with our software vendor to discuss updates to our tracking system. This is a low-priority task but still important. I'll handle that. Any input on timing?

Bianca: How about next Wednesday afternoon?

Chris: Works for me.

Arthur: Sounds good. I'll arrange it. Thanks, Bianca, Chris. Let's keep the momentum going.

Bianca: Absolutely, Arthur.

Chris: Will do.
"""


    from pydantic import BaseModel, Field
    from enum import Enum

    class Attendee(BaseModel):
        name: str
        occupation: str

    class Priority(str, Enum):
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"
        
    class ActionItem(BaseModel):
        index: int = Field(description="Sequential index for the action item")
        name: str = Field(description="Action item name")
        priority: Priority = Field(description="Action item priority")
        due_by: str = Field(description="When should the item be complete")
        assigned_attendee: str = Field(description="Name of the attendee to which action item was assigned")

    class Meeting(BaseModel):
        meeting_date: str
        meeting_location: str
        attendees: list[Attendee]
        action_items: list[ActionItem]


    # model instructions text, also known as system message
    inst_text = "Extract information."

    # the input query, including the above transcript
    in_text = "Extract information from this meeting transcript:\n\n" + transcript

    out = model.extract(Meeting,
                        in_text,
                        inst=inst_text)

    """ llama-cpp-python 0.2.56:
    expected = Meeting(meeting_date='10/10/2024', 
                       meeting_location='Confrenace Room A', 
                       attendees=[Attendee(name='Arthur', occupation='Logistics Supervisor'), 
                                  Attendee(name='Bianca', occupation='Operation Manager'), 
                                  Attendee(name='Chris', occupation='Fleet Coordinator')], 
                                  action_items=[ActionItem(index=1, name='Late deliveries', priority=Priority.HIGH, due_by='End of day', assigned_attendee='Bianca'),
                                                ActionItem(index=2, name='Driver training manual', priority=Priority.MEDIUM, due_by='Friday', assigned_attendee='None'),
                                                ActionItem(index=3, name='Software vendor update', priority=Priority.LOW, due_by='Next Wednesday', assigned_attendee='None')])
    assert_same(out, expected)
    """

    assert "2024" in out.meeting_date



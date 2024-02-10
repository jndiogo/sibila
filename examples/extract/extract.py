# load env variables from a .env if available:
env_path = "../../.env"
import os
if os.path.isfile(env_path):
    from dotenv import load_dotenv
    assert load_dotenv(env_path, override=True, verbose=True)


from sibila import ModelDir

from pydantic import BaseModel, Field
from typing import List


# delete any previous model
try: del model
except: ...

# to use a local model, assuming it's in ../../models/:
# add models folder config which also adds to ModelDir path
ModelDir.add("../../models/modeldir.json")
# set the model's filename - change to your own model
name = "llamacpp:openchat-3.5-1210.Q4_K_M.gguf"
model = ModelDir.create(name)

# to use an OpenAI model:
# model = ModelDir.create("openai:gpt-4")

text = """\
It was a breezy afternoon in a bustling café nestled in the heart of a vibrant city. Five strangers found themselves drawn together by the aromatic allure of freshly brewed coffee and the promise of engaging conversation.

Seated at a corner table was Lucy Bennett, a 28-year-old journalist from London, her pen poised to capture the essence of the world around her. Her eyes sparkled with curiosity, mirroring the dynamic energy of her beloved city.

Opposite Lucy sat Carlos Ramirez, a 35-year-old architect from the sun-kissed streets of Barcelona. With a sketchbook in hand, he exuded creativity, his passion for design evident in the thoughtful lines that adorned his face.

Next to them, lost in the melodies of her guitar, was Mia Chang, a 23-year-old musician from the bustling streets of Tokyo. Her fingers danced across the strings, weaving stories of love and longing, echoing the rhythm of her vibrant city.

Joining the trio was Ahmed Khan, a married 40-year-old engineer from the bustling metropolis of Mumbai. With a laptop at his side, he navigated the complexities of technology with ease, his intellect shining through the chaos of urban life.

Last but not least, leaning against the counter with an air of quiet confidence, was Isabella Santos, a 32-year-old fashion designer from the romantic streets of Paris. Her impeccable style and effortless grace reflected the timeless elegance of her beloved city.
"""

# model instructions text, also known as system message
inst_text = "Extract information."

in_text = "Extract person information from the following text:\n\n" + text


class Person(BaseModel):
    first_name: str
    last_name: str
    age: int
    occupation: str
    details_about_person: str
    source_location: str
    source_country: str
    is_married: bool

class Info(BaseModel):
    person_list: List[Person]
    
out = model.query_pydantic(Info,
                           inst_text,
                           in_text)

for person in out.person_list:
    print(person)

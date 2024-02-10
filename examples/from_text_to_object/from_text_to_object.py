# load env variables from a .env if available:
env_path = "../../.env"
import os
if os.path.isfile(env_path):
    from dotenv import load_dotenv
    assert load_dotenv(env_path, override=True, verbose=True)


from pydantic import BaseModel, Field
from typing import List

from sibila import ModelDir


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


# this class definition will be used to constrain the model output and initialize an instance object
class Keypoints(BaseModel):
    keypoint_list: List[str]

out = model.query_pydantic(Keypoints,
                           inst_text,
                           in_text)

for kpoint in out.keypoint_list:
    print("-", kpoint)

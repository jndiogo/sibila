In this example we'll ask the model to extract keypoints from a text:
- First in plain text format
- Then free JSON output (with fields selected by the model)
- Later constrained by a JSON schema (so that we can specify which fields)
- And finally by generating to a Pydantic object (from a class definition)

All the queries will be made at temperature=0, which is the default GenConf setting.
This means that the model is giving it's best (as in most probable) answer and that it will always output the same results, given the same inputs.

Also available as a Jupyter notebook or a Python script in the example's folder.

We'll start by creating either a local model or a GPT-4 model.

To use a local model, make sure you have its file in the folder "../../models". You can use any GGUF format model - [see here how to download the OpenChat model used below](https://jndiogo.github.io/sibila/models/local_model/#examples). If you use a different one, don't forget to set its filename in the name variable below, after the text "llamacpp:".

To use an OpenAI model, make sure you defined the env variable OPENAI_API_KEY with a valid token and uncomment the line after "# to use an OpenAI model:".
For an OpenAI model, make sure you defined the env variable OPENAI_API_KEY with a valid token and uncomment the line after "# to use an OpenAI model:".


```python
# load env variables like OPENAI_API_KEY from a .env file (if available)
try: from dotenv import load_dotenv; load_dotenv()
except: ...

from sibila import Models

# delete any previous model
try: del model
except: ...

# to use a local model, assuming it's in ../../models:
# setup models folder:
Models.setup("../../models")
# set the model's filename - change to your own model
model = Models.create("llamacpp:openchat-3.5-1210.Q4_K_M.gguf")

# to use an OpenAI model:
# model = Models.create("openai:gpt-4")
```

Let's use this fragment from Wikipedia's entry on the Fiji islands: https://en.wikipedia.org/wiki/


```python
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
```

Let's start with a free text query by calling model().


```python
in_text = "Extract 5 keypoints of the following text:\n" + doc

out = model(in_text, inst=inst_text)
print(out)
```

    1. Fiji is an island country located in Melanesia, part of Oceania in the South Pacific Ocean. It lies approximately 1,100 nautical miles north-northeast of New Zealand.
    2. The country consists of more than 330 islands with about 110 permanently inhabited islands and over 500 islets, totaling a land area of about 18,300 square kilometers.
    3. Approximately 87% of Fiji's total population of 924,610 live on the two major islands, Viti Levu and Vanua Levu, with a majority living on Viti Levu's coasts.
    4. The majority of Fiji's islands were formed by volcanic activity starting around 150 million years ago, with some geothermal activity still occurring on certain islands.
    5. Fiji has a complex history, transitioning from an independent kingdom to a British colony, then a Dominion, and finally a republic after a series of coups and constitutional changes. In 2014, a democratic election took place, marking a significant milestone in the country's political history.


These are quite reasonable keypoints!

Let's now ask for JSON output, taking care to explicitly request it in the query (in_text variable).

Instead of model() we now use json() which returns a Python dict.


```python
import pprint
pp = pprint.PrettyPrinter(width=300, sort_dicts=False)

in_text = "Extract 5 keypoints of the following text in JSON format:\n\n" + doc

out = model.json(in_text,
                 inst=inst_text)
pp.pprint(out)
```

    {'keypoints': [{'title': 'Location', 'description': 'Fiji is an island country in Melanesia, part of Oceania in the South Pacific Ocean.'},
                   {'title': 'Geography', 'description': 'Consists of more than 330 islands with about 110 permanently inhabited islands.'},
                   {'title': 'Population', 'description': 'Total population of 924,610 live on the two major islands, Viti Levu and Vanua Levu.'},
                   {'title': 'History', 'description': 'Humans have lived in Fiji since the second millennium BC with Austronesians, Melanesians, and Polynesian influences.'},
                   {'title': 'Political Status', 'description': 'Officially known as the Republic of Fiji, gained independence from British rule in 1970.'}]}


Note how the model chose to return different fields like "title" or "description".

Because we didn't specify which fields we want, each model will generate different ones.

To specify a fixed format, let's now generate by setting a JSON schema that defines which fields and types we want:


```python
json_schema = {
  "properties": {
    "keypoint_list": {
      "description": "Keypoint list",
      "items": {
        "type": "string",
        "description": "Keypoint"
      },
      "type": "array"
    }
  },
  "required": [
    "keypoint_list"
  ],
  "type": "object"
}
```

This JSON schema requests that the generated dict constains a "keypoint_list" with a list of strings.

We'll also use json(), now passing the json_schema:


```python
out = model.json(in_text,
                 inst=inst_text,
                 json_schema=json_schema)

print(out)
```

    {'keypoint_list': ['Fiji is an island country in Melanesia, part of Oceania in the South Pacific Ocean.', 'About 87% of the total population of 924,610 live on the two major islands, Viti Levu and Vanua Levu.', "The majority of Fiji's islands were formed by volcanic activity starting around 150 million years ago.", 'Humans have lived in Fiji since the second millennium BC—first Austronesians and later Melanesians, with some Polynesian influences.', "In 2014, a democratic election took place, with Bainimarama's FijiFirst party winning 59.2% of the vote."]}



```python
for kpoint in out["keypoint_list"]:
    print(kpoint)
```

    Fiji is an island country in Melanesia, part of Oceania in the South Pacific Ocean.
    About 87% of the total population of 924,610 live on the two major islands, Viti Levu and Vanua Levu.
    The majority of Fiji's islands were formed by volcanic activity starting around 150 million years ago.
    Humans have lived in Fiji since the second millennium BC—first Austronesians and later Melanesians, with some Polynesian influences.
    In 2014, a democratic election took place, with Bainimarama's FijiFirst party winning 59.2% of the vote.


It has generated a string list in the "keypoint_list" field, as we specified in the JSON schema.

This is better, but the problem with JSON schemas is that they can be quite hard to work with.

Let's use an easier way to specify the fields we want returned: Pydantic classes derived from BaseModel. This is way simpler to use than JSON schemas.


```python
from pydantic import BaseModel, Field

# this class definition will be used to constrain the model output and initialize an instance object
class Keypoints(BaseModel):
    keypoint_list: list[str]

out = model.pydantic(Keypoints,
                     in_text,
                     inst=inst_text)
print(out)
```

    keypoint_list=['Fiji is an island country in Melanesia, part of Oceania in the South Pacific Ocean.', 'About 87% of the total population of 924,610 live on the two major islands, Viti Levu and Vanua Levu.', "The majority of Fiji's islands were formed by volcanic activity starting around 150 million years ago.", 'Humans have lived in Fiji since the second millennium BC—first Austronesians and later Melanesians, with some Polynesian influences.', "In 2014, a democratic election took place, with Bainimarama's FijiFirst party winning 59.2% of the vote."]



```python
for kpoint in out.keypoint_list:
    print(kpoint)
```

    Fiji is an island country in Melanesia, part of Oceania in the South Pacific Ocean.
    About 87% of the total population of 924,610 live on the two major islands, Viti Levu and Vanua Levu.
    The majority of Fiji's islands were formed by volcanic activity starting around 150 million years ago.
    Humans have lived in Fiji since the second millennium BC—first Austronesians and later Melanesians, with some Polynesian influences.
    In 2014, a democratic election took place, with Bainimarama's FijiFirst party winning 59.2% of the vote.


The pydantic() method returns an object of class Keypoints, instantiated with the model output.

This is a much simpler way to extract structured data from model.

Please see other examples for more interesting objects. In particular, we did not add descriptions to the fields, which are important clues to help the model understand what we want.

Besides Pydantic classes, Sibila can also use Python's dataclass to extract structured data. This is a lighter and easier alternative to using Pydantic.
<!-- TODO: link to dataclass in concepts -->

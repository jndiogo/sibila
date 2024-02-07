In this example we'll ask the model to extract keypoints from a text:
- First in plain text format
- Then free JSON output (with fields selected by the model)
- Later constrained by a JSON schema (so that we can specify which fields)
- And finally by generating to a Pydantic object (from a class definition)

All the queries will be made at temperature=0, which is the default GenConf setting.
This means that the model is giving it's best (as in most probable) answer and that it will always output the same results, given the same inputs.

We'll start by creating either a local model or a GPT-4 model.

For a local model make sure you have its file in a folder like ../../models and have the right filename in the name variable.

For OpenAI models, make sure you defined the env variable OPENAI_API_KEY with a valid token.


```python
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

# this is the text with the model instructions, also known as system message.
inst_text = "Be helpful and provide concise answers."
```

Let's start with a free text query with query_gen().


```python
in_text = "Extract 5 keypoints of the following text:\n" + doc

out = model.query_gen(inst_text, in_text)
print(out)
```

    1. Fiji is an island country in Melanesia, part of Oceania in the South Pacific Ocean.
    2. It consists of more than 330 islands with a total land area of about 18,300 square kilometres (7,100 sq mi).
    3. About 87% of Fiji's population of 924,610 live on the two major islands, Viti Levu and Vanua Levu.
    4. The majority of Fiji's islands were formed by volcanic activity starting around 150 million years ago.
    5. Fiji gained independence from the British in 1970 and became a republic in 1987 after a series of coups d'état.


These are quite reasonable keypoints!

Let's now ask for JSON output, taking care to explicitly request it in the query (in_text variable).

Instead of query_gen() we now use query_json() which returns a Python dict. 


```python
import pprint
pp = pprint.PrettyPrinter(width=300, sort_dicts=False)

in_text = "Extract 5 keypoints of the following text in JSON format:\n\n" + doc

out = model.query_json(inst_text, in_text)
pp.pprint(out)
```

    {'keypoints': [{'Fiji_island_country': 'Fiji is an island country in Melanesia, part of Oceania in the South Pacific Ocean.'},
                   {'geography': 'It lies about 1,100 nautical miles north-northeast of New Zealand and consists of more than 330 islands with a total land area of about 18,300 square kilometres.'},
                   {'population_distribution': 'About 87% of the total population of 924,610 live on the two major islands, Viti Levu and Vanua Levu.'},
                   {'island_formation': "The majority of Fiji's islands were formed by volcanic activity starting around 150 million years ago."},
                   {'history_political_changes': 'Fiji operated as a Crown colony until 1970, became the Dominion of Fiji, and in 2014 held a democratic election after years of political turmoil.'}]}


Note how the model chose to return different fields like "Fiji_island_country" or "population_distribution".

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

We'll also use query_json(), now passing the json_schema:


```python
out = model.query_json(inst_text,
                       in_text,
                       json_schema=json_schema)
pp.pprint(out)
```

    {'keypoint_list': ['Fiji is an island country in Melanesia, part of Oceania in the South Pacific Ocean.',
                       "About 87% of Fiji's total population live on the two major islands, Viti Levu and Vanua Levu.",
                       "The majority of Fiji's islands were formed by volcanic activity starting around 150 million years ago.",
                       'Humans have lived in Fiji since the second millennium BC, first Austronesians and later Melanesians, with some Polynesian influences.',
                       "In 2014, a democratic election took place, with Bainimarama's FijiFirst party winning 59.2% of the vote."]}


It has generated a string list in the "keypoint_list" field, as we specified in the JSON schema.

This is better, but the problem with JSON schemas being that they're very unintuitive...

Let's use an easier way to specify the fields we want returned: Pydantic classes derived from BaseModel. This is much simpler to use than than JSON schemas.


```python
from pydantic import BaseModel, Field
from typing import List

# this class definition will be used to constrain the model output and initialize an instance object
class Keypoints(BaseModel):
    keypoint_list: List[str]

out = model.query_pydantic(Keypoints,
                           inst_text,
                           in_text)
pp.pprint(out)
```

    Keypoints(keypoint_list=['Fiji is an island country in Melanesia, part of Oceania in the South Pacific Ocean.', "About 87% of Fiji's total population live on the two major islands, Viti Levu and Vanua Levu.", "The majority of Fiji's islands were formed by volcanic activity starting around 150 million years ago.", 'Humans have lived in Fiji since the second millennium BC, first Austronesians and later Melanesians, with some Polynesian influences.', "In 2014, a democratic election took place in Fiji, with Bainimarama's FijiFirst party winning 59.2% of the vote."])


The query_pydantic() method returns an object (of class Keypoints) instantiated with the model output.

This allows much simpler handling of the model outputs.

Please see other examples for more interesting objects. In particular, we did not add descriptions to the fields, which are important to help the model understand what we want.

Sibila also includes a way to define the types and structure of Python dicts, called dictype, a lighter and easier alternative to using Pydantic.

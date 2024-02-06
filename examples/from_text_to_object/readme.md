In this example we'll ask the model to extract keypoints from a text:
- First in text format
- Then free JSON output
- Constrained by a JSON schema
- And finally by generating to a Pydantic object

We'll see this progression both with a local model and with GPT-4.

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

Create the local model and do our free text query with query_gen()


```python
from sibila import (
    ModelDir, GenConf
)

# adding models/ folder and its config
ModelDir.add("../../models/modeldir.json")
# to use some other local model, change the filename in the next line
name = "llamacpp:openchat-3.5-1210.Q4_K_M.gguf"

# delete an previous model if recreating
try: del local_model
except: ...

local_model = ModelDir.create(name)
```


```python
in_text = "Extract 5 keypoints of the following text:\n" + doc

out = local_model.query_gen(inst_text, in_text)
print(out)
```

    1. Fiji is an island country in Melanesia, part of Oceania in the South Pacific Ocean.
    2. It consists of more than 330 islands with a total land area of about 18,300 square kilometres (7,100 sq mi).
    3. About 87% of Fiji's population of 924,610 live on the two major islands, Viti Levu and Vanua Levu.
    4. The majority of Fiji's islands were formed by volcanic activity starting around 150 million years ago.
    5. Fiji gained independence from the British in 1970 and became a republic in 1987 after a series of coups d'état.


These are reasonale keypoints!

Let's now try GPT-4 - create the model object then query:


```python
oai_model = ModelDir.create("openai:gpt-4")

out = oai_model.query_gen(inst_text, in_text)
print(out)
```

    1. Fiji is an island country in Melanesia, part of Oceania, located about 1,100 nautical miles northeast of New Zealand, comprising over 330 islands and 500 islets with a total land area of approximately 18,300 square kilometers.
    
    2. The population of Fiji is around 924,610, with the majority living on the two main islands, Viti Levu and Vanua Levu, and the capital city is Suva, located on Viti Levu.
    
    3. Fiji's islands are mostly of volcanic origin, with ongoing geothermal activity on Vanua Levu and Taveuni, and non-volcanic geothermal systems on Viti Levu.
    
    4. Human settlement in Fiji dates back to the second millennium BC, with a history of Austronesian and Melanesian inhabitants, and later European contact in the 17th century. Fiji became a British colony in 1874 and gained independence in 1970.
    
    5. Fiji experienced political instability with several coups d'état, becoming a republic in 1987. After a military coup in 2006 and subsequent political turmoil, democratic elections were held in 2014, with the FijiFirst party winning and international observers recognizing the election as credible.


GPT-4 gave more comprehensive keypoints than the local model.

Let's now ask for JSON output, taking care to explicitly request it in the query (in_text variable).


```python
import pprint
pp = pprint.PrettyPrinter(width=300, sort_dicts=False)

in_text = "Extract 5 keypoints of the following text in JSON format:\n\n" + doc

out = local_model.query_json(inst_text, in_text)
pp.pprint(out)
```

    {'keypoints': [{'Fiji_island_country': 'Fiji is an island country in Melanesia, part of Oceania in the South Pacific Ocean.'},
                   {'geography': 'It lies about 1,100 nautical miles north-northeast of New Zealand and consists of more than 330 islands with a total land area of about 18,300 square kilometres.'},
                   {'population_distribution': 'About 87% of the total population of 924,610 live on the two major islands, Viti Levu and Vanua Levu.'},
                   {'island_formation': "The majority of Fiji's islands were formed by volcanic activity starting around 150 million years ago."},
                   {'history_political_changes': 'Fiji operated as a Crown colony until 1970, became the Dominion of Fiji, and in 2014 held a democratic election after years of political turmoil.'}]}


Instead of query_gen() we now use query_json() which returns a Python dict. 

Note how the model chose to return "title" and "description" keys.

What about GPT-4?


```python
out = oai_model.query_json(inst_text, in_text)
pp.pprint(out)
```

    {'keypoints': [{'location': 'Fiji is an island country in Melanesia, part of Oceania in the South Pacific Ocean.'},
                   {'geography': 'Fiji consists of more than 330 islands and over 500 islets, with a total land area of about 18,300 square kilometers.'},
                   {'population': "The majority of Fiji's population lives on the two major islands, Viti Levu and Vanua Levu, with a significant concentration in urban centers."},
                   {'history': 'Fiji has been inhabited since the second millennium BC and became an independent kingdom before becoming a British colony and gaining independence in 1970.'},
                   {'political': "Fiji experienced a series of coups d'état and became a republic in 1987, with a return to democratic elections in 2014, resulting in FijiFirst party's victory."}]}


Good keypoints also. Note that instead of "title" fields, the model chose to output them as key names.

Because we didn't specify which fields we want, each model generates different ones.

To specify a fixed format, let's generate by setting a JSON schema that defines which fields and types we want:


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

This JSON schema simply requests a list of strings.

Generate using local model, later GPT-4:


```python
out = local_model.query_json(inst_text,
                             in_text,
                             json_schema=json_schema)
pp.pprint(out)
```

    {'keypoint_list': ['Fiji is an island country in Melanesia, part of Oceania in the South Pacific Ocean.',
                       "About 87% of Fiji's total population live on the two major islands, Viti Levu and Vanua Levu.",
                       "The majority of Fiji's islands were formed by volcanic activity starting around 150 million years ago.",
                       'Humans have lived in Fiji since the second millennium BC, first Austronesians and later Melanesians, with some Polynesian influences.',
                       "In 2014, a democratic election took place, with Bainimarama's FijiFirst party winning 59.2% of the vote."]}



```python
out = oai_model.query_json(inst_text,
                           in_text,
                           json_schema=json_schema)
pp.pprint(out)
```

    {'keypoint_list': ['Fiji is an island country in Melanesia, part of Oceania in the South Pacific Ocean.',
                       'It comprises over 330 islands and 500 islets, with a total land area of about 18,300 square kilometers.',
                       'The majority of the population lives on the two major islands, Viti Levu and Vanua Levu, with the capital city being Suva.',
                       "Fiji's islands were mostly formed by volcanic activity, with some continuing geothermal activity today.",
                       'Fiji became independent from British colonial rule in 1970 and has experienced several coups, with the latest democratic election held in 2014.']}


It has generated a string list, as we specified in the JSON schema.

But the problem with JSON schemas is that they're very unintuitive.

Let's use Pydantic classes derived from BaseModel, to specify the fields we want returned. This is much simpler to use than than JSON schemas.


```python
from pydantic import BaseModel, Field
from typing import List

class Keypoints(BaseModel):
    keypoint_list: List[str]

out = local_model.query_pydantic(Keypoints,
                                 inst_text,
                                 in_text)
pp.pprint(out)
```

    Keypoints(keypoint_list=['Fiji is an island country in Melanesia, part of Oceania in the South Pacific Ocean.', "About 87% of Fiji's total population live on the two major islands, Viti Levu and Vanua Levu.", "The majority of Fiji's islands were formed by volcanic activity starting around 150 million years ago.", 'Humans have lived in Fiji since the second millennium BC, first Austronesians and later Melanesians, with some Polynesian influences.', "In 2014, a democratic election took place in Fiji, with Bainimarama's FijiFirst party winning 59.2% of the vote."])



```python
out = oai_model.query_pydantic(Keypoints,
                                 inst_text,
                                 in_text)
pp.pprint(out)
```

    Keypoints(keypoint_list=['Fiji is an island country in Melanesia, part of Oceania in the South Pacific Ocean, consisting of over 330 islands and 500 islets.', 'The country lies about 1,100 nautical miles north-northeast of New Zealand and has a total land area of about 18,300 square kilometers.', "The majority of Fiji's population lives on the two major islands, Viti Levu and Vanua Levu, with significant urban centers in Suva, Nadi, and Lautoka.", "Fiji's islands were mostly formed by volcanic activity, with ongoing geothermal activity on Vanua Levu and Taveuni.", 'Fiji has a complex history, including periods of British colonial rule, independence as a Dominion, military coups, and a return to democratic elections in 2014.'])


The query_pydantic() method returns an object (of class Keypoints) instantiated with the model output.

This allows much simpler handling of the model outputs. See other examples for more interesting objects. In particular, we did not add descriptions to the fields, which are important to help the model understand what we want.

Sibila also includes a way to define the types and structure of Python dicts, called dictype, a lighter and easier alternative to using Pydantic.

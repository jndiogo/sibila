In this example we'll extract information about all persons mentioned in a text.

Available as a [Jupyter notebook](extract.ipynb) or [Python script](extract.py).

To use a local model, make sure you have its file in the folder "../../models/". You can use any GGUF format model - [see here how to download the OpenChat model used below](https://jndiogo.github.io/sibila/setup-local-models/#default-model-used-in-the-examples-openchat). If you use a different one, don't forget to set its filename in the name variable below, after the text "llamacpp:".

To use an OpenAI model, make sure you defined the env variable OPENAI_API_KEY with a valid token and uncomment the line after "# to use an OpenAI model:".

Start by creating the model:


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

We'll use this text written in a flamboyant style, courtesy GPT three and a half:


```python
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
```


```python
from pydantic import BaseModel, Field
from typing import List
import pprint
pp = pprint.PrettyPrinter(width=300, sort_dicts=False)

class Person(BaseModel):
    first_name: str
    last_name: str
    age: int
    occupation: str
    source_location: str

class Info(BaseModel):
    person_list: List[Person]
    

in_text = "Extract person information from the following text:\n\n" + text

out = model.query_pydantic(Info,
                           inst_text,
                           in_text)
pp.pprint(out)
```

    Info(person_list=[Person(first_name='Lucy', last_name='Bennett', age=28, occupation='journalist', source_location='London'), Person(first_name='Carlos', last_name='Ramirez', age=35, occupation='architect', source_location='Barcelona'), Person(first_name='Mia', last_name='Chang', age=23, occupation='musician', source_location='Tokyo'), Person(first_name='Ahmed', last_name='Khan', age=40, occupation='engineer', source_location='Mumbai'), Person(first_name='Isabella', last_name='Santos', age=32, occupation='fashion designer', source_location='Paris')])



```python
for person in out.person_list:
    print(person)
```

    first_name='Lucy' last_name='Bennett' age=28 occupation='journalist' source_location='London'
    first_name='Carlos' last_name='Ramirez' age=35 occupation='architect' source_location='Barcelona'
    first_name='Mia' last_name='Chang' age=23 occupation='musician' source_location='Tokyo'
    first_name='Ahmed' last_name='Khan' age=40 occupation='engineer' source_location='Mumbai'
    first_name='Isabella' last_name='Santos' age=32 occupation='fashion designer' source_location='Paris'


It seems to be doing a good job of extracting the info we requested.

Let's add two more fields: the source country (which the model will have to figure from the source location) and a "details_about_person" field, which the model should quote from the info in the source text about each person.


```python
class Person(BaseModel):
    first_name: str
    last_name: str
    age: int
    occupation: str
    details_about_person: str
    source_location: str
    source_country: str

class Info(BaseModel):
    person_list: List[Person]
    
out = model.query_pydantic(Info,
                           inst_text,
                           in_text)

for person in out.person_list:
    print(person)
```

    first_name='Lucy' last_name='Bennett' age=28 occupation='journalist' details_about_person='her pen poised to capture the essence of the world around her' source_location='London' source_country='United Kingdom'
    first_name='Carlos' last_name='Ramirez' age=35 occupation='architect' details_about_person='exuded creativity, passion for design evident in the thoughtful lines that adorned his face' source_location='Barcelona' source_country='Spain'
    first_name='Mia' last_name='Chang' age=23 occupation='musician' details_about_person='fingers danced across the strings, weaving stories of love and longing' source_location='Tokyo' source_country='Japan'
    first_name='Ahmed' last_name='Khan' age=40 occupation='engineer' details_about_person='navigated the complexities of technology with ease, intellect shining through the chaos of urban life' source_location='Mumbai' source_country='India'
    first_name='Isabella' last_name='Santos' age=32 occupation='fashion designer' details_about_person='impeccable style and effortless grace reflected the timeless elegance of her beloved city' source_location='Paris' source_country='France'


Quite reasonable: the model is doing a good job and we didn't even add descriptions to the fields - it's inferring what we want from the field names only.

Let's now query an attribute that only one of the person have: being married. Adding the "is_married: bool" field to the Person class.


```python
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
```

    first_name='Lucy' last_name='Bennett' age=28 occupation='journalist' details_about_person='her pen poised to capture the essence of the world around her' source_location='London' source_country='United Kingdom' is_married=False
    first_name='Carlos' last_name='Ramirez' age=35 occupation='architect' details_about_person='exuded creativity, passion for design evident in the thoughtful lines that adorned his face' source_location='Barcelona' source_country='Spain' is_married=False
    first_name='Mia' last_name='Chang' age=23 occupation='musician' details_about_person='fingers danced across the strings, weaving stories of love and longing' source_location='Tokyo' source_country='Japan' is_married=False
    first_name='Ahmed' last_name='Khan' age=40 occupation='engineer' details_about_person='navigated the complexities of technology with ease, intellect shining through the chaos of urban life' source_location='Mumbai' source_country='India' is_married=True
    first_name='Isabella' last_name='Santos' age=32 occupation='fashion designer' details_about_person='impeccable style and effortless grace reflected the timeless elegance of her beloved city' source_location='Paris' source_country='France' is_married=False


From the five characters only Ahmed is mentioned to be married, and it is the one that the model marked with the is_married=True attribute.

This example is also available in a [dictype version here](readme_dictype.md).

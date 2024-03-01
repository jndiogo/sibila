This is the Python dataclass version of of the Pydantic extraction example. 

We'll extract information about all persons mentioned in a text.

To use a local model, make sure you have its file in the folder "../../models". You can use any GGUF format model - [see here how to download the OpenChat model used below](https://jndiogo.github.io/sibila/models/local_model/#examples). If you use a different one, don't forget to set its filename in the name variable below, after the text "llamacpp:".

To use an OpenAI model, make sure you defined the env variable OPENAI_API_KEY with a valid token and uncomment the line after "# to use an OpenAI model:".

Jupyter notebook and Python script versions are available in the example's folder.

Start by creating the model:


```python
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
from dataclasses import dataclass

@dataclass
class Person:
    first_name: str
    last_name: str
    age: int
    occupation: str
    source_location: str

# model instructions text, also known as system message
inst_text = "Extract information."

# the input query, including the above text
in_text = "Extract person information from the following text:\n\n" + text

out = model.extract(list[Person],
                    in_text,
                    inst=inst_text)

for person in out:
    print(person)
```

    Person(first_name='Lucy', last_name='Bennett', age=28, occupation='journalist', source_location='London')
    Person(first_name='Carlos', last_name='Ramirez', age=35, occupation='architect', source_location='Barcelona')
    Person(first_name='Mia', last_name='Chang', age=23, occupation='musician', source_location='Tokyo')
    Person(first_name='Ahmed', last_name='Khan', age=40, occupation='engineer', source_location='Mumbai')
    Person(first_name='Isabella', last_name='Santos', age=32, occupation='fashion designer', source_location='Paris')


It seems to be doing a good job of extracting the info we requested.

Let's add two more fields: the source country (which the model will have to figure from the source location) and a "details_about_person" field, which the model should quote from the info in the source text about each person.


```python
@dataclass
class Person:
    first_name: str
    last_name: str
    age: int
    occupation: str
    details_about_person: str
    source_location: str
    source_country: str

out = model.extract(list[Person],
                    in_text,
                    inst=inst_text)

for person in out:
    print(person)
```

    Person(first_name='Lucy', last_name='Bennett', age=28, occupation='journalist', details_about_person='a 28-year-old journalist from London, her pen poised to capture the essence of the world around her', source_location='London', source_country='United Kingdom')
    Person(first_name='Carlos', last_name='Ramirez', age=35, occupation='architect', details_about_person='a 35-year-old architect from the sun-kissed streets of Barcelona, with a sketchbook in hand, he exuded creativity', source_location='Barcelona', source_country='Spain')
    Person(first_name='Mia', last_name='Chang', age=23, occupation='musician', details_about_person='a 23-year-old musician from the bustling streets of Tokyo, her fingers danced across the strings, weaving stories of love and longing', source_location='Tokyo', source_country='Japan')
    Person(first_name='Ahmed', last_name='Khan', age=40, occupation='engineer', details_about_person='a married 40-year-old engineer from the bustling metropolis of Mumbai, with a laptop at his side, he navigated the complexities of technology with ease', source_location='Mumbai', source_country='India')
    Person(first_name='Isabella', last_name='Santos', age=32, occupation='fashion designer', details_about_person='a 32-year-old fashion designer from the romantic streets of Paris, her impeccable style and effortless grace reflected the timeless elegance of her beloved city', source_location='Paris', source_country='France')


Quite reasonable: the model is doing a good job and we didn't even add descriptions to the fields - it's inferring what we want from the field names only.

Let's now query an attribute that only one of the person have: being married. Adding the "is_married" field to the Person dataclass.


```python
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

out = model.extract(list[Person],
                    in_text,
                    inst=inst_text)

for person in out:
    print(person)
```

    Person(first_name='Lucy', last_name='Bennett', age=28, occupation='journalist', details_about_person='a 28-year-old journalist from London, her pen poised to capture the essence of the world around her', source_location='London', source_country='United Kingdom', is_married=False)
    Person(first_name='Carlos', last_name='Ramirez', age=35, occupation='architect', details_about_person='a 35-year-old architect from the sun-kissed streets of Barcelona, with a sketchbook in hand, he exuded creativity', source_location='Barcelona', source_country='Spain', is_married=False)
    Person(first_name='Mia', last_name='Chang', age=23, occupation='musician', details_about_person='a 23-year-old musician from the bustling streets of Tokyo, her fingers danced across the strings, weaving stories of love and longing', source_location='Tokyo', source_country='Japan', is_married=False)
    Person(first_name='Ahmed', last_name='Khan', age=40, occupation='engineer', details_about_person='a married 40-year-old engineer from the bustling metropolis of Mumbai, with a laptop at his side, he navigated the complexities of technology with ease', source_location='Mumbai', source_country='India', is_married=True)
    Person(first_name='Isabella', last_name='Santos', age=32, occupation='fashion designer', details_about_person='a 32-year-old fashion designer from the romantic streets of Paris, her impeccable style and effortless grace reflected the timeless elegance of her beloved city', source_location='Paris', source_country='France', is_married=False)


From the five characters only Ahmed is mentioned to be married, and it is the one that the model marked with the is_married=True attribute.

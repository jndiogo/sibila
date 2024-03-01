---
title: Dataclass
---

Besides simple types and enums, we can also extract objects whose structure is given by a dataclass definition:

!!! example
    ``` python
    from sibila import Models
    from dataclasses import dataclass

    Models.setup("../models")
    model = Models.create("llamacpp:openchat")

    @dataclass
    class Person:
        first_name: str
        last_name: str
        age: int
        occupation: str
        source_location: str

    in_text = """\
    Seated at a corner table was Lucy Bennett, a 28-year-old journalist from London, 
    her pen poised to capture the essence of the world around her. 
    Her eyes sparkled with curiosity, mirroring the dynamic energy of her beloved city.
    """

    model.extract(Person,
                  in_text)
    ```

    !!! success "Result"
        ``` python
        Person(first_name='Lucy', 
               last_name='Bennett',
               age=28, 
               occupation='journalist',
               source_location='London')
        ```


See the [Pydantic version here](pydantic.md).



We can extract a list of Person objects by using list[Person]:


!!! example
    ``` python
    in_text = """\
    Seated at a corner table was Lucy Bennett, a 28-year-old journalist from London, 
    her pen poised to capture the essence of the world around her. 
    Her eyes sparkled with curiosity, mirroring the dynamic energy of her beloved city.

    Opposite Lucy sat Carlos Ramirez, a 35-year-old architect from the sun-kissed 
    streets of Barcelona. With a sketchbook in hand, he exuded creativity, 
    his passion for design evident in the thoughtful lines that adorned his face.
    """

    model.extract(list[Person],
                  in_text)
    ```

    !!! success "Result"
        ``` python
        [Person(first_name='Lucy', 
                last_name='Bennett',
                age=28, 
                occupation='journalist',
                source_location='London'),
         Person(first_name='Carlos', 
                last_name='Ramirez',
                age=35,
                occupation='architect',
                source_location='Barcelona')]
        ```


As when extracting to simple types, we could also provide instructions by setting the inst argument.

Field annotations can also be provided to clarify what we want extracted for each field. This is done with Annotated[type, description"]:


!!! example
    ``` python
    from typing import Annotated

    Weekday = Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ]

    @dataclass
    class Period():
        start: Annotated[Weekday, "Day of arrival"]
        end: Annotated[Weekday, "Day of departure"]

    model.extract(Period,
                  "Right, well, I was planning to arrive on Wednesday and "
                  "only leave Sunday morning. Would that be okay?")
    ```

    !!! success "Result"
        ``` python
        Period(start='Wednesday', end='Sunday')
        ```


Check the [Extract dataclass example](../examples/extract_dataclass.md).


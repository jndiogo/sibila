---
title: Pydantic
---

Besides simple types and enums, we can also extract objects whose structure is given by a class derived from Pydantic's BaseModel definition:

!!! example
    ``` python
    from sibila import Models
    from pydantic import BaseModel

    Models.setup("../models")
    model = Models.create("llamacpp:openchat")

    class Person(BaseModel):
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


See the [dataclass version here](dataclass.md).



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


## Field annotations

As when extracting to simple types, we could also provide instructions by setting the inst argument. However, instructions are by nature general and when extracting structured data, it's harder to provide specific instructions for fields.

For this purpose, field annotations are more effective than instructions: they can be provided to clarify what we want extracted for each specific field.

For Pydantic this is done with Field(description="description") - see the "start" and "end" attributes of the Period class:


!!! example
    ``` python
    from pydantic import Field

    Weekday = Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ]

    class Period(BaseModel):
        start: Weekday = Field(description="Day of arrival")
        end: Weekday = Field(description="Day of departure")

    model.extract(Period,
                  "Right, well, I was planning to arrive on Wednesday and "
                  "only leave Sunday morning. Would that be okay?")
    ```

    !!! success "Result"
        ``` python
        Period(start='Wednesday', end='Sunday')
        ```


In this manner, the model can be informed of what is wanted for each specific field.


Check the [Extract Pydantic example](../examples/extract.md).

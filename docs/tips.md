# Tips and Tricks

Some general tips from experience with constrained model output in Sibila.

## Temperature

Sibila aims at exact results, so generation temperature defaults to 0. You should get the same results from the same model at all times.

For "creative" outputs, you can set the temperature to a non-zero value. This is done in GenConf, which can be passed in many places, for example during actual generation/extraction:


!!! example
    ``` python
    from sibila import (Models, GenConf)

    Models.setup("../models")

    model = Models.create("llamacpp:openchat") # default GenConf could be passed here

    for i in range(10):
        print(model.extract(int,
                    "Think of a random number between 1 and 100",
                    genconf=GenConf(temperature=2.)))
    ```

    !!! success "Result"
        ```
        72
        78
        75
        68
        39
        47
        53
        82
        72
        63
        ```




## Split entities into separate classes

Suppose you want to extract a list of person names from a group. You could use the following class:

``` python
class Group(BaseModel):
    persons: list[str] = Field(description="List of persons")
    group_info: str

out = model.extract(Group, in_text)
```

But it tends to work better to separate the Person entity into its own class and leave the list in Group:

``` python
class Person(BaseModel):
    name: str

class Group(BaseModel):
    persons: list[Person]
    group_info: str

out = model.extract(Group, in_text)
```

The same applies to the equivalent dataclass definitions.

Adding descriptions seems to always help, specially for non-trivial extraction. Without descriptions, the model can only look into variable names for clues on what's wanted, so it's important to tell it what we want by adding field descriptions.

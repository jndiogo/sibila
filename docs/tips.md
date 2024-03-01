# Tips and Tricks

Some general tips from experience with constrained model output in Sibila.


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

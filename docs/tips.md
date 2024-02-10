# Tips and Tricks

Some general tips from experience with constrained model output with Sibila.


## Split entities into separate classes

Suppose you want to extract a list of person names from a group. You could use the following class:

```python
class Group(BaseModel):
    persons: List[str] = Field(description="List of persons")

out = model.query_pydantic(Group,
                           inst_text,
                           in_text)
```

But it tends to work better to separate the Person entity into its own class and leave the list in Group:

```python
class Person(BaseModel):
    name: str

class Group(BaseModel):
    persons: List[Person]

out = model.query_pydantic(Group,
                           inst_text,
                           in_text)
```

The same applies to equivalent dictype definitions.

Adding descriptions seems to always help, specially for non-trivial extraction. Without descriptions, the model can only look into variable names for clues on what's wanted.

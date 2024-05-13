import pytest

from typing import Any, Optional, Union, Literal, Annotated, get_origin, get_args
import enum

from datetime import date, time, datetime

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from sibila.json_schema import (
    JSchemaConf,
    build_root_json_schema,
    json_schema_massage,
)



def test_json_schema_pydantic():

    class Person(BaseModel):
        name: str  
        occupation: str = Field(description="A peaceful occupation")

        # Optional, Union
        age0: Optional[int]
        age1: Optional[int] = Field(description="Description for age1")

        option0: Union[int,str]
        option1: Union[int,str] = Field(description="Description for option1")

        # defaults start here
        age2: Optional[int] = Field(default=None)
        age3: Optional[int] = Field(description="Description for age3", default=None)

        option2: Union[int,str] = 289
        option3: Union[int,str] = Field(description="Description for option1", default="Salsa")

        # default
        id0: str = "Lala"
        id1: str = Field(description="Description for id2", default="Loop")
        id2: str = Field(default_factory=lambda: "Lamp")
        id3: list[str] = Field(default_factory=lambda: ["Lamp"])
        id4: list[int] = Field(default_factory=lambda: [4,7])

        cloth: str = field(default="A value")

    json_schema1, _ = build_root_json_schema(Person, "output")
    print(json_schema1)

    assert json_schema1 == {
        'properties': {'name': {'title': 'Name', 'type': 'string'}, 
                       'occupation': {'description': 'A peaceful occupation', 'title': 'Occupation', 'type': 'string'}, 
                       'age0': {'anyOf': [{'type': 'integer'}, {'type': 'null'}], 'title': 'Age0'}, 
                       'age1': {'anyOf': [{'type': 'integer'}, {'type': 'null'}], 'description': 'Description for age1', 'title': 'Age1'}, 
                       'option0': {'anyOf': [{'type': 'integer'}, {'type': 'string'}], 'title': 'Option0'}, 
                       'option1': {'anyOf': [{'type': 'integer'}, {'type': 'string'}], 'description': 'Description for option1', 'title': 'Option1'}, 
                       'age2': {'anyOf': [{'type': 'integer'}, {'type': 'null'}], 'default': None, 'title': 'Age2'}, 
                       'age3': {'anyOf': [{'type': 'integer'}, {'type': 'null'}], 'default': None, 'description': 'Description for age3', 'title': 'Age3'}, 
                       'option2': {'anyOf': [{'type': 'integer'}, {'type': 'string'}], 'default': 289, 'title': 'Option2'}, 
                       'option3': {'anyOf': [{'type': 'integer'}, {'type': 'string'}], 'default': 'Salsa', 'description': 'Description for option1', 'title': 'Option3'}, 
                       'id0': {'default': 'Lala', 'title': 'Id0', 'type': 'string'}, 
                       'id1': {'default': 'Loop', 'description': 'Description for id2', 'title': 'Id1', 'type': 'string'}, 
                       'id2': {'title': 'Id2', 'type': 'string'}, 
                       'id3': {'items': {'type': 'string'}, 'title': 'Id3', 'type': 'array'}, 
                       'id4': {'items': {'type': 'integer'}, 'title': 'Id4', 'type': 'array'}, 
                       'cloth': {'default': 'A value', 'title': 'Cloth', 'type': 'string'}},                        
                       'required': ['name', 'occupation', 'age0', 'age1', 'option0', 'option1'], 
                       'title': 'Person', 
                       'type': 'object'}


    json_schema2 = json_schema_massage(json_schema1)

    assert json_schema2 == {
        'properties': {'name': {'type': 'string'},
   'occupation': {'description': 'A peaceful occupation', 'type': 'string'},
   'age0': {'anyOf': [{'type': 'integer'}, {'type': 'null'}]},
   'age1': {'anyOf': [{'type': 'integer'}, {'type': 'null'}],
    'description': 'Description for age1'},
   'option0': {'anyOf': [{'type': 'integer'}, {'type': 'string'}]},
   'option1': {'anyOf': [{'type': 'integer'}, {'type': 'string'}],
    'description': 'Description for option1'},
   'age2': {'anyOf': [{'type': 'integer'}, {'type': 'null'}], 'default': None},
   'age3': {'anyOf': [{'type': 'integer'}, {'type': 'null'}],
    'description': 'Description for age3',
    'default': None},
   'option2': {'anyOf': [{'type': 'integer'}, {'type': 'string'}],
    'default': 289},
   'option3': {'anyOf': [{'type': 'integer'}, {'type': 'string'}],
    'description': 'Description for option1',
    'default': 'Salsa'},
   'id0': {'type': 'string', 'default': 'Lala'},
   'id1': {'description': 'Description for id2',
    'type': 'string',
    'default': 'Loop'},
   'id2': {'type': 'string'},
   'id3': {'items': {'type': 'string'}, 'type': 'array'},
   'id4': {'items': {'type': 'integer'}, 'type': 'array'},
   'cloth': {'type': 'string', 'default': 'A value'}},
  'required': ['name', 'occupation', 'age0', 'age1', 'option0', 'option1'],
  'type': 'object'}






def test_json_schema_pydantic2():

    class AAA(BaseModel):
        zoom: int
        zaam: str
        
    class Person(BaseModel):
        name: str  
        song: AAA
        occupation: str  
        location: str = "lala"
        age: Optional[AAA] = None
        unio: Union[int,str]

    json_schema1, _ = build_root_json_schema(Person, "output")
    print(json_schema1)

    assert json_schema1 == {'$defs': {'AAA': {'properties': {'zoom': {'title': 'Zoom',
      'type': 'integer'},
     'zaam': {'title': 'Zaam', 'type': 'string'}},
    'required': ['zoom', 'zaam'],
    'title': 'AAA',
    'type': 'object'}},
  'properties': {'name': {'title': 'Name', 'type': 'string'},
   'song': {'$ref': '#/$defs/AAA'},
   'occupation': {'title': 'Occupation', 'type': 'string'},
   'location': {'default': 'lala', 'title': 'Location', 'type': 'string'},
   'age': {'anyOf': [{'$ref': '#/$defs/AAA'}, {'type': 'null'}],
    'default': None},
   'unio': {'anyOf': [{'type': 'integer'}, {'type': 'string'}],
    'title': 'Unio'}},
  'required': ['name', 'song', 'occupation', 'unio'],
  'title': 'Person',
  'type': 'object'}


    json_schema2 = json_schema_massage(json_schema1)

    assert json_schema2 == {'properties': {'name': {'type': 'string'},
   'song': {'properties': {'zoom': {'type': 'integer'},
     'zaam': {'type': 'string'}},
    'required': ['zoom', 'zaam'],
    'type': 'object'},
   'occupation': {'type': 'string'},
   'location': {'type': 'string', 'default': 'lala'},
   'age': {'anyOf': [{'properties': {'zoom': {'type': 'integer'},
       'zaam': {'type': 'string'}},
      'required': ['zoom', 'zaam'],
      'type': 'object'},
     {'type': 'null'}],
    'default': None},
   'unio': {'anyOf': [{'type': 'integer'}, {'type': 'string'}]}},
  'required': ['name', 'song', 'occupation', 'unio'],
  'type': 'object'}





def test_json_schema_dataclass():

    @dataclass
    class Person:
        name: str  
        occupation: Annotated[str, "A peaceful occupation"]

        # Optional, Union
        age0: Optional[int]
        age1: Annotated[Optional[int], "Description for age1"]

        option0: Union[int,str]
        option1: Annotated[Union[int,str], "Description for option1"]

        # defaults start here
        age2: Optional[int] = None
        age3: Annotated[Optional[int], "Description for age3"] = None

        option2: Union[int,str] = 289
        option3: Annotated[Union[int,str], "Description for option1"] = "Salsa"

        # default
        id0: str = "Lala"
        id1: Annotated[str, "Description for id2"] = "Loop"
        id2: str = field(default_factory=lambda: "Lamp")
        id3: list[str] = field(default_factory=lambda: ["Lamp"])
        id4: list[int] = field(default_factory=lambda: [4,7])

        cloth: str = field(default="A value")


    json_schema1, _ = build_root_json_schema(Person, "output")

    assert json_schema1 == {
        'properties': {'name': {'type': 'string'},
    'occupation': {'type': 'string', 'description': 'A peaceful occupation'},
    'age0': {'anyOf': [{'type': 'integer'}, {'type': 'null'}]},
    'age1': {'anyOf': [{'type': 'integer'}, {'type': 'null'}],
        'description': 'Description for age1'},
    'option0': {'anyOf': [{'type': 'integer'}, {'type': 'string'}]},
    'option1': {'anyOf': [{'type': 'integer'}, {'type': 'string'}],
        'description': 'Description for option1'},
    'age2': {'anyOf': [{'type': 'integer'}, {'type': 'null'}], 'default': None},
    'age3': {'anyOf': [{'type': 'integer'}, {'type': 'null'}],
        'description': 'Description for age3',
        'default': None},
    'option2': {'anyOf': [{'type': 'integer'}, {'type': 'string'}],
        'default': 289},
    'option3': {'anyOf': [{'type': 'integer'}, {'type': 'string'}],
        'description': 'Description for option1',
        'default': 'Salsa'},
    'id0': {'type': 'string', 'default': 'Lala'},
    'id1': {'type': 'string',
        'description': 'Description for id2',
        'default': 'Loop'},
    'id2': {'type': 'string', 'default': 'Lamp'},
    'id3': {'items': {'type': 'string'}, 'type': 'array', 'default': ['Lamp']},
    'id4': {'items': {'type': 'integer'}, 'type': 'array', 'default': [4, 7]},
    'cloth': {'type': 'string', 'default': 'A value'}},
    'required': ['name', 'occupation', 'age0', 'age1', 'option0', 'option1'],
    'type': 'object'}

    json_schema2 = json_schema_massage(json_schema1)

    assert json_schema2 == {
        'properties': {'name': {'type': 'string'},
   'occupation': {'type': 'string', 'description': 'A peaceful occupation'},
   'age0': {'anyOf': [{'type': 'integer'}, {'type': 'null'}]},
   'age1': {'anyOf': [{'type': 'integer'}, {'type': 'null'}],
    'description': 'Description for age1'},
   'option0': {'anyOf': [{'type': 'integer'}, {'type': 'string'}]},
   'option1': {'anyOf': [{'type': 'integer'}, {'type': 'string'}],
    'description': 'Description for option1'},
   'age2': {'anyOf': [{'type': 'integer'}, {'type': 'null'}], 'default': None},
   'age3': {'anyOf': [{'type': 'integer'}, {'type': 'null'}],
    'description': 'Description for age3',
    'default': None},
   'option2': {'anyOf': [{'type': 'integer'}, {'type': 'string'}],
    'default': 289},
   'option3': {'anyOf': [{'type': 'integer'}, {'type': 'string'}],
    'description': 'Description for option1',
    'default': 'Salsa'},
   'id0': {'type': 'string', 'default': 'Lala'},
   'id1': {'type': 'string',
    'description': 'Description for id2',
    'default': 'Loop'},
   'id2': {'type': 'string', 'default': 'Lamp'},
   'id3': {'items': {'type': 'string'}, 'type': 'array', 'default': ['Lamp']},
   'id4': {'items': {'type': 'integer'}, 'type': 'array', 'default': [4, 7]},
   'cloth': {'type': 'string', 'default': 'A value'}},
  'required': ['name', 'occupation', 'age0', 'age1', 'option0', 'option1'],
  'type': 'object'}
    



def test_json_schema_dataclass2():

    @dataclass
    class AAA():  
        zoom: int
        zaam: str
        
    @dataclass
    class Person():  
        name: str  
        song: AAA
        occupation: str  
        unio: Union[int,str]
        koor: int

        location: str = "lala"
        age: Optional[AAA] = None
        age: AAA = field(default_factory=lambda: AAA(2,"two")) # = Sub(2)
        rage: list[int] = field(default_factory=lambda: [2,4]) # field(default_factory=list)
        location: str = "lala"

    json_schema1, _ = build_root_json_schema(Person, "output")


    assert json_schema1 == {'properties': {'name': {'type': 'string'},
   'song': {'properties': {'zoom': {'type': 'integer'},
     'zaam': {'type': 'string'}},
    'required': ['zoom', 'zaam'],
    'type': 'object'},
   'occupation': {'type': 'string'},
   'unio': {'anyOf': [{'type': 'integer'}, {'type': 'string'}]},
   'koor': {'type': 'integer'},
   'location': {'type': 'string', 'default': 'lala'},
   'age': {'properties': {'zoom': {'type': 'integer'},
     'zaam': {'type': 'string'}},
    'required': ['zoom', 'zaam'],
    'type': 'object',
    'default': {'zoom': 2, 'zaam': 'two'}},
   'rage': {'items': {'type': 'integer'}, 'type': 'array', 'default': [2, 4]}},
  'required': ['name', 'song', 'occupation', 'unio', 'koor'],
  'type': 'object'}

    json_schema2 = json_schema_massage(json_schema1)

    assert json_schema2 == {'properties': {'name': {'type': 'string'},
   'song': {'properties': {'zoom': {'type': 'integer'},
     'zaam': {'type': 'string'}},
    'required': ['zoom', 'zaam'],
    'type': 'object'},
   'occupation': {'type': 'string'},
   'unio': {'anyOf': [{'type': 'integer'}, {'type': 'string'}]},
   'koor': {'type': 'integer'},
   'location': {'type': 'string', 'default': 'lala'},
   'age': {'properties': {'zoom': {'type': 'integer'},
     'zaam': {'type': 'string'}},
    'required': ['zoom', 'zaam'],
    'type': 'object',
    'default': {'zoom': 2, 'zaam': 'two'}},
   'rage': {'items': {'type': 'integer'}, 'type': 'array', 'default': [2, 4]}},
  'required': ['name', 'song', 'occupation', 'unio', 'koor'],
  'type': 'object'}



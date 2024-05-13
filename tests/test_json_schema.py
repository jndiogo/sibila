import pytest

from typing import Any, Optional, Union, Literal, Annotated, get_origin, get_args
import enum

from datetime import date, time, datetime

from dataclasses import dataclass

from pydantic import BaseModel, Field

from sibila.json_schema import (
    JSchemaConf,
    get_type,
    get_type_list,
    build_root_json_schema,
    get_final_type,
    create_final_instance,
)



def test_get_type_prims():

    # prim_types
    res = get_type(bool)
    assert res == (bool, None, {})

    res = get_type(int)
    assert res == (int, None, {})

    res = get_type(float)
    assert res == (float, None, {})

    res = get_type(str)
    assert res == (str, None, {})

    res = get_type(Annotated[str, "Item desc"])
    assert res == (str, 'Item desc', {})





    
def test_get_type_enums():

    # List enums
    res = get_type([True,False])
    assert res == (bool, None, {"enum_list": [True, False]})

    res = get_type([1,2,3])
    assert res == (int, None, {"enum_list": [1, 2, 3]})

    res = get_type([1.0,2.0,3.0])
    assert res == (float, None, {"enum_list": [1.0, 2.0, 3.0]})

    res = get_type(["good", "neutral", "bad"])
    assert res == (str, None, {"enum_list": ['good', 'neutral', 'bad']})

    # Annotated is not allowed with [type]

    with pytest.raises(TypeError):
        res = get_type([1,2,"3"])

    class Good():
        a=1
    class Bad():
        a=2
    with pytest.raises(TypeError):
        res = get_type([Good, Bad])



    # Literals
    res = get_type(Literal[True,False])
    assert res == (bool, None, {"enum_list": [True, False]})

    res = get_type(Literal[1,2,3])
    assert res == (int, None, {"enum_list": [1, 2, 3]})

    res = get_type(Annotated[Literal[1,2,3], "Desc"])
    assert res == (int, "Desc", {"enum_list": [1, 2, 3]})

    with pytest.raises(TypeError):
        res = get_type(Literal[1,2.0,3])



    # Enum* classes
    class Color(enum.Enum):
        RED = 1
        GREEN = 2
        BLUE = 3    
    res = get_type(Color)
    assert res == (int, None, {"enum_list": [1, 2, 3]})

    class Color(enum.IntEnum):
        RED = 1
        GREEN = 2
        BLUE = 3    
    res = get_type(Color)
    assert res == (int, None, {"enum_list": [1, 2, 3]})

    class Color(enum.StrEnum):
        RED = "1"
        GREEN = "2"
        BLUE = "3"
    res = get_type(Color)
    assert res == (str, None, {"enum_list": ["1", "2", "3"]})

    class Color(enum.Enum):
        RED = True
        GREEN = False
        BLUE = True
    res = get_type(Color)
    assert res == (bool, None, {"enum_list": [True, False]})


    class Color(enum.IntEnum):
        RED = 1
        GREEN = 2
        BLUE = 3    
    res = get_type(Annotated[Color, "Desc"])
    assert res == (int, "Desc", {"enum_list": [1, 2, 3]})


    class Color(enum.Enum):
        RED = "1"
        GREEN = "2"
        BLUE = 3
    with pytest.raises(TypeError):
        res = get_type(Color)




def test_get_type_datetime():

    res = get_type(datetime)
    assert res == (str, None, {'str_format': 'date-time'})

    res = get_type(Annotated[datetime,"desc"])
    assert res == (str, 'desc', {'str_format': 'date-time'})
    

    res = get_type(date)
    assert res == (str, None, {'str_format': 'date'})

    res = get_type(Annotated[date,"desc"])
    assert res == (str, 'desc', {'str_format': 'date'})


    res = get_type(time)
    assert res == (str, None, {'str_format': 'time'})

    res = get_type(Annotated[time,"desc"])
    assert res == (str, 'desc', {'str_format': 'time'})



def test_get_type_dataclass():

    @dataclass
    class Invent:
        """Class for keeping track of an item in inventory."""
        name: str
        unit_price: float
        counts_list: list[int]
        anno_field: Annotated[str, "This is anno"] = "30"
        quantity: int = 0

    res = get_type(Invent)
    assert res == (Invent, None, {})

    res = get_type(Annotated[Invent, "Desc"])
    assert res == (Invent, 'Desc', {})





def test_get_type_BaseModel():

    class UserDetail(BaseModel):
        """ Details of a user """
        name: str
        age: int
    
    res = get_type(UserDetail)
    assert res == (UserDetail, None, {})

    res = get_type(Annotated[UserDetail, "Desc"])
    assert res == (UserDetail, 'Desc', {})









def test_get_type_list():

    # list[prim_type]
    res = get_type_list(list[int])
    assert res == (int, None, {}, None)

    res = get_type_list(Annotated[list[str], "List desc"])
    assert res == (str, None, {}, 'List desc')

    res = get_type_list(list[Annotated[str, "Item desc"]])
    assert res == (str, 'Item desc', {}, None)

    res = get_type_list(Annotated[list[Annotated[str, "Item desc"]], "List desc"])
    assert res == (str, 'Item desc', {}, 'List desc')



    # enums
    res = get_type_list(list[Literal["a", "b"]])
    assert res == (str, None, {"enum_list": ['a', 'b']}, None)

    class Color(enum.IntEnum):
        RED = 1
        GREEN = 2
        BLUE = 3    

    res = get_type_list(list[Color])
    assert res == (int, None, {"enum_list": [1, 2, 3]}, None)



    # datetime
    res = get_type_list(list[datetime])
    assert res == (str, None, {'str_format': 'date-time'}, None)

    res = get_type_list(Annotated[list[datetime], "List desc"])
    assert res == (str, None, {'str_format': 'date-time'}, 'List desc')


    res = get_type_list(list[date])
    assert res == (str, None, {'str_format': 'date'}, None)

    res = get_type_list(Annotated[list[date], "List desc"])
    assert res == (str, None, {'str_format': 'date'}, 'List desc')


    res = get_type_list(list[time])
    assert res == (str, None, {'str_format': 'time'}, None)

    res = get_type_list(Annotated[list[time], "List desc"])
    assert res == (str, None, {'str_format': 'time'}, 'List desc')


    # dataclass
    @dataclass
    class Invent:
        """Class for keeping track of an item in inventory."""
        name: str
        unit_price: float
        counts_list: list[int]
        anno_field: Annotated[str, "This is anno"] = "30"
        quantity: int = 0

    res = get_type_list(list[Invent])
    assert res == (Invent, None, {}, None)

    res = get_type_list(Annotated[list[Invent], "list desc"])
    assert res == (Invent, None, {}, 'list desc')




    # BaseModel
    class UserDetail(BaseModel):
        """ Details of a user """
        name: str
        age: int

    res = get_type_list(list[UserDetail])
    assert res == (UserDetail, None, {}, None)
    
    res = get_type_list(list[Annotated[UserDetail, "Item desc"]])
    assert res == (UserDetail, 'Item desc', {}, None)

    res = get_type_list(Annotated[list[UserDetail], "List desc"])
    assert res == (UserDetail, None, {}, 'List desc')

    res = get_type_list(Annotated[list[Annotated[UserDetail, "Item desc"]], "List desc"])
    assert res == (UserDetail, "Item desc", {}, 'List desc')

    class Good():
        a=1
    with pytest.raises(TypeError):
        res = get_type_list(list[Good])

    with pytest.raises(TypeError):
        res = get_type_list(list[list[int]])








def test_build_root_json_schema():
    """ Only base types, not BaseModel, whose json_schema generation is external """

    res = build_root_json_schema(int, "output")
    assert res == ({'properties': {'output': {'type': 'integer'}},
                    'required': ['output'],
                    'type': 'object'},
                    True)


    res = build_root_json_schema(list[int], "output")
    assert res == ({'properties': {'output': {'items': {'type': 'integer'}, 'type': 'array'}},
                    'required': ['output'],
                    'type': 'object'},
                    True)


    # datetime
    res = build_root_json_schema(datetime, "output")
    assert res == ({'properties': {'output': {'format': 'date-time', 'type': 'string'}},
                    'required': ['output'],
                    'type': 'object'},
                    True)

    res = build_root_json_schema(list[datetime], "output")
    assert res == ({'properties': {'output': {'items': {'format': 'date-time', 'type': 'string'},
                        'type': 'array'}},
                    'required': ['output'],
                    'type': 'object'},
                    True)



    res = build_root_json_schema(date, "output")
    assert res == ({'properties': {'output': {'format': 'date', 'type': 'string'}},
                    'required': ['output'],
                    'type': 'object'},
                    True)

    res = build_root_json_schema(list[date], "output")
    assert res == ({'properties': {'output': {'items': {'format': 'date', 'type': 'string'},
                        'type': 'array'}},
                    'required': ['output'],
                    'type': 'object'},
                    True)



    res = build_root_json_schema(time, "output")
    assert res == ({'properties': {'output': {'format': 'time', 'type': 'string'}},
                    'required': ['output'],
                    'type': 'object'},
                    True)

    res = build_root_json_schema(list[time], "output")
    assert res == ({'properties': {'output': {'items': {'format': 'time', 'type': 'string'},
                    'type': 'array'}},
                    'required': ['output'],
                    'type': 'object'},
                    True)


    # dataclass
    @dataclass
    class Invent:
        """Class for keeping track of an item in inventory."""
        name: str
        counts_list: list[int]
        unit_price: float = 0.1
        anno_field: Annotated[str, "This is anno"] = "30"

    res = build_root_json_schema(Invent, "output")
    assert res == ({'description': 'Class for keeping track of an item in inventory.',
                    'properties': {'name': {'type': 'string'},
                                'counts_list': {'items': {'type': 'integer'}, 'type': 'array'},
                                'unit_price': {'type': 'number', 'default': 0.1},
                                'anno_field': {'description': 'This is anno',
                                    'type': 'string',
                                    'default': '30'}
                                },
                    'required': ['name', 'counts_list'],
                    'type': 'object'},
                    False)
    
    res = build_root_json_schema(Annotated[Invent, "Invent desc"], "output")
    assert res == ({'description': 'Invent desc',
                    'properties': {'name': {'type': 'string'},
                    'counts_list': {'items': {'type': 'integer'}, 'type': 'array'},
                    'unit_price': {'type': 'number', 'default': 0.1},
                    'anno_field': {'description': 'This is anno',
                        'type': 'string',
                        'default': '30'}},
                    'required': ['name', 'counts_list'],
                    'type': 'object'},
                    False)

    res = build_root_json_schema(list[Invent], "output")
    assert res == ({'properties': {'output': {'items': {'description': 'Class for keeping track of an item in inventory.',
                                        'properties': {'name': {'type': 'string'},
                                        'counts_list': {'items': {'type': 'integer'}, 'type': 'array'},
                                        'unit_price': {'type': 'number', 'default': 0.1},
                                        'anno_field': {'description': 'This is anno',
                                        'type': 'string',
                                        'default': '30'}},
                                        'required': ['name', 'counts_list'],
                                        'type': 'object'},
                                        'type': 'array'}
                                    },
                    'required': ['output'],
                    'type': 'object'},
                    True)


    @dataclass
    class Inner:
        """Class Inner."""
        iname: str
        ifield: int = 2
        
    @dataclass
    class Invent:
        """Class for keeping track of an item in inventory."""
        inner: Inner
        num: int = 4

    res = build_root_json_schema(Invent, "output")
    assert res == ({'description': 'Class for keeping track of an item in inventory.',
                    'properties': {'inner': {'description': 'Class Inner.',
                        'properties': {'iname': {'type': 'string'},
                        'ifield': {'type': 'integer', 'default': 2}},
                        'required': ['iname'],
                        'type': 'object'},
                    'num': {'type': 'integer', 'default': 4}},
                    'required': ['inner'],
                    'type': 'object'},
                    False)
    














def test_get_final_type():

    res = get_final_type(int)
    assert res == (int, False)

    res = get_final_type([1.0,2.0])
    assert res == (float, False)

    res = get_final_type(Literal[1,2])
    assert res == (int, False)

    class Color(enum.IntEnum):
        RED = 1
        GREEN = 2
        BLUE = 3    
    res = get_final_type(Color)
    assert res == (Color, False)


    # datetime
    res = get_final_type(datetime)
    assert res == (datetime, False)

    res = get_final_type(date)
    assert res == (date, False)

    res = get_final_type(time)
    assert res == (time, False)


    # BaseModel
    class UserDetail(BaseModel):
        """ Details of a user """
        name: str
        age: int

    res = get_final_type(UserDetail)
    assert res == (UserDetail, False)


    # lists
    target = list[UserDetail]
    res = get_final_type(target)
    assert res == (UserDetail, True)

    target = Annotated[list[UserDetail], "Desc"]
    res = get_final_type(target)
    assert res == (UserDetail, True)





def test_create_final_instance():

    res = create_final_instance(datetime, False, "1984-05-11T15:10:00", JSchemaConf())
    assert res == datetime(1984, 5, 11, 15, 10)

    res = create_final_instance(datetime, True, 
                                ["1984-05-11T15:10:00", "2014-06-05T00:00:00"], 
                                JSchemaConf())
    assert res == [datetime(1984, 5, 11, 15, 10), datetime(2014, 6, 5, 0, 0)]


    res = create_final_instance(date, False, "1984-05-11", JSchemaConf())
    assert res == date(1984, 5, 11)

    res = create_final_instance(date, True, 
                                ["1984-05-11", "2014-06-05"], 
                                JSchemaConf())
    assert res == [date(1984, 5, 11), date(2014, 6, 5)]


    res = create_final_instance(time, False, "15:10:00", JSchemaConf())
    assert res == time(15, 10)

    res = create_final_instance(time, True, 
                                ["15:10:00", "00:00:00"], 
                                JSchemaConf())
    assert res == [time(15, 10), time(0, 0)]




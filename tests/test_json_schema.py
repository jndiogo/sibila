import pytest

from typing import Any, Optional, Union, Literal, Annotated, get_origin, get_args
import enum

from sibila.json_utils import (
    get_type,
    get_type_list,
    build_type_json_schema,
    build_array_type_json_schema,
    build_object_type_json_schema
)



def test_get_type_prims():

    kwargs = {"allow_enums": False,
              "allow_BaseModel": False,
              "allow_dictype": False}
    
    # prim_types
    res = get_type(bool, **kwargs)
    assert res == (bool, None, None)

    res = get_type(int, **kwargs)
    assert res == (int, None, None)

    res = get_type(float, **kwargs)
    assert res == (float, None, None)

    res = get_type(str, **kwargs)
    assert res == (str, None, None)

    res = get_type(Annotated[str, "Item desc"], **kwargs)
    assert res == (str, 'Item desc', None)





    
def test_get_type_enums():

    kwargs = {"allow_enums": False, # <--------------------------
              "allow_BaseModel": False,
              "allow_dictype": False}

    # List enums
    res = get_type(["good", "neutral", "bad"], **kwargs)
    assert res == (None, None, None)

    res = get_type([True, False], **kwargs)
    assert res == (None, None, None)

    # Literals
    res = get_type(Literal[1,2,3], **kwargs)
    assert res == (None, None, None)





    kwargs = {"allow_enums": True, # <--------------------------
              "allow_BaseModel": False,
              "allow_dictype": False}

    # List enums
    res = get_type([True,False], **kwargs)
    assert res == (bool, None, [True, False])

    res = get_type([1,2,3], **kwargs)
    assert res == (int, None, [1, 2, 3])

    res = get_type([1.0,2.0,3.0], **kwargs)
    assert res == (float, None, [1.0, 2.0, 3.0])

    res = get_type(["good", "neutral", "bad"], **kwargs)
    assert res == (str, None, ['good', 'neutral', 'bad'])

    # Annotated is not allowed with [type]

    with pytest.raises(TypeError):
        res = get_type([1,2,"3"], **kwargs)

    class Good():
        a=1
    class Bad():
        a=2
    with pytest.raises(TypeError):
        res = get_type([Good, Bad], **kwargs)



    # Literals
    res = get_type(Literal[True,False], **kwargs)
    assert res == (bool, None, [True, False])

    res = get_type(Literal[1,2,3], **kwargs)
    assert res == (int, None, [1, 2, 3])

    res = get_type(Annotated[Literal[1,2,3], "Desc"], **kwargs)
    assert res == (int, "Desc", [1, 2, 3])

    with pytest.raises(TypeError):
        res = get_type(Literal[1,2.0,3], **kwargs)



    # Enum* classes
    class Color(enum.Enum):
        RED = 1
        GREEN = 2
        BLUE = 3    
    res = get_type(Color, **kwargs)
    assert res == (int, None, [1, 2, 3])

    class Color(enum.IntEnum):
        RED = 1
        GREEN = 2
        BLUE = 3    
    res = get_type(Color, **kwargs)
    assert res == (int, None, [1, 2, 3])

    class Color(enum.StrEnum):
        RED = "1"
        GREEN = "2"
        BLUE = "3"
    res = get_type(Color, **kwargs)
    assert res == (str, None, ["1", "2", "3"])

    class Color(enum.Enum):
        RED = True
        GREEN = False
        BLUE = True
    res = get_type(Color, **kwargs)
    assert res == (bool, None, [True, False])


    class Color(enum.IntEnum):
        RED = 1
        GREEN = 2
        BLUE = 3    
    res = get_type(Annotated[Color, "Desc"], **kwargs)
    assert res == (int, "Desc", [1, 2, 3])


    class Color(enum.Enum):
        RED = "1"
        GREEN = "2"
        BLUE = 3
    with pytest.raises(TypeError):
        res = get_type(Color, **kwargs)







def test_get_type_list():

    # list[prim_type]
    res = get_type_list(list[int])
    assert res == (int, None, None)


    res = get_type_list(Annotated[list[str], "List desc"])
    assert res == (str, 'List desc', None)

    res = get_type_list(list[Annotated[str, "Item desc"]])
    assert res == (str, None, 'Item desc')

    res = get_type_list(Annotated[list[Annotated[str, "Item desc"]], "List desc"])
    assert res == (str, 'List desc', 'Item desc')

    class Good():
        a=1
    with pytest.raises(TypeError):
        res = get_type_list(list[Good])

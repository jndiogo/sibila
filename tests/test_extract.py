import pytest

from typing import Any, Optional, Union, Literal, Annotated, get_origin, get_args
import enum

from datetime import date, time, datetime, timezone

from dataclasses import dataclass, is_dataclass, fields, MISSING

import logging
logging.basicConfig(level=logging.DEBUG)

from pydantic import BaseModel, Field

from sibila.json_schema import (
    JSchemaConf
)

from sibila.null import NullModel


model = NullModel()


def extract(target: Any,
             text: str,
             response: str,
             result: Any,
             *,
             inst: Optional[str]=None):
    model.set_response(response)
    assert model.extract(target, text, inst=inst) == result

def classify(target: Any,
             text: str,
             response: str,
             result: Any,
             *,
             inst: Optional[str]=None):
    model.set_response(response)
    assert model.classify(target, text, inst=inst) == result






def test_extract_root_types():

    extract(bool,
            "It's a great time to surf",
            '{"output": true}',
            True)

    extract(bool,
            "I'll never do it",
            '{"output": false}',
            False)

    extract(list[Literal["dog", "horse", "car", "bus"]], 
             "Dogs and horses and tractors and a school bus",
             '{"output": ["dog", "horse", "car", "bus"]}',
             ['dog', 'horse', 'car', 'bus'])


    from enum import Enum
    class Tag(Enum):
        DOG = "dog"
        HORSE = "horse"
        CAR = "car"
        BUS = "bus"
        OTHER = "other"

    extract(list[Tag],
            "Dogs and horses and a ball",
            '{"output": ["dog", "horse", "other"]}',
            ['dog', 'horse', 'other'],
            inst="Select 'other' if no other category fits")


    extract(list[int],
            "21 years and ten ponies. Five elephants?",
            '{"output": [21, 10, 5]}',
            [21, 10, 5],
            inst="Extract numbers")





    extract(datetime, 
            "Eleven of May, 1984, 10 past 15 AM.", 
            '{"output":"1984-05-11T15:10:00+00:00"}',
            datetime(1984, 5, 11, 15, 10, tzinfo=timezone.utc),
            inst="Date and time in ISO8601 format")




    @dataclass
    class Inv:
        """Class for keeping track of an item in inventory."""
        name: str
        unit_price: float
        alistia: list[int]
        anno: Annotated[str, "This is anno"] = "30"
        quantity_on_hand: int = 0

    extract(Inv,
            "A lot of nice things but no meaning at all",
            '{ "name": "Item", "unit_price": 0, "alistia": [], "anno": "30", "quantity_on_hand": 0}',
            Inv(name='Item', unit_price=0, alistia=[], anno='30', quantity_on_hand=0))





    class UserDetail(BaseModel):
        """ Details about a user """
        name: str
        age: int

    extract(list[UserDetail],
            "Paul and his buddy Mathilda, they love to ride. He's 68 years old, she's 81.",
            '{"output": [{"name": "Paul", "age": 68}, {"name": "Mathilda", "age": 81}]}',
            [UserDetail(name='Paul', age=68), UserDetail(name='Mathilda', age=81)])

    class UserDetail(BaseModel):
        """ Details about a user """
        name: str = Field(..., description="The name for the user")
        age: int = Field(..., description="The user's age")

    extract(list[UserDetail],
            "Jane's 99 years old, Paul is 75.",
            '{"output": [{"name": "Jane", "age": 99}, {"name": "Paul", "age": 75}]}',
            [UserDetail(name='Jane', age=99), UserDetail(name='Paul', age=75)])






def test_classify():

    classify(["yes", "no"], 
             "I will never make it on time",
             '{"output": "no"}',
             "no")

    classify(["dog", "horse", "car", "bus"], 
             "Dogs and horses and tractors and a school bus",
             '{"output": "bus"}',
             "bus")


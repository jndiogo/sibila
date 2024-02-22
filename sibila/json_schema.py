from typing import Any, Optional, Union, Literal, Annotated, get_origin, get_args
from dataclasses import dataclass, field
from enum import Enum

import json

import logging
logger = logging.getLogger(__name__)

from datetime import date, time, datetime

from pydantic import BaseModel, TypeAdapter, ValidationError

from .dictype import json_schema_from_dictype

from .utils import (
    is_subclass_of,
    synth_desc
)



@dataclass
class JSchemaConf:
    """
    Configuration for JSON schema massaging and validation.
    """

    # json_schema_massage() configuration:
    resolve_refs: bool = True
    """Set for $ref references to be resolved and replaced with actual definition."""
    
    collapse_single_combines: bool = True
    """Any single-valued "oneOf"/"anyOf" is replaced with the actual value."""
    
    description_from_title: int = 0
    """If a value doesn't have a description entry, make one from its title or name.
        0: don't
        1: copy title or name to description
        2: 1: + capitalize first letter and convert _ to space: class_label -> "class label". """
    
    force_all_required: bool = False
    """Force all entries in an object to be required (except removed defaults if remove_with_default=True)."""
    
    remove_with_default: bool = False
    """Delete any values that have a "default" annotation."""

    default_to_last: bool = True
    """Move any default value entry into the last position of properties dict."""

    additional_allowed_root_keys: list = field(default_factory=list)
    """By default only the following properties are allowed in schema's root:
         description, properties, type, required, additionalProperties, allOf, anyOf, oneOf, not
       Add to this list to allow additional root properties."""

    # ideas for more simplification:
    #   force_no_additional_properties: sets "additionalProperties": false on all type=object

    
    # pydantic_obj_from_json() configuration:
    pydantic_strict_validation: Optional[bool] = None
    """Validate JSON values in a strict manner or not. None means validate individually per each value in the obj. (for example in pydantic with: Field(strict=True))."""





# after https://github.com/pydantic/pydantic/issues/889#issuecomment-850312496
def json_schema_resolve_refs(schema: dict,
                             del_root_defs: bool = True,
                             max_tries: int = 100) -> dict:
    
    schema = schema.copy()

    root_defs = set()
    
    def replace_value_in_dict(item, original_schema):
        if isinstance(item, list):
            return [replace_value_in_dict(i, original_schema) for i in item]
            
        elif isinstance(item, dict):
            if list(item.keys()) == ['$ref']:
                definitions = item['$ref'][2:].split('/')
                root_defs.add(definitions[0])
                res = original_schema.copy()
                for definition in definitions:
                    res = res[definition]
                return res
                
            else:
                return {key: replace_value_in_dict(i, original_schema) for key, i in item.items()}
                
        else:
            return item        
    
    
    for i in range(max_tries):
        if '$ref' not in json.dumps(schema):
            break
        schema = replace_value_in_dict(schema.copy(), schema.copy())

    # remove root defs
    if del_root_defs:
        for k in root_defs:
            if k in schema.keys():
                del schema[k]
    
    return schema




def json_schema_massage(sch: dict,
                        schemaconf: Optional[JSchemaConf] = JSchemaConf(),
                        DEB: Optional[bool] = False) -> dict:

    """
    Massages JSON schema object to simplify as much as possible and remove all non-essential keys. 
    Resolves $refs and eliminates definitions.

    """
   
    # resolve $refs -> defs
    if schemaconf.resolve_refs:
        sch = json_schema_resolve_refs(sch)
        if DEB:
            from pprint import pprint
            print("resolved:")
            pprint(sch)
            
   
    # root: clean all other than:
    allowed_in_root = ["description",
                       "properties", 
                       "type", 
                       "required",
                       "additionalProperties",
                       "allOf", "anyOf", "oneOf", "not"] + schemaconf.additional_allowed_root_keys
        
    root = {
        k: v for k, v in sch.items() if k in allowed_in_root
    }
    

    
    def recurse_object_or_items(dic: dict):
        if "title" in dic:
            if schemaconf.description_from_title and "description" not in dic:
                dic["description"] = synth_desc(schemaconf.description_from_title - 1, 
                                                dic["title"])

            del dic["title"]

        clean(dic)

        
    def recurse_combine(lis: list):
        for dic in lis:
            if "title" in dic:
                if schemaconf.description_from_title and "description" not in dic:
                    dic["description"] = synth_desc(schemaconf.description_from_title - 1, 
                                                    dic["title"])
    
                del dic["title"]

            clean(dic)

    
    def clean(root: dict):

        if DEB: print("===", root.keys())
        
        if "properties" in root:
            required_keys = []

            vars_dict = root["properties"]

            # for each prop var
            kl = list(vars_dict.keys())
            if DEB: print("---properties", kl)

            # for dict fields in properties
            for k in kl:

                if schemaconf.force_all_required:
                    required_keys.append(k)
                
                default_to_last_value = None
                if "default" in vars_dict[k]:
                    if schemaconf.remove_with_default: # skip entries with default values
                        del vars_dict[k]
                        continue
                    elif schemaconf.default_to_last: # save and delete
                        default_to_last_value = [vars_dict[k]["default"]] # because default can be None
                        del vars_dict[k]["default"]
        
                recurse_object_or_items(vars_dict[k])

                if default_to_last_value is not None:
                    vars_dict[k]["default"] = default_to_last_value[0]
            
            if schemaconf.force_all_required:
                root["required"] = required_keys

        
        elif "items" in root:            
            vars_dict = root["items"]

            if DEB: print("---items", list(vars_dict.keys()))
            
            recurse_object_or_items(vars_dict)

        
        if "allOf" in root:
            which = "allOf"
        elif "anyOf" in root:
            which = "anyOf"
        elif "oneOf" in root:
            which = "oneOf"
        else:
            which = None

        if which is not None: # a combine
            if DEB: print(f"---{which}")                

            if len(root[which]) == 1 and schemaconf.collapse_single_combines:
                item = root[which][0]
                del root[which]
                root.update(item)
                recurse_object_or_items(root)
            else:
                recurse_combine(root[which])

                
    clean(root)

    return root





# ============================================================================ Pydantic - JSON schema

def json_schema_from_pydantic(cls: BaseModel) -> dict:
    return cls.model_json_schema()


def pydantic_obj_from_json(cls: BaseModel, 
                           obj_init: dict,
                           schemaconf: Optional[JSchemaConf] = JSchemaConf()
                           ) -> BaseModel:

    """
    For info on strict_validation see:
    https://docs.pydantic.dev/latest/concepts/strict_mode/

    strict_validation can be applied per class:
        class MyModel(BaseModel):
            model_config = ConfigDict(strict=False)
            
    or per Field:
        class Model(BaseModel):
            x: int = Field(strict=True)
            y: int = Field(strict=False)                

    https://docs.pydantic.dev/latest/api/type_adapter/
    """
    
    adapter = TypeAdapter(cls)
    try:
        obj = adapter.validate_python(obj_init, 
                                        strict=schemaconf.pydantic_strict_validation)
        return obj
        
    except ValidationError as e:
        raise TypeError(str(e))
            



    


# ============================================================================ JSON schema from types

def is_prim_type(type_: Any,
                 allow_bool: bool):
    if not isinstance(type_, type):
        return False
    
    # cannot use issubclass which would accept IntEnum as int
    if type_ is str or type_ is float or type_ is int:
        return True
    else:
        return allow_bool and type_ is bool




def get_type(type_: Any,
             allow_enums: bool,
             allow_BaseModel: bool,
             allow_dictype: bool) -> tuple:
    """
    type_ can be any non-list type
    prim_type
    enum (if allow_enums)
    BaseModel (if allow_BaseModel)
    dict -> dictype definition (if allow_dictype)

    prim_type: can be Annotated[T, "Desc"]
        bool
        int
        float
        str
        
    if allow_enums:
        enums: can be Annotated[T, "Desc"]
            [1, 2, 3] or ["a","b"] - all items of the same prim_type
            Literal['year', 'name'] - all items of the same prim_type
            Enum, EnumInt, EnumStr, (Enum, int),... - all items of the same prim_type

    datetime.datetime

    Returns (type_, anno_desc, enum_list) or (None,,) if not a supported simple type.    
    """

    enum_list = None
    anno_desc = None

    if get_origin(type_) is Annotated:
        args = list(get_args(type_))
        type_ = args[0]
        if len(args) > 1:
            anno_desc = args[1]

    if is_prim_type(type_, allow_bool=True):
        ...

    elif allow_BaseModel and is_subclass_of(type_, BaseModel):
        ...

    elif allow_dictype and isinstance(type_, dict):
        ...

    elif allow_enums:

        if get_origin(type_) is Literal:
            enum_list = list(get_args(type_))
            type_ = type(enum_list[0])

        elif is_subclass_of(type_, Enum):
            enum_list = [e.value for e in type_]
            type_ = type(enum_list[0])

        elif isinstance(type_, list): # enum as value list
            enum_list = type_[:]
            type_ = type(type_[0])

        else:
            type_ = None

        if type_ is not None: # enum consistency checks
            if not is_prim_type(type_, allow_bool=True):
                raise TypeError(f"Base type is not one of bool, int, float, str: '{type_}'")

            if enum_list is not None:
                if not all([type_ is type(e) for e in enum_list]):
                    raise TypeError(f"All enum values must have the same type in '{enum_list}'")

    else:
        type_ = None

    return type_, anno_desc, enum_list
    






def get_type_list(type_: Any) -> tuple:
    """
    list of values of a prim_type, BaseModel, dictype: 
        list[prim_type] - for example list[int]

    can be Annotated[list[T], "List desc"] and/or list[Annotated[T, "Item desc"]]

    Returns: item_type, item_anno_desc, item_enum_list, list_anno_desc
    """

    item_anno_desc = None
    item_enum_list = None
    list_anno_desc = None

    if get_origin(type_) is Annotated:
        args = list(get_args(type_))
        type_ = args[0]
        if len(args) > 1:
            list_anno_desc = args[1]

    if get_origin(type_) is list: # list[type]
        args = list(get_args(type_))
        type_ = args[0]

    else:
        type_ = None

    if type_ is not None:
        type_, item_anno_desc, item_enum_list = get_type(type_, 
                                                         allow_enums=True,
                                                         allow_BaseModel=True,
                                                         allow_dictype=True)

        if type_ is None: # list type consistency
            raise TypeError(f"List item type is not bool, int, float, str, enum, BaseModel or a dictype definition: '{type_}'")

    return type_, item_anno_desc, item_enum_list, list_anno_desc

    





def build_type_json_schema(type_: Any, 
                           desc: Optional[str] = None,
                           # for prim_type or enum
                           enum_list: Optional[list] = None,                                  
                           default: Optional[Any] = None,
                           format: Optional[str] = None) -> dict:
    """Render a valid JSON Schema specification for an accepted type.
    type_ can be any non-list type
    prim_type
    enum
    BaseModel
    dict -> dictype definition

    @TODO: update to accepted and args

    Args:
        type_: Above supported types.
        desc: Optional description for field. If given, will override any Annotated type's description.
        default: _description_. Defaults to None.
        format: _description_. Defaults to None.

    Returns:
        A dict whose json.dumps() serialization is a valid JSON Schema specification for type.
    """

    out_type, anno_desc, enum_list2 = get_type(type_, 
                                               allow_enums=True,
                                               allow_BaseModel=True,
                                               allow_dictype=True)
    if out_type is None:
        raise TypeError(f"Unsupported type: '{type(type_)}'")
    
    if desc is None:
        desc = anno_desc

    if is_subclass_of(out_type, BaseModel):
        out = out_type.model_json_schema()

        if desc is not None:
            out["description"] = desc

    elif isinstance(out_type, dict): # dictype
        out = json_schema_from_dictype(out_type, desc)

    else: # prim_type or enum

        out = {}

        if desc:
            out["description"] = desc

        if enum_list is None:
            enum_list = enum_list2
        if enum_list is not None:
            out["enum"] = enum_list

        if format is not None:
            if out_type is not str:
                raise TypeError("Arg format is only valid for str type.")

            out["format"] = format

        if default is not None:
            if not isinstance(default, out_type):
                raise TypeError(f"Arg default is not of type {out_type}.")

            out["default"] = default

        out["type"] = get_json_type(out_type)

    return out



def build_array_type_json_schema(item_type: Any,
                                 item_desc: Optional[str] = None,
                                 item_enum_list: Optional[list] = None,
                                 list_desc: Optional[str] = None) -> dict:
    """_summary_

    Args:
        item_type: The raw item type without annotations

    Returns:
        _description_
    """

    items_repr = build_type_json_schema(item_type, 
                                        item_desc,
                                        item_enum_list)

    out = {}

    if list_desc:
        out["description"] = list_desc

    out["items"] = items_repr

    out["type"] = "array"

    return out



def build_object_type_json_schema(properties_repr: dict,
                                  desc: Optional[str] = None,
                                  required_keys: Optional[list[str]] = None) -> dict:
    """_summary_

    Args:
        properties_repr: A dict with "name": repr.
        desc: _description_. Defaults to None.
        required_keys: _description_. Defaults to None meaning all keys are required.

    Returns:
        _description_
    """


    out = {}

    if desc:
        out["description"] = desc

    out["properties"] = properties_repr

    all_keys = list(properties_repr.keys())
    if required_keys is None:
        required_keys = all_keys
    else:
        if not all([k in all_keys for k in required_keys]):
            raise ValueError("Arg required_keys has unknown properties keys")

    out["required"] = required_keys        

    out["type"] = "object"

    return out




def build_any_json_schema(type_: Any, 
                          output_key_name: str) -> tuple[dict,bool]:
    """_summary_

    Args:
        type_: _description_
        output_key_name: _description_

    Raises:
        TypeError: _description_

    Returns:
        JSON schema dict, create_output_key
    """

    # type list
    item_type, item_desc, item_enum_list, list_desc = get_type_list(type_)    
    if item_type is not None:            
        # build json schema for list of type_
        array_repr = build_array_type_json_schema(item_type, 
                                                  item_desc, 
                                                  item_enum_list,
                                                  list_desc)

        schema = build_object_type_json_schema({output_key_name: array_repr})

        create_output_key = True

    else: # prim, enum, BaseModel
        type_, desc, enum_list = get_type(type_, 
                                          allow_enums=True,
                                          allow_BaseModel=True,
                                          allow_dictype=True)
        if type_ is not None:
            schema = build_type_json_schema(type_, 
                                            desc,
                                            enum_list)
            create_output_key = schema.get("type") != "object"
            
            if create_output_key:
                schema = build_object_type_json_schema({output_key_name: schema})

        else:
            raise TypeError(f"Unknown target type '{type_}'")


    return schema, create_output_key


def get_final_type(type_: Any) -> Any:

    """ Final type can be:
    BaseModel derived
    Enum derived    
    prim_type (simple type or type of items in Literal or list["a","b"])
    """

    # dig through list and annotations until a valid type is found

    is_list = False

    while True:
        orig = get_origin(type_)
        if orig in (Annotated, list): # get rid of annotations and list[]
            is_list = is_list or orig == list

            args = list(get_args(type_))
            type_ = args[0]
            continue

        if is_subclass_of(type_, BaseModel):
            break
        
        elif is_subclass_of(type_, Enum):
            enum_list = [e.value for e in type_]
            type_ = type(enum_list[0])
            break

        elif get_origin(type_) is Literal:
            enum_list = list(get_args(type_))
            type_ = type(enum_list[0])
            break

        elif isinstance(type_, list): # enum list: ["a", "b"]
            type_ = type(type_[0])
            break
        
        elif (type_ is str or 
              type_ is float or 
              type_ is int or 
              type_ is bool):
            break
        
        else:
            raise TypeError(f"Unknown final type: '{type_}'")

    return type_, is_list





def create_final_instance(type_: Any,
                          is_list: bool,
                          val: Any,
                          schemaconf: JSchemaConf) -> Any:
    

    def create_item(type_: Any,
                    val: Any) -> Any:
        
        if is_subclass_of(type_, BaseModel):
            obj = pydantic_obj_from_json(type_, 
                                         val,
                                         schemaconf=schemaconf)
            return obj
        
        elif (type_ is str or 
              type_ is float or 
              type_ is int or 
              type_ is bool):
            value = type_(val) 
            return value
        
        else:
            raise TypeError(f"Unexpected value of type '{type_}' for value '{val}'")


    if is_list:
        if type(val) is not list: # check, just in case
            raise TypeError(f"Value is not a list: '{val}'")
        
        out = []
        for item_val in val:
            out.append(create_item(type_, item_val))

        return out

    else:
        return create_item(type_, val)











def get_json_type(t: Any) -> str:
    JSON_TYPE_FROM_PY_TYPE = {
        str: "string",
        float: "number",            
        int: "integer",            
        bool: "boolean"
    }
    if t not in JSON_TYPE_FROM_PY_TYPE:
        raise TypeError(f"Unknown type '{t}'")
    
    return JSON_TYPE_FROM_PY_TYPE[t]



# Useful:
# Online JSON schema validator: https://www.jsonschemavalidator.net/
# JSON pretty formatter: https://jsonformatter.org/json-pretty-print

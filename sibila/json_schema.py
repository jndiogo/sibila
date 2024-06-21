"""JSON Schema generator functions"""

from typing import Any, Optional, Union, Literal, Annotated, get_origin, get_args
from enum import Enum

from copy import copy, deepcopy

import json

import logging
logger = logging.getLogger(__name__)

from datetime import date, time, datetime

from dataclasses import dataclass, is_dataclass, field, fields, asdict, MISSING

from pydantic import BaseModel, TypeAdapter, ValidationError

from .utils import (
    is_subclass_of,
    synth_desc
)



@dataclass
class JSchemaConf:
    """Configuration for JSON schema massaging and validation."""

    # json_schema_massage() configuration:
    resolve_refs: bool = True
    """Set for $ref references to be resolved and replaced with actual definition."""
    
    collapse_single_combines: bool = True
    """Any single-valued "oneOf"/"anyOf" is replaced with the actual value."""
    
    description_from_title: int = 0
    """If a value doesn't have a description entry, make one from its title or name.

    - 0: don't make description from name
    - 1: copy title or name to description
    - 2: 1: + capitalize first letter and convert _ to space: class_label -> "class label". """
    
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

    
    def clone(self):
        """Return a copy of this configuration."""
        return copy(self)







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
                        schemaconf: JSchemaConf = JSchemaConf(),
                        DEB: bool = False) -> dict:

    """
    Massages JSON schema object to simplify as much as possible and remove all non-essential keys.
    Resolves $refs and eliminates definitions.
    """
   
    sch = deepcopy(sch)

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




# ============================================================================ Pydantic BaseModel

def json_schema_from_pydantic(cls: BaseModel) -> dict:
    return cls.model_json_schema()


def pydantic_obj_from_json(cls: BaseModel, 
                           obj_init: dict,
                           schemaconf: Optional[JSchemaConf] = None
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
    
    if schemaconf is None:
        schemaconf = JSchemaConf()

    adapter = TypeAdapter(cls)
    try:
        obj = adapter.validate_python(obj_init, 
                                      strict=schemaconf.pydantic_strict_validation)
        return obj
        
    except ValidationError as e:
        raise TypeError(str(e))
            



    


# ============================================================================ JSON Schema construction from types

def is_prim_type(type_: Any):
    
    # cannot use issubclass which would accept IntEnum as int
    return (type_ is str or 
            type_ is float or 
            type_ is int or 
            type_ is bool)





def get_type(type_: Any) -> tuple:
    """Identifies a simple type, that can be generated by model.
    Argument type_ can be one of the following non-list types:

    - prim_type:

        - bool
        - int
        - float
        - str

    - typing.Union
        
    - enums:

        - [1, 2, 3] or ["a","b"] - all items of the same prim_type
        - Literal['year', 'name'] - all items of the same prim_type
        - Enum, EnumInt, EnumStr, (Enum, int),... - all items of the same prim_type

    - datetime/date/time

    - a list in the form:
        - list[type]
        
        For example list[int]. The list can be annotated:
            Annotated[list[T], "List desc"]
        And/or the list item type can be annotated:
            list[Annotated[T, "Item desc"]]

    - dataclass with fields of the above supported types (or dataclass).

    - Pydantic BaseModel

    All types can be Annotated[T, "Desc"], for example: 
        count: int
    Can be annotated as:
        count: Annotated[int, "How many units?"]

    Args:
        type_: Type to be simplified, one of the above.

    Returns:
        Returns (type_, anno_desc, options) or (None,,) if not a supported type.
        options: optional keys
            "str_format": str
            "enum_list": []
    """

    anno_desc = None
    options = {}

    if get_origin(type_) is Annotated:
        args = list(get_args(type_))
        type_ = args[0]
        if len(args) > 1:
            anno_desc = args[1]


    if (is_prim_type(type_) or
        is_dataclass(type_) or
        is_subclass_of(type_, BaseModel) or
        get_origin(type_) is Union):
        ...

    elif type_ is type(None):
        ...

    elif is_subclass_of(type_, datetime):
        type_ = str
        options["str_format"] = "date-time"

    elif is_subclass_of(type_, date):
        type_ = str
        options["str_format"] = "date"

    elif is_subclass_of(type_, time):
        type_ = str
        options["str_format"] = "time"

    else: # enums
        type_, enum_list = get_enum_type(type_)

        if type_ is not None:
            options["enum_list"] = enum_list

    return type_, anno_desc, options
    



def get_enum_type(type_: Any) -> tuple:
    """Identifies an enumeration that can be generated by model. 

    Args:
        type_: Can be one of the following forms:

            - [1, 2, 3] or ["a","b"] - all items of the same prim_type
            - Literal['year', 'name'] - all items of the same prim_type
            - Enum, EnumInt, EnumStr, (Enum, int),... - all items of the same prim_type

    Raises:
        TypeError: If not a valid enumeration.

    Returns:
        A tuple in the form enum_type, list_of_enum_values.
        Where enum_type if a value accepted by get_type().
    """

    enum_list = None

    if get_origin(type_) is Literal:
        enum_list = list(get_args(type_))
        type_ = type(enum_list[0])

    elif is_subclass_of(type_, Enum):
        enum_list = [e.value for e in type_]
        type_ = type(enum_list[0])

    elif isinstance(type_, list): # enum as value list: ["a","b"]
        enum_list = type_[:]
        type_ = type(type_[0])

    else:
        type_ = None

    if type_ is not None: # enum consistency checks
        if not is_prim_type(type_):
            raise TypeError(f"Base type is not one of bool, int, float, str: '{type_}'")

        if not enum_list:
            raise TypeError("Enum must have at least one value")

        if not all([type_ is type(e) for e in enum_list]):
            raise TypeError(f"All enum values must have the same type in '{enum_list}'")
        
    return type_, enum_list




def get_type_list(type_: Any) -> tuple:
    """Identify a type which is a list of types accepted by get_type().

    Args:
        type_: A type in the form: 
            - list[type]

                For example list[int]. The list can be annotated:
                    Annotated[list[T], "List desc"]
                And/or the list item type can be annotated:
                    list[Annotated[T, "Item desc"]]

    Raises:
        TypeError: If not a valid list.

    Returns:
        A tuple in the form: item_type, item_anno_desc, item_options, list_anno_desc.
    """

    item_anno_desc = None
    item_options = {}
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
        type_, item_anno_desc, item_options = get_type(type_)

        if type_ is None: # list type consistency
            raise TypeError(f"List item type is not a supported type like bool, int, float, str, enum, dataclass, datetime, BaseModel: '{type_}'")

    return type_, item_anno_desc, item_options, list_anno_desc

    





def build_type_json_schema(type_: Any, 
                           desc: Optional[str] = None,
                           options: dict = {},
                           default: Optional[Any] = MISSING) -> dict:
    """Render a JSON Schema specification for an accepted simple type.

    Args:
        type_: A type supported by get_type().
        desc: Optional description for field. Defaults to None, but if given, will override any Annotated type's description.
        options: Type options for specifying enums, str_format. Defaults to {}.
        default: A default value for field. Defaults to MISSING.

    Returns:
        A dict whose json.dumps() serialization is a valid JSON Schema specification for the given type_ arg.
    """

    out_type, anno_desc, options2 = get_type(type_)
    if out_type is None:
        raise TypeError(f"Unsupported type: '{type(type_)}'")
    
    if is_dataclass(out_type):
        out = build_dataclass_object_json_schema(out_type)

    elif is_subclass_of(out_type, BaseModel):
        out = out_type.model_json_schema()

    elif get_origin(out_type) is Union:
        args = list(get_args(out_type))
        out_json = [build_type_json_schema(t) for t in args]
        out = {"anyOf": out_json} 

    else: # prim_type or enum or type(None)
        out = {}

        enum_list = options.get("enum_list")
        if enum_list is None:
            enum_list = options2.get("enum_list")
        if enum_list is not None:
            out["enum"] = enum_list

        str_format = options.get("str_format")
        if str_format is None:
            str_format = options2.get("str_format")
        if str_format is not None:
            if out_type is not str:
                raise TypeError("Arg str_format is only valid for str type.")

            out["format"] = str_format

        out["type"] = get_json_type(out_type)

    desc = desc or anno_desc
    if desc is not None:
        out["description"] = str(desc) # ensure string

    if default is not MISSING:
        # @TODO: better checking for Union origin types, deal with Literals, for example
        if (get_origin(out_type) is not Union and
            not isinstance(default, out_type) and
            not isinstance(default, dict)):
            raise TypeError(f"Arg default is not of type {out_type} nor dict.")
        
        out["default"] = default

    return out



def build_array_json_schema(item_type: Any,
                            item_desc: Optional[str] = None,
                            item_options: dict = {},
                            list_desc: Optional[str] = None,
                            default: Optional[Any] = MISSING) -> dict:
    """Render a JSON Schema specification for an list/array.

    Args:
        item_type: Type for array items.
        item_desc: Optional description for an array item. Defaults to None, but if given, will override any Annotated type's description.
        item_options: Optional array item options. Defaults to {}.
        list_desc: Optional description for the entire array. Defaults to None.
        default: Optional default value. Defaults to MISSING.

    Returns:
        A dict whose json.dumps() serialization is a valid JSON Schema specification for the given array.
    """

    items_repr = build_type_json_schema(item_type, 
                                        item_desc,
                                        options=item_options)

    out: dict[str,Any] = {}

    if list_desc:
        out["description"] = str(list_desc) # ensure string

    out["items"] = items_repr

    out["type"] = "array"

    if default is not MISSING:
        out["default"] = default

    return out








def build_type_or_array_json_schema(type_: Any,
                                    default: Optional[Any] = MISSING) -> tuple[dict,bool]:
    """Build a JSON schema for given simple type or array. See build_*() functions for details."""

    # type list
    item_type, item_desc, item_options, list_desc = get_type_list(type_)

    if item_type is not None:            
        # build json schema for list of type_
        schema = build_array_json_schema(item_type, 
                                         item_desc, 
                                         item_options,
                                         list_desc,
                                         default)
        is_object = False

    else: # prim, enum, datetime, dataclass, BaseModel
        out_type, desc, options = get_type(type_)
        if out_type is None:
            raise TypeError(f"Unknown target type '{type_}'")

        schema = build_type_json_schema(out_type, 
                                        desc,
                                        options=options,
                                        default=default)
        
        is_object = schema.get("type") == "object"

    return schema, is_object










def build_object_json_schema(properties_repr: dict,
                             desc: Optional[str] = None,
                             required_keys: Optional[list[str]] = None) -> dict:
    """Build a JSON schema for an object type."""

    out: dict[str,Any] = {}

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


def get_from_default_factory(obj: Any) -> Any:

    if is_dataclass(obj) and not isinstance(obj, type): # dataclass obj
        return asdict(obj)
    elif isinstance(obj, BaseModel):
        return dict(obj)
    elif isinstance(obj, list) or is_prim_type(type(obj)):
        return obj
    
    return None
    



def build_dataclass_object_json_schema(type_: Any) -> dict:
    """Build a JSON schema for a dataclass type."""

    desc = type_.__doc__
    if desc.startswith(type_.__name__ + "(") and desc[-1] == ")": # ignore automatic doc from dataclass params
        desc = None        

    props = {}
    required = []

    flds = fields(type_)
    for fl in flds:
        name = fl.name
        prop_type = fl.type # can be Annotated[] -> desc
        if fl.default is MISSING:
            if fl.default_factory is not MISSING:
                default = get_from_default_factory(fl.default_factory())
            else: # not default means required
                default = MISSING
                required.append(name)
        else:
            default = fl.default

        prop_schema, _ = build_type_or_array_json_schema(prop_type,
                                                         default=default)

        props[name] = prop_schema

    if not required: # we can't have an empty dataclass: make all required
        required = list(props.keys())

    out = build_object_json_schema(props,
                                   desc,
                                   required)

    return out






def build_root_json_schema(type_: Any, 
                           output_key_name: str) -> tuple[dict,bool]:
    """Build a JSON schema for a type: simple type, list, array, dataclass or Pydantic BaseModel."""

    schema, is_object = build_type_or_array_json_schema(type_)

    if not is_object: # need to create an object in root
        schema = build_object_json_schema({output_key_name: schema})

    return schema, not is_object








def get_final_type(type_: Any) -> Any:
    """Final type is the actual value type that will be instantiated with the model's response."""

    is_list = False

    # dig through list and annotations until a valid type is found
    while True:
        orig = get_origin(type_)
        if orig in (Annotated, list): # get rid of annotations and list[]
            is_list = is_list or orig == list

            args = list(get_args(type_))
            type_ = args[0]
            continue


        if (is_dataclass(type_) or
            is_subclass_of(type_, BaseModel) or
            is_subclass_of(type_, datetime) or
            is_subclass_of(type_, date) or
            is_subclass_of(type_, time)):
            break
        
        elif isinstance(type_, list): # enum list: ["a", "b"]
            type_ = type(type_[0])
            break
        
        elif get_origin(type_) is Literal: # Literals resolve to their prim_types
            enum_list = list(get_args(type_))
            type_ = type(enum_list[0])
            break

        elif is_subclass_of(type_, Enum): # Enum types resolve to themselves
            break

        elif is_prim_type(type_):
            break
        
        else:
            raise TypeError(f"Unknown final type: '{type_}'")

    return type_, is_list





def create_final_instance(type_: Any,
                          is_list: bool,
                          val: Any,
                          schemaconf: Optional[JSchemaConf] = None) -> Any:
    
    """Instantiate an object with the model's answer."""
    
    if schemaconf is None:
        schemaconf = JSchemaConf()

    def create_item(type_: Any,
                    val: Any) -> Any:

        if is_dataclass(type_):
            if not isinstance(val, dict):
                raise TypeError(f"Expecting dict to initialize dataclass but got: '{val}'")
            obj = type_(**val)
            return obj

        elif is_subclass_of(type_, BaseModel):
            if not isinstance(val, dict):
                raise TypeError(f"Expecting dict to initialize BaseModel but got: '{val}'")
            obj = pydantic_obj_from_json(type_, 
                                         val,
                                         schemaconf=schemaconf)
            return obj
        
        elif is_subclass_of(type_, datetime):
            if not isinstance(val, str):
                raise TypeError(f"Expecting str to initialize datetime but got: '{val}'")
            return datetime.fromisoformat(val)

        elif is_subclass_of(type_, date):
            if not isinstance(val, str):
                raise TypeError(f"Expecting str to initialize date but got: '{val}'")
            return date.fromisoformat(val)

        elif is_subclass_of(type_, time):
            if not isinstance(val, str):
                raise TypeError(f"Expecting str to initialize time but got: '{val}'")
            return time.fromisoformat(val)

        elif is_subclass_of(type_, Enum):
            return type_(val)
        
        # Literals also resolve to their prim_types        
        elif is_prim_type(type_):
            value = type_(val) 
            return value
        
        else:
            raise TypeError(f"Unexpected value of type '{type_}' for value '{val}'")


    if is_list:
        if type(val) is not list: # check, just in case...
            raise TypeError(f"Value is not a list: '{val}'")
        
        out = []
        for item_val in val:
            out.append(create_item(type_, item_val))

        return out

    else:
        return create_item(type_, val)








def get_json_type(t: Any) -> str:
    """Translate from native types to JSON's."""
    JSON_TYPE_FROM_PY_TYPE = {
        str: "string",
        float: "number",            
        int: "integer",            
        bool: "boolean",
        type(None): "null",
    }
    if t not in JSON_TYPE_FROM_PY_TYPE:
        raise TypeError(f"Unknown type '{t}'")
    
    return JSON_TYPE_FROM_PY_TYPE[t]



# Useful:
# string formats: https://json-schema.org/understanding-json-schema/reference/string#built-in-formats
# Online JSON schema validator: https://jsonschema.dev/
# JSON pretty formatter: https://jsonformatter.org/json-pretty-print

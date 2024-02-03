"""A simple way to define typed fields in a dict. Kind of a simpler and convenient pydantic for dicts.

A dictype is defined in python dict, where each entry represents a typed data field with an optional description.

For example:
```
dictype = {
    "summary": { "type": str, "desc": "A general summary of stuff" },
    "names": { "type": [str], "desc": "List of stuff names" },
    "cost": { "type": float, "desc": "Cost of stuff", "optional": True },
    "kind": { "type": ["Open", "Closed"], "desc": "The kind of stuff" },
    
    # For lists, desc can be split in list_description|item_description:
    "colors": { "type": [str], "desc": "Color list for stuff|Color names" },

    # Or desc can be a list, first item for the list, second for its items: 
    "colors2": { "type": [str], "desc": ["Color list for stuff", "Color names"] },
}
```    

# Allowed types
```
Primitive types:
    bool
    int
    float
    str
```

```
Enums: Represented by a list of values (only primitive types).
    For example:

    "kind": { "type": ["Open", "Closed"], "desc": "The kind of stuff" },
```

```
Lists: A list of an allowed type.
    For example to define an str list (note the [] around str):

    "names": { "type": [str], "desc": "List of stuff names" },
```

```
Dicts: A field can also be a dict, which allows for composing hierarchies.
    For example, person_type is first defined and then used below:

    person_type = {
        "name": { "type": str, "desc": "Name of person" },
        "occupation": { "type": str, "desc": "Person's occupation" },
    }
    
    team_type = {
        "chief": {"type": person_type, "desc": "Team's chief"},
        "members": {"type": [person_type], "desc": "List of team members"}
    }
```

# Alternative shorter notation
```
Fields can also be specified as an ordered list with the following entries:
    type, desc, "optional"/True

    For example:

    dictype = {
        "summary": [str, "A general summary of stuff"],
        "names": [ [str], "List of stuff names" ],
        "cost": [ float, "Cost of stuff", "optional" ],
        "kind": [ ["Open", "Closed"], "The kind of stuff" ],
        
        # For lists, desc can be split in two:
        "colors": [ [str], "Color list for stuff|Color names" ],
        "colors2": [ [str], ["Color list for stuff", "Color names"] ],
    }

    The order in the list is: type, dec, "optional/True
``` 


"""

from typing import Any, Optional, Union


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


def safe_string(text: str) -> str:
    text = text.replace("\n", "\\n")
    return text




def make_prop(prop_name: str, 
              type_name: str, # JSON schema type name
              desc: Optional[str] = None,
              items: Optional[dict] = None,
              enum: Optional[list[Any]] = None,
              desc_from_title: int = 0) -> dict:
    """
    description_from_prop_name: 2: capitalize first letter and convert _ to space
    """
    pd = {}
    if desc is not None:
        pd["description"] = safe_string(desc)
        
    elif desc_from_title: # synth desc from prop name
        desc = prop_name
        if desc_from_title == 2:
            desc = desc.replace("_", " ").capitalize()
        pd["description"] = desc

    if items is not None:
        assert enum is None, "Only one of items or enum"
        pd["items"] = items
    if enum is not None:
        assert items is None, "Only one of items or enum"
        pd["enum"] = enum
        
    pd["type"] = type_name

    return pd


def make_enum(prop_name: str, 
              ftype: Any,                  
              desc: Optional[str] = None,
              desc_from_title: int = 0
              ):
    
    # check all types equal
    # flake8 ignores: E721 do not compare types, for exact checks use `is` / `is not`, for instance checks use `isinstance()`

    type_ftype0 = type(ftype[0])
    if not all([type(f) is type_ftype0 for f in ftype]):
        raise TypeError(f"All enum values must be of the same type: '{prop_name}': {type_ftype0}")
        
    type_name = get_json_type(type_ftype0)
    out = make_prop(prop_name, type_name, desc, enum=ftype,
                    desc_from_title=desc_from_title)

    return out



def make_prim(ftype: Any,
              name: str,
              desc: Union[str,None],
              desc_from_title: int = 0
              ) -> dict:

    if isinstance(ftype, dict):
        out = json_schema_from_dictype(ftype, desc)
    else:
        type_name = get_json_type(ftype)
        out = make_prop(name, type_name, desc, 
                        desc_from_title=desc_from_title)
        
    return out




def json_schema_from_dictype(dictype: dict,
                             desc: Optional[str] = None,
                             desc_from_title: int = 0) -> dict:
    """Create a JSON schema representation of a dictype.

    Args:
        dictype: The dictype.
        desc: Top-level description of the dictype. Defaults to None.
        desc_from_title: Should title be used as description for fields that don't have one?. Values: 0=No, 1=Yes, 2=Yes + capitalize first letter and convert _ to space. Defaults to 0.

    Raises:
        TypeError: If dictype inconsistencies found.

    Returns:
        A dict with the created JSON schema.
    """

    out = {}
    
    if desc is not None:
        out["description"] = desc

    props = out["properties"] = {} 

    required_names = []
    
    for index, (name, field_) in enumerate(dictype.items()):

        # extract field's keys
        if isinstance(field_, dict):
            if "type" not in field_:
                raise TypeError(f"All fields must include a type key at '{name}': {field_}")
            ftype = field_["type"]
        
            fdesc = field_.get("desc") or field_.get("description")
            foptional = field_.get("optional")
            
        elif isinstance(field_, list):
            if len(field_) < 1:
                raise TypeError(f"Fields specified as lists must at least have a type entry at '{name}': {field_}")
            ftype = field_[0]

            if len(field_) >= 2:
                fdesc = field_[1]
            else:
                fdesc = None
            
            if len(field_) >= 3:
                foptional = field_[2]
                
                if len(field_) > 3:
                    raise TypeError(f"Fields specified as lists must have 3 entries at most (type, desc, optional) at '{name}': {field_}")                 
            else:
                foptional = None
                
        else:
            raise TypeError(f"Fields can only be specified as dicts or lists at '{name}': {field_}")                             

        # verify field values
        if fdesc is not None:
            # extract "list desc" | "item desc"
            if isinstance(fdesc, str):
                fdesc = fdesc.split('|')
                if len(fdesc) == 1:
                    fdesc.append(None)
            elif isinstance(fdesc, list):
                raise TypeError(f"Field's desc can only be list or str (divide str with '|' char) at '{name}': {field_}")
                
            if len(fdesc) > 2:
                raise TypeError(f"Fields's desc can only have up to 2 strings at '{name}': {field_}")
        else:
            fdesc = [None,None]
                
        if foptional is not None:
            if isinstance(foptional, str):
                foptional = foptional.lower()
                foptional = (foptional == "optional") or (foptional == "true")
            elif not isinstance(foptional, bool):
                raise TypeError(f"Field's optional can only be bool or str at '{name}': {field_}")
        else:
            foptional = False

        
        # now emit
        if isinstance(ftype, list): # list=[type] or [enum1,enum2]

            if len(ftype) == 1: # a list of type or of enum

                if isinstance(ftype[0], list): # list of enum
                    items = make_enum('', ftype[0], fdesc[1],
                                      desc_from_title=desc_from_title)
                    
                else: # list of type
                    items = make_prim(ftype[0], '', fdesc[1],
                                      desc_from_title=desc_from_title)
                    
                props[name] = make_prop(name, "array", fdesc[0], items=items,
                                        desc_from_title=desc_from_title)
            
            else: # list of enum
                if fdesc[1] is not None:
                    raise TypeError(f"Enums must have single description at '{name}': {field_}")
                    
                props[name] = make_enum(name, ftype, fdesc[0],
                                        desc_from_title=desc_from_title)
               
        else: # a primitive type
            if fdesc[1] is not None:
                raise TypeError(f"Primitive types must have single description at '{name}': {field_}")
            
            props[name] = make_prim(ftype, name, fdesc[0],
                                    desc_from_title=desc_from_title)

        
        if not foptional:
            required_names.append(name)

    
    if len(required_names):
        out["required"] = required_names

    out["type"] = "object"
    
    return out



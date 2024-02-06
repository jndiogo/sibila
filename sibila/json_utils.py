from typing import Any, Optional, Union
from dataclasses import dataclass, field

import json

import logging
logger = logging.getLogger(__name__)

from .dictype import json_schema_from_dictype




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
    
    description_from_title: bool = False
    """If a value doesn't have a description entry, make one from its title or name."""
    
    force_all_required: bool = False
    """Force all entries in an object to be required (except removed defaults if remove_with_default=True)."""
    
    remove_with_default: bool = False
    """Delete any values that have a "default" annotation."""

    default_to_last: bool = True
    """Move any default value entry into the last position of properties dict."""

    additional_allowed_root_keys: list = field(default_factory=list)
    """By default only "properties", "type", "required", "additionalProperties", "allOf", "anyOf", "oneOf", "not" are allowed in root - add to this setting for aditional ones."""

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

    @TODO: description_from_title could be an int, that when = 2 would transform title="ClassLabels" or "class_labels" into description="Class labels"
    """

   
    # resolve $refs -> defs
    if schemaconf.resolve_refs:
        sch = json_schema_resolve_refs(sch)
        if DEB:
            from pprint import pprint
            print("resolved:")
            pprint(sch)
            
   
    # root: clean all other than:
    allowed_in_root = ["properties", 
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
                dic["description"] = dic["title"]

            del dic["title"]

        clean(dic)
        
    def recurse_combine(lis: list):
        for dic in lis:
            if "title" in dic:
                if schemaconf.description_from_title and "description" not in dic:
                    dic["description"] = dic["title"]
    
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
try:
    from pydantic import BaseModel, TypeAdapter, ValidationError
    has_pydantic = True
except ImportError:
    has_pydantic = False
                                   
def pydantic_get_class_info(p: Any, # BaseModel
                          
                            default_name: Any = None,
                            default_description: Any = None,
                          
                            schemaconf: Optional[JSchemaConf] = JSchemaConf(),
                            DEB: Optional[bool] = False
                            ) -> tuple[str,str,dict]:

    """ Accepts a BaseModel object or a JSON schema dict.
    Returns name, description, json schema parameters from a pydantic BaseModel object """
   
    if not has_pydantic:
        raise Exception("Please install pydantic by running: pip install pydantic")

    if has_pydantic and issubclass(p, BaseModel):
        dic = p.model_json_schema()
    else:
        raise ValueError("Only pydantic BaseModel allowed for param p")

    if "title" in dic:
        name = dic["title"]
    else:
        name = default_name

    if "description" in dic:
        description = dic["description"]
    else:
        description = default_description

    parameters = json_schema_massage(dic,
                                     schemaconf,
                                     DEB)
    
    return name, description, parameters    



def pydantic_get_class_parameters(p: Any, # BaseModel
                                      
                                  schemaconf: Optional[JSchemaConf] = JSchemaConf(),
                                     
                                  DEB: Optional[bool] = False                         
                                  ) -> tuple[str,str,dict]:

    _,_, parameters = pydantic_get_class_info(p,
                                              None, None,
                                             
                                              schemaconf,
                                              DEB
                                              )

    return parameters


def pydantic_obj_from_json(cls, 
                           obj_init: dict,
                           schemaconf: Optional[JSchemaConf] = JSchemaConf()
                           ) -> object:

    if has_pydantic:
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
            
    else:
        raise Exception("Please install pydantic by running: pip install pydantic")





    
# ============================================================================ Dictype - JSON schema

def dictype_get_json_schema(dictype: dict,
                                      
                            schemaconf: Optional[JSchemaConf] = JSchemaConf(),
                                     
                            DEB: Optional[bool] = False                         
                            ) -> dict:

    out = json_schema_from_dictype(dictype, 
                                   desc_from_title=2 if schemaconf.description_from_title else 0)

    # is this needed at all on the extremelly clean output of json_schema_from_dictype()?
    schema = json_schema_massage(out,
                                 schemaconf,
                                 DEB)

    return schema

















# ============================================================================ Grammars

"""
llama.cpp json-schema to grammar converter adapted from llama_cpp_python:
https://github.com/abetlen/llama-cpp-python

Originally from llama.cpp:
https://github.com/ggerganov/llama.cpp/blob/master/examples/json-schema-to-grammar.py
https://github.com/ggerganov/llama.cpp/tree/master/grammars
"""

import re

# whitespace is constrained to a single space char to prevent model "running away" in
# whitespace. Also maybe improves generation quality?
SPACE_RULE = '" "?'

PRIMITIVE_RULES = {
    "boolean": '("true" | "false") space',
    "number": '("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? space',
    "integer": '("-"? ([0-9] | [1-9] [0-9]*)) space',
    "string": r""" "\"" (
        [^"\\\n] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\"" space """,
    "null": '"null" space',
}

INVALID_RULE_CHARS_RE = re.compile(r"[^a-zA-Z0-9-]+")
GRAMMAR_LITERAL_ESCAPE_RE = re.compile(r'[\r\n"]')
GRAMMAR_LITERAL_ESCAPES = {"\r": "\\r", "\n": "\\n", '"': '\\"'}


class SchemaConverter:
    def __init__(self, 
                 prop_order):
        self._prop_order = prop_order
        self._rules = {"space": SPACE_RULE}
        self._defs: dict[str, Any] = {}

    def _format_literal(self, 
                        literal: str):
        escaped: str = GRAMMAR_LITERAL_ESCAPE_RE.sub(
            lambda m: GRAMMAR_LITERAL_ESCAPES.get(m.group(0)), json.dumps(literal)
        )
        return f'"{escaped}"'

    def _add_rule(self, 
                  name: str, 
                  rule: str):
        esc_name = INVALID_RULE_CHARS_RE.sub("-", name)
        if esc_name not in self._rules or self._rules[esc_name] == rule:
            key = esc_name
        else:
            i = 0
            while f"{esc_name}{i}" in self._rules:
                i += 1
            key = f"{esc_name}{i}"
        self._rules[key] = rule
        return key

    def visit(self, 
              schema: dict[str, Any], 
              name: str) -> str:
        
        rule_name = name or "root"

        if "$defs" in schema:
            # add defs to self._defs for later inlining
            for def_name, def_schema in schema["$defs"].items():
                self._defs[def_name] = def_schema

        if "oneOf" in schema or "anyOf" in schema:
            rule = " | ".join(
                (
                    self.visit(alt_schema, f'{name}{"-" if name else ""}{i}')
                    for i, alt_schema in enumerate(
                        schema.get("oneOf") or schema["anyOf"]
                    )
                )
            )
            return self._add_rule(rule_name, rule)

        elif "const" in schema:
            return self._add_rule(rule_name, self._format_literal(schema["const"]))

        elif "enum" in schema:
            rule = " | ".join((self._format_literal(v) for v in schema["enum"]))
            return self._add_rule(rule_name, rule)

        elif "$ref" in schema:
            ref = schema["$ref"]
            assert ref.startswith("#/$defs/"), f"Unrecognized schema: {schema}"
            # inline $defs
            def_name = ref[len("#/$defs/"):]
            def_schema = self._defs[def_name]
            return self.visit(def_schema, f'{name}{"-" if name else ""}{def_name}')


        schema_type: Optional[str] = schema.get("type") # type: ignore
        assert isinstance(schema_type, str), f"Unrecognized schema: {schema}"

        if schema_type == "object" and "properties" in schema:
            
            if len(self._prop_order):
                prop_pairs = dict(sorted(
                    schema["properties"].items(),
                    # sort by position in prop_order (if specified) then by key
                    key=lambda kv: (self._prop_order.get(kv[0], len(self._prop_order)), kv[0]),
                ))
            else:
                prop_pairs = schema["properties"]

            
            # split names into required, not_required
            required: str = schema.get("required") or []
            not_required = [n for n in prop_pairs if n not in required]
            
            if len(required) == 0: # force all to be required: or the leading comma for not_required may cause broken JSON
                logger.debug(f"Rule '{rule_name}': GBNF grammar cannot parse rule with only optional items: making all items required")
                required = not_required
                not_required = []

            
            def emit_prop(prop_name: str, 
                          prop_schema: str,
                          is_required: bool,
                          index: int
                          ) -> str:
                
                prop_rule_name = self.visit(
                    prop_schema,
                    f'{name}{"-" if name else ""}{prop_name}'
                )

                out = ''                
                
                if not is_required:
                    out += ' ('
                    
                if index > 0:
                    out += ' "," space'

                out += rf' {self._format_literal(prop_name)} space ":" space {prop_rule_name}'

                if not is_required:
                    out += ' )?'

                return out
                

            rule = '"{" space'

            index = 0 # used to add separating "," - only 

            # requireds first
            for prop_name in required:
                prop_schema = prop_pairs[prop_name]
                
                rule += emit_prop(prop_name, prop_schema, 
                                  True, index)
                index += 1

            # non-requireds then
            for prop_name in not_required:
                prop_schema = prop_pairs[prop_name]
                
                rule += emit_prop(prop_name, prop_schema, 
                                  False, index)
                index += 1            
            
            rule += ' "}" space'

            return self._add_rule(rule_name, rule)
            

        elif schema_type == "array" and "items" in schema:
            # TODO `prefixItems` keyword
            item_rule_name = self.visit(
                schema["items"], f'{name}{"-" if name else ""}item'
            )
            rule = (
                f'"[" space ({item_rule_name} ("," space {item_rule_name})*)? "]" space'
            )
            return self._add_rule(rule_name, rule)

        else:
            assert schema_type in PRIMITIVE_RULES, f"Unrecognized schema: {schema}"
            return self._add_rule(
                "root" if rule_name == "root" else schema_type,
                PRIMITIVE_RULES[schema_type],
            )

    def format_grammar(self):
        return "\n".join((f"{name} ::= {rule}" for name, rule in self._rules.items()))


def gbnf_from_json_schema(schema: Union[str,dict],
                          prop_order: Optional[list[str]] = []):
    
    """
    prop_order sorting is probably a bad idea, because it makes output order different from the schema example order, which may unnecessarily confuse the model
    """

    if isinstance(schema, str):
        schema = json.loads(schema)
        
    prop_order = {name: idx for idx, name in enumerate(prop_order)}
    
    converter = SchemaConverter(prop_order)
    converter.visit(schema, "")
    
    return converter.format_grammar()



"""
JSON_GBNF from llama.cpp:
https://github.com/ggerganov/llama.cpp/tree/master/grammars
string rule altered to disallow raw \n inside ""
"""

JSON_GBNF = r"""
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\n] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

ws ::= ([ \t\n] ws)?
"""



# Useful:
# Online JSON schema validator: https://www.jsonschemavalidator.net/
# JSON pretty formatter: https://jsonformatter.org/json-pretty-print

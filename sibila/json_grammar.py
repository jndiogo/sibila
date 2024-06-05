"""JSON Schema grammar generation utilities for local models."""

from typing import Any, Optional, Union, Literal, Annotated, get_origin, get_args

import json, re

import logging
logger = logging.getLogger(__name__)



# ============================================================================ Grammars

"""
llama.cpp json-schema to grammar converter adapted from llama_cpp_python:
https://github.com/abetlen/llama-cpp-python

Originally from llama.cpp:
https://github.com/ggerganov/llama.cpp/blob/master/examples/json_schema_to_grammar.py
https://github.com/ggerganov/llama.cpp/tree/master/grammars
"""


# whitespace is constrained to a single space char to prevent model "running away" in
# whitespace. Also maybe improves generation quality?
SPACE_RULE = '" "?'

PRIMITIVE_RULES = {
    "boolean": '("true" | "false") space',
    "number": '("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? space',
    "integer": '("-"? ([0-9] | [1-9] [0-9]*)) space',
    "string": r""" "\"" (
        [^"\\\x7F\x00-\x1F] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\"" space """,
    "null": '"null" space',
}

INVALID_RULE_CHARS_RE = re.compile(r"[^a-zA-Z0-9-]+")
GRAMMAR_LITERAL_ESCAPE_RE = re.compile(r'[\r\n"]')
GRAMMAR_LITERAL_ESCAPES = {"\r": "\\r", "\n": "\\n", '"': '\\"'}


class SchemaConverter:
    _prop_order: dict
    _rules: dict
    _defs: dict

    def __init__(self, 
                 prop_order: dict):
        self._prop_order = prop_order
        self._rules = {"space": SPACE_RULE}
        self._defs: dict[str, Any] = {}

    def _format_literal(self, 
                        literal: str):
        escaped: str = GRAMMAR_LITERAL_ESCAPE_RE.sub(
            lambda m: GRAMMAR_LITERAL_ESCAPES.get(m.group(0)), json.dumps(literal) # type: ignore[arg-type,return-value]
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
            required: list = schema.get("required") or []
            not_required = [n for n in prop_pairs if n not in required]
            
            if len(required) == 0: # force all to be required: or the leading comma for not_required may cause broken JSON
                logger.debug(f"Rule '{rule_name}': GBNF grammar cannot parse rule with only optional items: making all items required")
                required = not_required
                not_required = []

            
            def emit_prop(prop_name: str, 
                          prop_schema: dict[str, Any],
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
                          prop_order: list[str] = []):
    
    """
    prop_order sorting is probably a bad idea, because it makes output order different from the schema example order, which may unnecessarily confuse the model
    """
    out_schema: dict[str, Any]
    if isinstance(schema, str):
        out_schema = json.loads(schema)
    else:
        out_schema = schema
        
    prop_order_inv: dict[str,int] = {name: idx for idx, name in enumerate(prop_order)}
    
    converter = SchemaConverter(prop_order_inv)
    converter.visit(out_schema, "")
    
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
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

ws ::= ([ \t\n] ws)?
"""


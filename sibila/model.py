"""Base model classes.

- Model: Base class for an LLM model, containing most common functionality.
- TextModel: Base class for a model with text-based input/output.
- FormattedTextModel: Base class for a model that uses formatted text (chat templates) for input/output.
- MessagesModel: Base class for a model with message-based input/output.
- Tokenizer: Base tokenizer class to encode and decode tokens, measure text length in tokens, track special tokens.
"""

from typing import Any, Optional, Union, Callable, Annotated, Literal, get_origin, get_args
from abc import ABC, abstractmethod
from enum import IntEnum

import sys, json

from copy import copy
from pprint import pformat

import logging
logger = logging.getLogger(__name__)

from dataclasses import dataclass, is_dataclass

from jsonschema import ( # type: ignore[import-untyped]
    validate as json_schema_validate, 
    ValidationError as json_schema_ValidationError,
    SchemaError as json_schema_SchemaError
)

import jinja2
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment
version = jinja2.__version__.split('.')
if int(version[0]) < 3:
    raise ImportError

def jinja_raise_exception(message):
    raise TemplateError(message)            

from pydantic import BaseModel


from .gen import (
    GenConf,
    GenRes,
    GenError,
    GenOut
)

from .thread import (
    Thread,
    Msg
)

from .json_schema import (
    JSchemaConf,
    json_schema_massage,
    json_schema_from_pydantic,
    pydantic_obj_from_json,
    build_dataclass_object_json_schema,
    build_root_json_schema,
    get_enum_type,
    get_final_type,
    create_final_instance
)

from .utils import (
    is_subclass_of,
    join_text
)




    
class Tokenizer(ABC):
    """Base tokenizer class to encode and decode tokens, measure text length in tokens, track special tokens."""

    bos_token_id: Union[int, None] # beginning of sentence
    bos_token: Union[str, None]

    eos_token_id: Union[int, None] # end of sentence
    eos_token: Union[str, None]

    pad_token_id: Union[int, None] # padding
    pad_token: Union[str, None]

    unk_token_id: Union[int, None] # unknown token
    unk_token: Union[str, None]

    
    def special_tokens(self) -> list[Any]:
        return [(self.bos_token_id, self.bos_token),
                (self.eos_token_id, self.eos_token),
                (self.pad_token_id, self.pad_token),
                (self.unk_token_id, self.unk_token)]
    
    def special_tokens_map(self) -> dict:
        return {"bos_token": self.bos_token,
                "eos_token": self.eos_token,
                "pad_token": self.pad_token,
                "unk_token": self.unk_token}
    
    @abstractmethod
    def encode(self, 
               text: str) -> list[int]:
        """Encode text into model tokens. Inverse of Decode().

        Args:
            text: Text to be encoded.

        Returns:
            A list of ints with the encoded tokens.
        """
        

    @abstractmethod
    def decode(self,
               token_ids: list[int],
               skip_special: bool = True) -> str:
        """Decode model tokens to text. Inverse of Encode().

        Args:
            token_ids: List of model tokens.
            skip_special: Don't decode special tokens like bos and eos. Defaults to True.

        Returns:
            Decoded text.
        """

    def token_len(self, 
                  text: str) -> int:
        """Returns token length for given text.

        Args:
            text: Text to be measured.

        Returns:
            Token length for given text.
        """

        tokens = self.encode(text)
        return len(tokens)        


    def __str__(self):
        return f"""\
bos={self.bos_token_id}='{self.bos_token}'
eos={self.eos_token_id}='{self.eos_token}'
pad={self.pad_token_id}='{self.pad_token}'
unk={self.unk_token_id}='{self.unk_token}'"""


    def __repr__(self):
        return str(self)







class Model(ABC):
    """Model is an abstract base class for common LLM model functionality. Many of the useful methods like extract() or json() are implemented here.

    It should not be instantiated directly, instead LlamaCppModel, OpenAIModel, etc, all derive from this class.
    """

    
    is_local_model: bool
    """Is the model running locally?"""
    
    is_message_model: bool
    """Is communication with the model message-based or text-based/token-based?"""

    tokenizer: Union[Tokenizer,None]
    """Tokenizer used to encode text. Some remote models don't have tokenizer and token length is estimated"""

    genconf: GenConf
    """Generation configuration: options used during gen()."""

    json_format_instructors: dict
    """If GenConf.json / GenConf.json_schema is used, these strings are appended to first thread msg of either instructions or IN kind. See initialization below."""

    json_in_dumps_kwargs: dict
    """Object to string formatting options for json.dumps() calls. See initialization below."""

    ctx_len: int
    """Maximum context token length, including input and model output. There can be a limit for output tokens in the max_tokens_limit."""

    max_tokens_limit: int
    """Some models limit the size of emitted output tokens."""

    maybe_image_input: bool
    """Does the model support images as input? A value of False is definitive, a value of True is actually a maybe, as some providers don't give this information. Check the model specs to be certain."""

    output_key_name: str
    """Name used when an output key needs to be created for JSON output."""

    output_fn_name: str
    """Function name for models that return JSON with a Tools/Functions-style API."""
    
    PROVIDER_NAME: str = NotImplemented
    """Provider prefix that this class handles."""

    
    def __init__(self,
                 is_local_model: bool,
                 genconf: Union[GenConf, None],
                 schemaconf: Union[JSchemaConf, None],
                 tokenizer: Union[Tokenizer, None]):
        """Initializer for base model type, shared by actual model classes like LlamaCpp, OpenAI, etc.

        Args:
            is_local_model: Is the model running locally?
            genconf: Default generation configuration options, used if generation call doesn't supply one.
            schemaconf: Default configuration for JSON schema validation, used if generation call doesn't supply one.
            tokenizer: Tokenizer used to encode text (even for message-based models).
        """
        
        self.is_local_model = is_local_model
        
        self.ctx_len = 0
        self.max_tokens_limit = sys.maxsize
        self.output_key_name = "output"
        self.output_fn_name = "json_out"

        self.tokenizer = tokenizer # type: ignore[assignment]

        if genconf is None:
            self.genconf = GenConf()
        else:
            self.genconf = genconf.clone()
        
        if schemaconf is None:
            self.schemaconf = JSchemaConf()
        else:
            self.schemaconf = schemaconf.clone()



        # set either "json" or "json_schema" key values to None to skip.
        self.json_format_instructors = {
            "json": {
                "bypass_if": ["json"], # bypass appending if all lowercase text values are present in thread
                "append_text": "Output JSON.",
                "sep_count": 2
            },
            "json_schema": {
                "bypass_if": ["json", "schema"],
                "append_text": "Output JSON matching the following schema:\n{{json_schema}}",
                "sep_count": 2
            }
        }

        # text going to model: tight, without \u00xx
        self.json_in_dumps_kwargs = {
            "indent": None,
            "ensure_ascii": False
        } 


    @abstractmethod
    def close(self):
        """Close model, release resources like memory or net connections."""
        ...




    
    # ======================================================== internal generation points    
            

    def gen(self,
            thread: Thread,
            genconf: Optional[GenConf] = None,
            ) -> GenOut:
        """Text generation from a Thread, used by the other model generation methods.
        Doesn't raise an exception if an error occurs, always returns GenOut.

        Args:
            thread: The Thread to use as model input.
            genconf: Model generation configuration. Defaults to None, which uses model's default.

        Raises:
            RuntimeError: If unable to generate.
            NotImplementedError: If method was not defined by a derived class.

        Returns:
            A GenOut object with result, generated text, etc.
            The output text is in GenOut.text.
        """
        raise NotImplementedError


    async def gen_async(self,
                        thread: Thread,
                        genconf: Optional[GenConf] = None,
                        ) -> GenOut:
        """Async text generation from a Thread, used by the other model generation methods.
        Doesn't raise an exception if an error occurs, always returns GenOut.

        Args:
            thread: The Thread to use as model input.
            genconf: Model generation configuration. Defaults to None, which uses model's default.

        Raises:
            RuntimeError: If unable to generate.
            NotImplementedError: If method was not defined by a derived class.

        Returns:
            A GenOut object with result, generated text, etc.
            The output text is in GenOut.text.
        """
        raise NotImplementedError
    


    
    
    def _gen_json_pre(self,
                      thread: Thread,
                      json_schema: Union[dict,str,None],
                      genconf: Union[GenConf,None],

                      massage_schema: bool,
                      schemaconf: Union[JSchemaConf, None]
                      ) -> list:

        if genconf is None:
            genconf = self.genconf

        if genconf.json_schema is not None and json_schema is not None:
            logger.warning("Both arg json_schema and genconf.json_schema are set: using json_schema arg")

        if json_schema is not None:
            if schemaconf is None:
                schemaconf = self.schemaconf

            logger.debug("JSON schema conf:\n" + pformat(schemaconf))

            if massage_schema:
                if not isinstance(json_schema, dict):
                    json_schema = json.loads(json_schema)

                json_schema = json_schema_massage(json_schema, schemaconf) # type: ignore[arg-type]
                logger.debug("Massaged JSON schema:\n" + pformat(json_schema))

        return [thread, genconf(format="json", 
                                json_schema=json_schema)]


    def gen_json(self,
                 thread: Thread,
                 json_schema: Union[dict,str,None],
                 genconf: Optional[GenConf] = None,

                 massage_schema: bool = True,
                 schemaconf: Optional[JSchemaConf] = None,
                 ) -> GenOut:
        """JSON/JSON-schema constrained generation, returning a Python dict of values, conditioned or not by a JSON schema.
        Doesn't raise an exception if an error occurs, always returns GenOut.

        Args:
            thread: The Thread to use as model input.
            json_schema: A JSON schema describing the dict fields that will be output. None means no schema (free JSON output).
            genconf: Model generation configuration. Defaults to None, which uses model's default.
            massage_schema: Simplify schema. Defaults to True.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to Defaults to None, which uses model's default.

        Raises:
            RuntimeError: If unable to generate.

        Returns:
            A GenOut object with result, generated text, etc. The output dict is in GenOut.dic.
        """

        args = self._gen_json_pre(thread,
                                  json_schema,
                                  genconf,
                                  massage_schema,
                                  schemaconf)
        return self.gen(*args)
    

    async def gen_json_async(self,
                             thread: Thread,
                             json_schema: Union[dict,str,None],
                             genconf: Optional[GenConf] = None,

                             massage_schema: bool = True,
                             schemaconf: Optional[JSchemaConf] = None,
                             ) -> GenOut:
        """JSON/JSON-schema constrained generation, returning a Python dict of values, conditioned or not by a JSON schema.
        Doesn't raise an exception if an error occurs, always returns GenOut.

        Args:
            thread: The Thread to use as model input.
            json_schema: A JSON schema describing the dict fields that will be output. None means no schema (free JSON output).
            genconf: Model generation configuration. Defaults to None, which uses model's default.
            massage_schema: Simplify schema. Defaults to True.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to Defaults to None, which uses model's default.

        Raises:
            RuntimeError: If unable to generate.

        Returns:
            A GenOut object with result, generated text, etc. The output dict is in GenOut.dic.
        """

        args = self._gen_json_pre(thread,
                                  json_schema,
                                  genconf,
                                  massage_schema,
                                  schemaconf)
        return await self.gen_async(*args)





    
    def _gen_dataclass_pre(self,
                           cls: Any # a dataclass
                           ) -> dict:
        
        if is_dataclass(cls):
            schema = build_dataclass_object_json_schema(cls)
        else:
            raise TypeError("Only dataclass allowed for cls argument")        
        return schema

    def _gen_dataclass_post(self,
                            out: GenOut,
                            cls: Any,
                            schemaconf: Union[JSchemaConf,None]
                            ) -> GenOut:
    
        if out.dic is not None:
            try:
                obj = create_final_instance(cls, 
                                            is_list=False,
                                            val=out.dic,
                                            schemaconf=schemaconf)
                out.value = obj
                
            except TypeError as e:
                out.res = GenRes.ERROR_JSON_SCHEMA_VAL # error initializing object from JSON
                out.text += f"\nJSON Schema error: {e}"
        else:
            # out.res already holds the right error
            ...
        
        return out




    def gen_dataclass(self,
                      cls: Any, # a dataclass
                      thread: Thread,
                      genconf: Optional[GenConf] = None,
                      schemaconf: Optional[JSchemaConf] = None
                      ) -> GenOut:
        """Constrained generation after a dataclass definition.
        An initialized dataclass object is returned in the "value" field of the returned dict.
        Doesn't raise an exception if an error occurs, always returns GenOut containing the created object.

        Args:
            cls: A dataclass definition.
            thread: The Thread object to use as model input.
            genconf: Model generation configuration. Defaults to None, which uses model's default.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to None, which uses model's default.

        Raises:
            RuntimeError: If unable to generate.

        Returns:
            A GenOut object with result, generated text, etc. The initialized dataclass object is in GenOut.value.
        """

        schema = self._gen_dataclass_pre(cls)

        out = self.gen_json(thread,
                            schema,
                            genconf,
                            massage_schema=True,
                            schemaconf=schemaconf)
    
        return self._gen_dataclass_post(out,
                                        cls,
                                        schemaconf)



    async def gen_dataclass_async(self,
                                  cls: Any, # a dataclass
                                  thread: Thread,
                                  genconf: Optional[GenConf] = None,
                                  schemaconf: Optional[JSchemaConf] = None
                                  ) -> GenOut:
        """Async constrained generation after a dataclass definition.
        An initialized dataclass object is returned in the "value" field of the returned dict.
        Doesn't raise an exception if an error occurs, always returns GenOut containing the created object.

        Args:
            cls: A dataclass definition.
            thread: The Thread object to use as model input.
            genconf: Model generation configuration. Defaults to None, which uses model's default.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to None, which uses model's default.

        Raises:
            RuntimeError: If unable to generate.

        Returns:
            A GenOut object with result, generated text, etc. The initialized dataclass object is in GenOut.value.
        """

        schema = self._gen_dataclass_pre(cls)

        out = await self.gen_json_async(thread,
                                        schema,
                                        genconf,
                                        massage_schema=True,
                                        schemaconf=schemaconf)
    
        return self._gen_dataclass_post(out,
                                        cls,
                                        schemaconf)







    def _gen_pydantic_pre(self,
                          cls: Any # a Pydantic BaseModel class
                          ) -> dict:

        if is_subclass_of(cls, BaseModel):
            schema = json_schema_from_pydantic(cls)
        else:
            raise TypeError("Only pydantic BaseModel allowed for cls argument")
        
        return schema


    def _gen_pydantic_post(self,
                           out: GenOut,
                           cls: Any, # a Pydantic BaseModel class
                           schemaconf: Union[JSchemaConf,None]
                           ) -> GenOut:
    
        if out.dic is not None:
            try:
                obj = pydantic_obj_from_json(cls, 
                                             out.dic,
                                             schemaconf=schemaconf)
                out.value = obj
                
            except TypeError as e:
                out.res = GenRes.ERROR_JSON_SCHEMA_VAL # error validating for object (by Pydantic), but JSON is valid for its schema
                out.text += f"\nJSON Schema error: {e}"
        else:
            # out.res already holds the right error
            ...
        
        return out




    def gen_pydantic(self,
                     cls: Any, # a Pydantic BaseModel class
                     thread: Thread,
                     genconf: Optional[GenConf] = None,
                     schemaconf: Optional[JSchemaConf] = None
                     ) -> GenOut:
        """Constrained generation after a Pydantic BaseModel-derived class definition.
        An initialized Pydantic BaseModel object is returned in the "value" field of the returned dict.
        Doesn't raise an exception if an error occurs, always returns GenOut containing the created object.

        Args:
            cls: A class derived from a Pydantic BaseModel class.
            thread: The Thread to use as model input.
            genconf: Model generation configuration. Defaults to None, which uses model's default.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to None, which uses model's default.

        Raises:
            RuntimeError: If unable to generate.
            TypeError: When cls is not a Pydantic BaseClass.

        Returns:
            A GenOut object with result, generated text, etc. The initialized Pydantic BaseModel-derived object is in GenOut.value.
        """

        schema = self._gen_pydantic_pre(cls)

        out = self.gen_json(thread,
                            schema,
                            genconf,
                            massage_schema=True,
                            schemaconf=schemaconf)
    
        return self._gen_pydantic_post(out,
                                       cls,
                                       schemaconf)



    async def gen_pydantic_async(self,
                                 cls: Any, # a Pydantic BaseModel class
                                 thread: Thread,
                                 genconf: Optional[GenConf] = None,
                                 schemaconf: Optional[JSchemaConf] = None
                                 ) -> GenOut:
        """Async constrained generation after a Pydantic BaseModel-derived class definition.
        An initialized Pydantic BaseModel object is returned in the "value" field of the returned dict.
        Doesn't raise an exception if an error occurs, always returns GenOut containing the created object.

        Args:
            cls: A class derived from a Pydantic BaseModel class.
            thread: The Thread to use as model input.
            genconf: Model generation configuration. Defaults to None, which uses model's default.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to None, which uses model's default.

        Raises:
            RuntimeError: If unable to generate.
            TypeError: When cls is not a Pydantic BaseClass.

        Returns:
            A GenOut object with result, generated text, etc. The initialized Pydantic BaseModel-derived object is in GenOut.value.
        """

        schema = self._gen_pydantic_pre(cls)

        out = await self.gen_json_async(thread,
                                        schema,
                                        genconf,
                                        massage_schema=True,
                                        schemaconf=schemaconf)

        return self._gen_pydantic_post(out,
                                       cls,
                                       schemaconf)






    def _gen_extract_pre(self,
                         target: Any,
                         thread: Thread,
                         genconf: Union[GenConf, None],
                         schemaconf: Union[JSchemaConf, None]
                         ) -> Any:

        schema, created_output_key = build_root_json_schema(target, 
                                                            self.output_key_name)
        final_type, is_list = get_final_type(target)

        if schemaconf is None:
            schemaconf = JSchemaConf()

        return (thread,
                schema,
                genconf,                
                created_output_key,
                final_type, 
                is_list,
                schemaconf)


    def _gen_extract_post(self,
                          out: GenOut,
                          created_output_key: bool,
                          final_type: Any, 
                          is_list: bool,
                          schemaconf: JSchemaConf
                          ) -> GenOut:

        if out.dic is not None:

            if created_output_key:
                if self.output_key_name in out.dic:
                    val = out.dic[self.output_key_name]
                else:
                    out.res = GenRes.ERROR_JSON_SCHEMA_VAL # JSON error
                    out.text += f"\nExpecting key '{self.output_key_name}'."
                    return out

            else:
                val = out.dic

            try:
                value = create_final_instance(final_type, 
                                              is_list, 
                                              val,
                                              schemaconf=schemaconf)
                out.value = value

            except TypeError as e:
                out.res = GenRes.ERROR_JSON_SCHEMA_VAL # error validating, but JSON is valid for its schema
                out.text += f"\nJSON Schema error: {e}"

        else:
            # out.res already holds the right error
            ...

        return out



    def gen_extract(self,
                    target: Any,
                    thread: Thread,
                    genconf: Optional[GenConf] = None,
                    schemaconf: Optional[JSchemaConf] = None
                    ) -> GenOut:
        """Free type constrained generation: an instance of the given type is initialized with the model's output.
        The initialized value is placed in the "value" field of the returned dict.
        Doesn't raise an exception if an error occurs, always returns GenOut.

        The following target types are accepted:

            - prim_type:
                bool
                int
                float
                str
                
            - enums:
                [1, 2, 3] or ["a","b"] - all items of the same prim_type
                Literal['year', 'name'] - all items of the same prim_type
                Enum, EnumInt, EnumStr, (Enum, int),... - all items of the same prim_type

            - datetime/date/time

            - a list in the form:
                list[type] - for example list[int]. 
                The list can be annotated:
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
            target: One of the above types.
            thread: The Thread to use as model input.
            genconf: Model generation configuration. Defaults to None, which uses model's default.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to None, which uses model's default.

        Raises:
            RuntimeError: If unable to generate.

        Returns:
            A GenOut object with model's results and instantiated type in the "value" field.
        """

        (thread,
         schema,
         genconf, 
         created_output_key, 
         final_type, 
         is_list, 
         schemaconf) = self._gen_extract_pre(target,
                                             thread,
                                             genconf,
                                             schemaconf)

        out = self.gen_json(thread,
                            schema,
                            genconf,
                            massage_schema=True,
                            schemaconf=schemaconf)
    
        return self._gen_extract_post(out,
                                      created_output_key,
                                      final_type,
                                      is_list,
                                      schemaconf) # type: ignore [arg-type]




    async def gen_extract_async(self,
                                target: Any,
                                thread: Thread,
                                genconf: Optional[GenConf] = None,
                                schemaconf: Optional[JSchemaConf] = None
                                ) -> GenOut:
        """Async free type constrained generation: an instance of the given type is initialized with the model's output.
        The initialized value is placed in the "value" field of the returned dict.
        Doesn't raise an exception if an error occurs, always returns GenOut.

        The following target types are accepted:

            - prim_type:
                bool
                int
                float
                str
                
            - enums:
                [1, 2, 3] or ["a","b"] - all items of the same prim_type
                Literal['year', 'name'] - all items of the same prim_type
                Enum, EnumInt, EnumStr, (Enum, int),... - all items of the same prim_type

            - datetime/date/time

            - a list in the form:
                list[type] - for example list[int]. 
                The list can be annotated:
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
            target: One of the above types.
            thread: The Thread to use as model input.
            genconf: Model generation configuration. Defaults to None, which uses model's default.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to None, which uses model's default.

        Raises:
            RuntimeError: If unable to generate.

        Returns:
            A GenOut object with model's results and instantiated type in the "value" field.
        """
        (thread,
         schema,
         genconf, 
         created_output_key, 
         final_type, 
         is_list, 
         schemaconf) = self._gen_extract_pre(target,
                                             thread,
                                             genconf,
                                             schemaconf)

        out = await self.gen_json_async(thread,
                                        schema,
                                        genconf,
                                        massage_schema=True,
                                        schemaconf=schemaconf)
    
        return self._gen_extract_post(out,
                                      created_output_key,
                                      final_type,
                                      is_list,
                                      schemaconf) # type: ignore [arg-type]







    # ======================================================== user generation points    


    def __call__(self,             
                 query: Union[Thread,Msg,tuple,str],
                 *,
                 inst: Optional[str] = None,

                 genconf: Optional[GenConf] = None,
                 ok_length_is_error: bool = False
                 ) -> str:
        """Text generation from a Thread or plain text, used by the other model generation methods. Same as call().

        Args:
            query: A Thread or a single IN message given as Msg, list, tuple or str. List and tuple should contain the same args as for creating Msg.
            inst: Instruction message for model. Will override Thread's inst, if set. Defaults to None.
            genconf: Model generation configuration. Defaults to None, which uses model's default.
            ok_length_is_error: Should a result of GenRes.OK_LENGTH be considered an error and raise?

        Raises:
            GenError: If an error occurred. This can be a model error, or an invalid JSON output error.
            RuntimeError: If unable to generate.

        Returns:
            Text generated by model.
        """

        return self.call(query,
                         inst=inst,
                         genconf=genconf,
                         ok_length_is_error=ok_length_is_error)


    def call(self,             
             query: Union[Thread,Msg,tuple,str],
             *,
             inst: Optional[str] = None,

             genconf: Optional[GenConf] = None,
             ok_length_is_error: bool = False
             ) -> str:
        """Text generation from a Thread or plain text, used by the other model generation methods.

        Args:
            query: A Thread or a single IN message given as Msg, list, tuple or str. List and tuple should contain the same args as for creating Msg.
            inst: Instruction message for model. Will override Thread's inst, if set. Defaults to None.
            genconf: Model generation configuration. Defaults to None, which uses model's default.
            ok_length_is_error: Should a result of GenRes.OK_LENGTH be considered an error and raise?

        Raises:
            GenError: If an error occurred. This can be a model error, or an invalid JSON output error.
            RuntimeError: If unable to generate.

        Returns:
            Text generated by model.
        """
        
        thread = Thread.ensure(query, inst)

        out = self.gen(thread=thread, 
                       genconf=genconf)
        
        GenError.raise_if_error(out,
                                ok_length_is_error=ok_length_is_error)

        return out.text


    async def call_async(self,
                         query: Union[Thread,Msg,tuple,str],
                         *,
                         inst: Optional[str] = None,
 
                         genconf: Optional[GenConf] = None,
                         ok_length_is_error: bool = False
                         ) -> str:
        """Text generation from a Thread or plain text, used by the other model generation methods.

        Args:
            query: A Thread or a single IN message given as Msg, list, tuple or str. List and tuple should contain the same args as for creating Msg.
            inst: Instruction message for model. Will override Thread's inst, if set. Defaults to None.
            genconf: Model generation configuration. Defaults to None, which uses model's default.
            ok_length_is_error: Should a result of GenRes.OK_LENGTH be considered an error and raise?

        Raises:
            GenError: If an error occurred. This can be a model error, or an invalid JSON output error.
            RuntimeError: If unable to generate.

        Returns:
            Text generated by model.
        """
        
        thread = Thread.ensure(query, inst)

        out = await self.gen_async(thread=thread, 
                                   genconf=genconf)
        
        GenError.raise_if_error(out,
                                ok_length_is_error=ok_length_is_error)

        return out.text









    def json(self,
             query: Union[Thread,Msg,tuple,str],
             *,
             json_schema: Union[dict,str,None] = None,
             inst: Optional[str] = None,

             genconf: Optional[GenConf] = None,
             massage_schema: bool = True,
             schemaconf: Optional[JSchemaConf] = None,
             ) -> dict:
        """JSON/JSON-schema constrained generation, returning a Python dict of values, constrained or not by a JSON schema.
        Raises GenError if unable to get a valid/schema-validated JSON.

        Args:
            query: A Thread or a single IN message given as Msg, list, tuple or str. List and tuple should contain the same args as for creating Msg.
            json_schema: A JSON schema describing the dict fields that will be output. None means no schema (free JSON output).
            inst: Instruction message for model. Will override Thread's inst, if set. Defaults to None.
            genconf: Model generation configuration. Defaults to None, which uses model's default.
            massage_schema: Simplify schema. Defaults to True.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to None, which uses model's default.

        Raises:
            GenError: If an error occurred, for example an invalid JSON schema output error. See GenError.
            RuntimeError: If unable to generate.

        Returns:
            A dict from model's JSON response, following genconf.jsonschema, if provided.
        """        

        thread = Thread.ensure(query, inst)

        out = self.gen_json(thread,
                            json_schema,                            
                            genconf,
                            massage_schema,
                            schemaconf)
        
        GenError.raise_if_error(out,
                                ok_length_is_error=False) # as valid JSON can still be produced

        return out.dic # type: ignore[return-value]






    async def json_async(self,             
                         query: Union[Thread,Msg,tuple,str],
                         *,
                         json_schema: Union[dict,str,None] = None,
                         inst: Optional[str] = None,

                         genconf: Optional[GenConf] = None,
                         massage_schema: bool = True,
                         schemaconf: Optional[JSchemaConf] = None,
                         ) -> dict:
        """JSON/JSON-schema constrained generation, returning a Python dict of values, constrained or not by a JSON schema.
        Raises GenError if unable to get a valid/schema-validated JSON.

        Args:
            query: A Thread or a single IN message given as Msg, list, tuple or str. List and tuple should contain the same args as for creating Msg.
            json_schema: A JSON schema describing the dict fields that will be output. None means no schema (free JSON output).
            inst: Instruction message for model. Will override Thread's inst, if set. Defaults to None.
            genconf: Model generation configuration. Defaults to None, which uses model's default.
            massage_schema: Simplify schema. Defaults to True.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to None, which uses model's default.

        Raises:
            GenError: If an error occurred, for example an invalid JSON schema output error. See GenError.
            RuntimeError: If unable to generate.

        Returns:
            A dict from model's JSON response, following genconf.jsonschema, if provided.
        """        

        thread = Thread.ensure(query, inst)

        out = await self.gen_json_async(thread,
                                        json_schema,
                                        genconf,
                                        massage_schema,
                                        schemaconf)
        
        GenError.raise_if_error(out,
                                ok_length_is_error=False) # as valid JSON can still be produced

        return out.dic # type: ignore[return-value]











    def dataclass(self, # noqa: F811
                  cls: Any, # a dataclass definition

                  query: Union[Thread,Msg,tuple,str],
                  *,
                  inst: Optional[str] = None,

                  genconf: Optional[GenConf] = None,
                  schemaconf: Optional[JSchemaConf] = None
                  ) -> Any: # a dataclass object
        """Constrained generation after a dataclass definition, resulting in an object initialized with the model's response.
        Raises GenError if unable to get a valid response that follows the dataclass definition.

        Args:
            cls: A dataclass definition.
            query: A Thread or a single IN message given as Msg, list, tuple or str. List and tuple should contain the same args as for creating Msg.
            inst: Instruction message for model. Will override Thread's inst, if set. Defaults to None.
            genconf: Model generation configuration. Defaults to None, which uses model's default.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to None, which uses model's default.

        Raises:
            GenError: If an error occurred, for example invalid object initialization. See GenError.
            RuntimeError: If unable to generate.

        Returns:
            An object of class cls (derived from dataclass) initialized from the constrained JSON output.
        """

        thread = Thread.ensure(query, inst)

        out = self.gen_dataclass(cls,
                                 thread,
                                 genconf,
                                 schemaconf)

        GenError.raise_if_error(out,
                                ok_length_is_error=False) # as valid JSON can still be produced

        return out.value



    async def dataclass_async(self, # noqa: E811
                              cls: Any, # a dataclass definition

                              query: Union[Thread,Msg,tuple,str],
                              *,
                              inst: Optional[str] = None,

                              genconf: Optional[GenConf] = None,
                              schemaconf: Optional[JSchemaConf] = None
                              ) -> Any: # a dataclass object
        """Async constrained generation after a dataclass definition, resulting in an object initialized with the model's response.
        Raises GenError if unable to get a valid response that follows the dataclass definition.

        Args:
            cls: A dataclass definition.
            query: A Thread or a single IN message given as Msg, list, tuple or str. List and tuple should contain the same args as for creating Msg.
            inst: Instruction message for model. Will override Thread's inst, if set. Defaults to None.
            genconf: Model generation configuration. Defaults to None, which uses model's default.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to None, which uses model's default.

        Raises:
            GenError: If an error occurred, for example invalid object initialization. See GenError.
            RuntimeError: If unable to generate.

        Returns:
            An object of class cls (derived from dataclass) initialized from the constrained JSON output.
        """

        thread = Thread.ensure(query, inst)

        out = await self.gen_dataclass_async(cls,
                                             thread,
                                             genconf,
                                             schemaconf)

        GenError.raise_if_error(out,
                                ok_length_is_error=False) # as valid JSON can still be produced

        return out.value









    def pydantic(self,
                 cls: Any, # a Pydantic BaseModel class

                 query: Union[Thread,Msg,tuple,str],
                 *,
                 inst: Optional[str] = None,

                 genconf: Optional[GenConf] = None,
                 schemaconf: Optional[JSchemaConf] = None
                 ) -> Any: # a Pydantic BaseModel object
        """Constrained generation after a Pydantic BaseModel-derived class definition.
        Results in an object initialized with the model response.
        Raises GenError if unable to get a valid dict that follows the BaseModel class definition.

        Args:
            cls: A class derived from a Pydantic BaseModel class.
            query: A Thread or a single IN message given as Msg, list, tuple or str. List and tuple should contain the same args as for creating Msg.
            inst: Instruction message for model. Will override Thread's inst, if set. Defaults to None.
            genconf: Model generation configuration. Defaults to None, which uses model's default.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to None, which uses model's default.

        Raises:
            GenError: If an error occurred, for example an invalid BaseModel object. See GenError.
            RuntimeError: If unable to generate.

        Returns:
            A Pydantic object of class cls (derived from BaseModel) initialized from the constrained JSON output.
        """

        thread = Thread.ensure(query, inst)

        out = self.gen_pydantic(cls,
                                thread,
                                genconf,
                                schemaconf)

        GenError.raise_if_error(out,
                                ok_length_is_error=False) # as valid JSON can still be produced

        return out.value


    async def pydantic_async(self,
                             cls: Any, # a Pydantic BaseModel class

                             query: Union[Thread,Msg,tuple,str],
                             *,
                             inst: Optional[str] = None,

                             genconf: Optional[GenConf] = None,
                             schemaconf: Optional[JSchemaConf] = None
                             ) -> Any: # a Pydantic BaseModel object
        """Async constrained generation after a Pydantic BaseModel-derived class definition.
        Results in an object initialized with the model response.
        Raises GenError if unable to get a valid dict that follows the BaseModel class definition.

        Args:
            cls: A class derived from a Pydantic BaseModel class.
            query: A Thread or a single IN message given as Msg, list, tuple or str. List and tuple should contain the same args as for creating Msg.
            inst: Instruction message for model. Will override Thread's inst, if set. Defaults to None.
            genconf: Model generation configuration. Defaults to None, which uses model's default.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to None, which uses model's default.

        Raises:
            GenError: If an error occurred, for example an invalid BaseModel object. See GenError.
            RuntimeError: If unable to generate.

        Returns:
            A Pydantic object of class cls (derived from BaseModel) initialized from the constrained JSON output.
        """

        thread = Thread.ensure(query, inst)

        out = await self.gen_pydantic_async(cls,
                                            thread,
                                            genconf,
                                            schemaconf)

        GenError.raise_if_error(out,
                                ok_length_is_error=False) # as valid JSON can still be produced

        return out.value







    def extract(self,
                target: Any,

                query: Union[Thread,Msg,tuple,str],
                *,
                inst: Optional[str] = None,

                genconf: Optional[GenConf] = None,
                schemaconf: Optional[JSchemaConf] = None
                ) -> Any:        
        """Type-constrained generation: an instance of the given type will be initialized with the model's output.
        The following target types are accepted:

        - prim_type:

            - bool
            - int
            - float
            - str
            
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
            target: One of the above types.
            query: A Thread or a single IN message given as Msg, list, tuple or str. List and tuple should contain the same args as for creating Msg.
            inst: Instruction message for model. Will override Thread's inst, if set. Defaults to None.
            genconf: Model generation configuration. Defaults to None, which uses model's default.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to None, which uses model's default.

        Raises:
            GenError: If an error occurred, for example invalid object initialization. See GenError.
            RuntimeError: If unable to generate.

        Returns:
            A value of target arg type instantiated with the model's output.
        """

        thread = Thread.ensure(query, inst)

        out = self.gen_extract(target,
                               thread,
                               genconf,
                               schemaconf)

        GenError.raise_if_error(out,
                                ok_length_is_error=False) # as valid JSON can still be produced

        return out.value





    async def extract_async(self,
                            target: Any,

                            query: Union[Thread,Msg,tuple,str],
                            *,
                            inst: Optional[str] = None,

                            genconf: Optional[GenConf] = None,
                            schemaconf: Optional[JSchemaConf] = None
                            ) -> Any:        
        """Async type-constrained generation: an instance of the given type will be initialized with the model's output.
        The following target types are accepted:

        - prim_type:

            - bool
            - int
            - float
            - str
            
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
            target: One of the above types.
            query: A Thread or a single IN message given as Msg, list, tuple or str. List and tuple should contain the same args as for creating Msg.
            inst: Instruction message for model. Will override Thread's inst, if set. Defaults to None.
            genconf: Model generation configuration. Defaults to None, which uses model's default.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to None, which uses model's default.

        Raises:
            GenError: If an error occurred, for example invalid object initialization. See GenError.
            RuntimeError: If unable to generate.

        Returns:
            A value of target arg type instantiated with the model's output.
        """

        thread = Thread.ensure(query, inst)

        out = await self.gen_extract_async(target,
                                           thread,
                                           genconf,
                                           schemaconf)

        GenError.raise_if_error(out,
                                ok_length_is_error=False) # as valid JSON can still be produced

        return out.value









    def classify(self,
                 labels: Any,

                 query: Union[Thread,Msg,tuple,str],
                 *,
                 inst: Optional[str] = None,

                 genconf: Optional[GenConf] = None,
                 schemaconf: Optional[JSchemaConf] = None
                 ) -> Any:
        """Returns a classification from one of the given enumeration values
        The following ways to specify the valid labels are accepted:

        - [1, 2, 3] or ["a","b"] - all items of the same prim_type
        - Literal['year', 'name'] - all items of the same prim_type
        - Enum, EnumInt, EnumStr, (Enum, int),... - all items of the same prim_type

        Args:
            labels: One of the above types.
            query: A Thread or a single IN message given as Msg, list, tuple or str. List and tuple should contain the same args as for creating Msg.
            inst: Instruction message for model. Will override Thread's inst, if set. Defaults to None.
            genconf: Model generation configuration. Defaults to None, which uses model's default.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to None, which uses model's default.

        Raises:
            GenError: If an error occurred. See GenError.
            RuntimeError: If unable to generate.

        Returns:
            One of the given labels, as classified by the model.
        """

        # verify it's a valid enum "type"
        type_,_ = get_enum_type(labels)
        if type_ is None:
            raise TypeError("Arg labels must be one of Literal, Enum class or a list of str, float or int items")
        
        return self.extract(labels,
                            query,
                            inst=inst,
                            genconf=genconf,
                            schemaconf=schemaconf)



    async def classify_async(self,
                             labels: Any,

                             query: Union[Thread,Msg,tuple,str],
                             *,
                             inst: Optional[str] = None,

                             genconf: Optional[GenConf] = None,
                             schemaconf: Optional[JSchemaConf] = None
                             ) -> Any:
        """Returns a classification from one of the given enumeration values
        The following ways to specify the valid labels are accepted:

        - [1, 2, 3] or ["a","b"] - all items of the same prim_type
        - Literal['year', 'name'] - all items of the same prim_type
        - Enum, EnumInt, EnumStr, (Enum, int),... - all items of the same prim_type

        Args:
            labels: One of the above types.
            query: A Thread or a single IN message given as Msg, list, tuple or str. List and tuple should contain the same args as for creating Msg.
            inst: Instruction message for model. Will override Thread's inst, if set. Defaults to None.
            genconf: Model generation configuration. Defaults to None, which uses model's default.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to None, which uses model's default.

        Raises:
            GenError: If an error occurred. See GenError.
            RuntimeError: If unable to generate.

        Returns:
            One of the given labels, as classified by the model.
        """

        # verify it's a valid enum "type"
        type_,_ = get_enum_type(labels)
        if type_ is None:
            raise TypeError("Arg labels must be one of Literal, Enum class or a list of str, float or int items")
        
        return await self.extract_async(labels,
                                        query,
                                        inst=inst,
                                        genconf=genconf,
                                        schemaconf=schemaconf)









    
    # ======================================================== properties

     

    @abstractmethod
    def token_len(self,
                  thread_or_text: Union[Thread,str],
                  genconf: Optional[GenConf] = None) -> int:
        """Calculate or estimate the token length for a Thread or a plain text string.
        In some cases where it's not possible to calculate the exact token count, 
        this function should give a conservative (upper bound) estimate.
        It's up to the implementation whether to account for side information like JSON Schema,
        but it must reflect the model's context token accounting.
        Thread or text must be the final text which will passed to model.

        Args:
            thread_or_text: Final thread or text to be passed to model.
            genconf: Model generation configuration. Defaults to None.

        Returns:
            Number of tokens occupied.
        """
        ...



    @property
    def token_len_lambda(self) -> Callable[[Union[Thread,str],Optional[GenConf]], int]:
        return lambda thread_or_text, genconf=None: self.token_len(thread_or_text, genconf) # type: ignore[misc]




    def calc_max_max_tokens(self,
                            input_token_len: int) -> int:
        """Calculate maximum length of tokens that the model can output (as in GenConf.max_tokens).
        May be smaller than remaining length due to max_tokens_limit attribute.

        Args:
            input_token_len: Length in tokens of model input.

        Returns:
            Maximum tokens that can be generated by model.
        """

        token_len = self.ctx_len - input_token_len
        return min(token_len, self.max_tokens_limit)



    def resolve_genconf_max_tokens(self,
                                   input_token_len: int,
                                   genconf: GenConf) -> int:
        """Resolve genconf.max_tokens to a definitive value, depending on input, ctx_len and max_tokens_limit"""
        
        avail_output_tokens = self.calc_max_max_tokens(input_token_len)
        if avail_output_tokens <= 0:
            raise ValueError(f"""Input token length ({input_token_len}) doesn't fit available ctx_len ({self.ctx_len}) or max_tokens_limit ({self.max_tokens_limit})""")

        # calc maximum possible output
        resolved_max_tokens = genconf.resolve_max_tokens(self.ctx_len, self.max_tokens_limit)

        # ensure avail_output_tokens fits resolved_max_tokens
        max_tokens = min(resolved_max_tokens, avail_output_tokens)
        return max_tokens



    @classmethod
    def known_models(cls,
                     api_key: Optional[str] = None) -> Union[list[str], None]:
        """If the model can only use a fixed set of models, return their names. Otherwise, return None.

        Args:
            api_key: If the model provider requires an API key, pass it here or set it in the respective env variable.

        Returns:
            Returns a list of known models or None if unable to fetch it.
        """
        return None





    @abstractmethod
    def name(self) -> str:
        """Model (short) name."""
        ...

    @abstractmethod
    def desc(self) -> str:
        """Model description."""
        ...
    
    def info(self) -> str:
        """Model description and config information."""
        return f"desc='{self.desc()}',\n" \
               f"ctx_len={self.ctx_len}, max_tokens_limit={self.max_tokens_limit},\n" \
               f"genconf={self.genconf}"
               

    @classmethod
    @abstractmethod
    def provider_version(cls) -> str:
        """Provider library version: provider x.y.z
        Ex. llama-cpp-python-0.2.44
        """
        ...

    @classmethod
    def version(cls) -> str:
        """Sibila version + provider version
        Ex: sibila=0.2.3 provider=llama-cpp-python-0.2.44
        """        
        from .__init__ import __version__ as version  # type: ignore[import-not-found]
        return f"sibila={version} provider={cls.provider_version()}"
        


    def __str__(self):
        return self.desc_info()





    
    
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = internal
    
    def _prepare_gen_thread(self,
                            thread: Thread,
                            genconf: GenConf) -> Thread:
        """Perform common pre-generation operations of the thread that will be used.

        Args:
            thread: Input thread that may be augmented here. If not modified, the same thread is returned.
            genconf: Generation configuration.

        Raises:
            ValueError: If empty thread.

        Returns:
            A thread with A GenOut object with result, generated text, etc.
        """

        logger.debug(f"GenConf: {genconf}")

        if not len(thread):
            raise ValueError("Cannot generate from an empty thread")
        
        if thread[0].kind != Msg.Kind.IN:
            logger.warning("First thread message is not an IN message.")
        if thread[-1].kind != Msg.Kind.IN:
            logger.warning("Last thread message is not an IN message.")

        if genconf.format == "json":
            if genconf.json_schema is None:
                inst = self.json_format_instructors["json"]
            else:
                inst = self.json_format_instructors["json_schema"]

            if inst is not None:
                bypass = all([thread.has_text_lower(t) for t in inst["bypass_if"]])
    
                if not bypass:
                    thread = thread.clone()
                    text = inst["append_text"]
                    
                    if genconf.json_schema is not None:
                        if isinstance(genconf.json_schema, str): 
                            sc = genconf.json_schema
                        else:
                            sc = json.dumps(genconf.json_schema, **self.json_in_dumps_kwargs)

                        text = text.replace("{{json_schema}}", sc)

                    first_text = thread[0].text
                    first_text = join_text(first_text, 
                                           text,
                                           thread.join_sep * inst["sep_count"]) # will separate by join_sep * sep_count
                    thread[0].text = first_text
                    
                    logger.debug(f'Appended {"json" if genconf.json_schema is None else "json_schema"} instruction to first message.')

            
        logger.debug(f"Prepare: {str(thread)}")
        
        return thread


    
    
    def _prepare_gen_out(self,
                         response: Union[str,dict, None],
                         finish: str,
                         genconf: GenConf) -> GenOut:
        """Perform common operations over the generated model response.

        Args:
            response: Text or a JSON object obtained from model.
            finish: Model finish reason - only "stop", "length" are used, others will map as GenRes.ERROR_MODEL.

        Returns:
            A GenOut object with result, generated text, etc. 
        """

        logger.debug(f"Response finish='{finish}': █{response}█")
        
        if response is None:
            response = ''

        if genconf.format == "json":
            if isinstance(response, str):
                response = response.strip()

                if "\\u" in response:
                    # dumps(with default ensure_ascii=False) -> ascii (subset of utf-8) -> text.encode("latin1") -> latin1 ->
                    #   decode("unicode-escape") -> utf-8
                    response = response.encode("latin1").decode("unicode-escape")

                # some troubled remote models may include chit-chat after the JSON
                begin = response.find("{")
                if begin > 0:
                    response = response[begin:]
                end = response.rfind("}")
                if end > 0:
                    response = response[:end + 1]

            try:
                if isinstance(response, str):
                    dic = json.loads(response)
                else:
                    dic = response

                if genconf.json_schema is not None:
                    if isinstance(genconf.json_schema, str):
                        schema_dic = json.loads(genconf.json_schema)
                    else:
                        schema_dic = genconf.json_schema
                        
                    json_schema_validate(dic, schema_dic)

                out = GenOut(res=GenRes.from_finish_reason(finish),
                             text=str(response),
                             dic=dic)
            
            except json.JSONDecodeError as err:
                out = GenOut(res=GenRes.from_finish_reason("!json"),
                             text=f"'{err}' {response}")
                
            except json_schema_ValidationError as err:
                logger.info(f"JSON schema validation error: {err.message}")
                out = GenOut(res=GenRes.from_finish_reason("!json_schema_val"),
                             text=f"'{err.message}' {response}",
                             dic=dic)
                
            except json_schema_SchemaError as err:
                logger.info(f"JSON schema error: {err.message}")
                out = GenOut(res=GenRes.from_finish_reason("!json_schema_error"),
                             text=f"'{err.message}' {response}",
                             dic=dic)

        else:
            if isinstance(response, str):
                response = response.strip()
            else:
                response = str(response)

            out = GenOut(res=GenRes.from_finish_reason(finish),
                         text=response)
            
       
        return out
    








from .models import Models


class FormattedTextModel(Model, ABC):
    """Model that uses formatted text (chat templates) for input/output."""
    
    format_template: Union[str,None]
    _jinja_compiled_template: Union[Any,None]

    def __init__(self,
                 is_local_model: bool,
                 genconf: Union[GenConf, None],
                 schemaconf: Union[JSchemaConf, None],
                 tokenizer: Union[Tokenizer, None]):

        super().__init__(is_local_model,
                         genconf,
                         schemaconf,
                         tokenizer)

        self.is_message_model = False

        self.format_template = None
        self._jinja_compiled_template = None
        
    
    def init_format(self,
                    format: Union[str,None],
                    format_search_order: list,
                    info: dict):
        """Search for a chat template format suitable for model.

        Arg format_search_order is a list of str flags:
            "name: match by name
            "meta_template": use model file metadata's template
            "folder_json": search from "models.json" and/or "formats.json" in the same folder
            ex: ["name", "meta_template", "folder_json"]
        
        Arg info required fields: {
            "name": "filename.gguf"
            "path": full path to model
            "meta_template_name": "chat_template"
        }

        Args:
            format: Format name or actual Jinja template.
            format_search_order: List of flags specifying an order for format searching.
            info: Extra data for format searching.

        Raises:
            ValueError: If a chat template format was not found.

        Returns:
            A Jinja template format.
        """

        # handle chat template format
        def search_format() -> str:

            for order in format_search_order:

                if order == "name":
                    fmt = Models.match_format_template(info["name"])
                    if fmt is not None:
                        return fmt
                        
                elif order == "meta_template":
                    md = self.get_metadata()
                    key_name = info["meta_template_name"]
                    if key_name in md:
                        logger.debug(f"Format from model file metadata ('{key_name}') template='{md[key_name]}'")
                        fmt = md[key_name]
                        return fmt # type: ignore[return-value]
                    
                elif order == "folder_json":
                    fmt = Models.folder_match_format_template(info["path"])
                    if fmt is not None:
                        return fmt

            raise ValueError(f"Could not find chat template format for model '{info['name']}'. "
                             "Fine-tuned models cannot work well without the right format. "
                             "Please provide a chat template in the 'format' argument: either as a Jinja template, or the format name if already defined in Models' formats.json. "
                             "See the docs for more information.")

        
        if format is not None: # format was passed (call or Models' model entry)
            if '{{' in format: # an str with a jinja template
                self.format_template = format
                
            else: # a format name
                self.format_template = Models.get_format_template(format)
                if self.format_template is None:
                    logger.warning(f"Could not find format name '{format}'. Will further search in model's metadata and own folder.")
                
        if self.format_template is None: # attempt to find format by other means
            self.format_template = search_format() # will raise if unable to find


        # setup jinja template
        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        jinja_env.globals["raise_exception"] = jinja_raise_exception
        self._jinja_compiled_template = jinja_env.from_string(self.format_template)






    def gen(self, 
            thread: Thread,
            genconf: Optional[GenConf] = None,
            ) -> GenOut:
        """Text generation from a Thread, used by the other model generation methods.
        Doesn't raise an exception if an error occurs, always returns GenOut.

        Args:
            thread: The Thread object to use as model input.
            genconf: Model generation configuration. Defaults to None, which uses model's default.

        Raises:
            ValueError: If trying to generate from an empty prompt.
            RuntimeError: If unable to generate.

        Returns:
            A GenOut object with result, generated text, etc. 
        """

        if genconf is None:
            genconf = self.genconf

        text,finish = self._gen_thread(thread, genconf)

        return self._prepare_gen_out(text, finish, genconf)




    async def gen_async(self, 
                        thread: Thread,
                        genconf: Optional[GenConf] = None,
                        ) -> GenOut:
        """Asynchronous generation from a Thread, used by the other model generation methods.
        Doesn't raise an exception if an error occurs, always returns GenOut.

        Args:
            thread: The Thread object to use as model input.
            genconf: Model generation configuration. Defaults to None, which uses model's default.

        Raises:
            ValueError: If trying to generate from an empty prompt.
            RuntimeError: If unable to generate.

        Returns:
            A GenOut object with result, generated text, etc. 
        """

        if genconf is None:
            genconf = self.genconf

        text,finish = await self._gen_thread_async(thread, genconf)

        return self._prepare_gen_out(text, finish, genconf)



    
    @abstractmethod
    def _gen_thread(self,
                    thread: Thread,
                    genconf: GenConf) -> tuple[str,str]:
        """Generate from formatted text.

        Args:
            thread: Input Thread object.
            genconf: Model generation configuration.

        Raises:
            RuntimeError: If unable to generate.

        Returns:
            Tuple of strings: generated_text, finish_reason.
        """
        ...


    @abstractmethod
    async def _gen_thread_async(self,
                                thread: Thread,
                                genconf: GenConf) -> tuple[str,str]:
        """Asynchronously generate from formatted text.

        Args:
            thread: Input Thread object.
            genconf: Model generation configuration.

        Raises:
            RuntimeError: If unable to generate.

        Returns:
            Tuple of strings: generated_text, finish_reason.
        """
        ...



    def get_metadata(self) -> dict:
        """Returns model metadata."""
        return {}













class MessagesModel(Model, ABC):
    """Model with message-based communication."""

    def __init__(self,
                 is_local_model: bool,
                 genconf: Union[GenConf,None],
                 schemaconf: Union[JSchemaConf,None],
                 tokenizer: Union[Tokenizer, None]):

        super().__init__(is_local_model,
                         genconf,
                         schemaconf,
                         tokenizer)
        
        self.is_message_model = True
                


    def resolve_settings(self,
                         provider: str,
                         name: str,
                         keys: list[str]) -> dict:
        """
        Return: dict with values for found keys.            
        """

        # 1: locate a provider:name entry in Models (in res/base_models.json or overridden by user settings)
        res_name = provider + ":" + name
        provider, name, args = Models.resolve_model_entry(res_name)

        logger.debug(f"Resolved '{res_name}' to '{provider}:{name}' with defaults: {args}")

        # 2: update located defaults with args
        out = {}
        for key in keys:
            if key in args:
                out[key] = args[key]
        return out




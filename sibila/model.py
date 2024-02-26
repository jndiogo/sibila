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

import json

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
    MsgKind
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

from .utils import is_subclass_of



    



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

    @property
    def token_len_lambda(self) -> Callable[[str], int]:
        return lambda text: self.token_len(text)


    def __str__(self):
        return f"""\
bos={self.bos_token_id}='{self.bos_token}'
eos={self.eos_token_id}='{self.eos_token}'
pad={self.pad_token_id}='{self.pad_token}'
unk={self.unk_token_id}='{self.unk_token}'"""








class Model(ABC):
    """Model is a base class for an LLM model with common functionality.

    LlamaCppModel, OpenAIModel, etc, derive from this class.
    """

    
    is_local_model: bool
    """Is the model running locally?"""
    
    is_message_model: bool
    """Is communication with the model message-based or text-based/token-based?"""

    tokenizer: Tokenizer
    """Tokenizer used to encode text (even for message-based models)."""

    genconf: GenConf
    """Generation configuration: options used during gen()."""

    json_format_instructors: dict
    """If GenConf.json / GenConf.json_schema is used, these strings are appended to first thread msg of either SYS or IN kind. See initialization below."""

    json_in_dumps_kwargs: dict
    """Object to string formatting options for json.dumps() calls. See initialization below."""

    _ctx_len: int
    """Context size: shared input and output context length."""


    
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
        
        self._ctx_len = 0

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
            NotImplementedError: If method was not defined by a derived class.

        Returns:
            A GenOut object with result, generated text, etc.
            The output text is in GenOut.text.
        """
        raise NotImplementedError


    


    
    

    def gen_json(self,
                 json_schema: Union[dict,str,None],
                
                 thread: Thread,
                 genconf: Optional[GenConf] = None,

                 massage_schema: bool = True,
                 schemaconf: Optional[JSchemaConf] = None,
                 ) -> GenOut:
        """JSON/JSON-schema constrained generation, returning a Python dict of values, conditioned or not by a JSON schema.
        Doesn't raise an exception if an error occurs, always returns GenOut.

        Args:
            json_schema: A JSON schema describing the dict fields that will be output. None means no schema (free JSON output).
            thread: The Thread to use as model input.
            genconf: Model generation configuration. Defaults to None, which uses model's default.
            massage_schema: Simplify schema. Defaults to True.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to Defaults to None, which uses model's default.

        Returns:
            A GenOut object with result, generated text, etc. The output dict is in GenOut.dic.
        """

        if genconf is None:
            genconf = self.genconf

        if genconf.json_schema is not None and json_schema is not None:
            logger.warn("Both arg json_schema and genconf.json_schema are set: using json_schema arg")

        if json_schema is not None:
            if schemaconf is None:
                schemaconf = self.schemaconf

            logger.debug("JSON schema conf:\n" + pformat(schemaconf))

            if massage_schema:
                if not isinstance(json_schema, dict):
                    json_schema = json.loads(json_schema)

                json_schema = json_schema_massage(json_schema, schemaconf) # type: ignore[arg-type]
                logger.debug("Massaged JSON schema:\n" + pformat(json_schema))

        out = self.gen(thread, 
                       genconf(format="json", 
                               json_schema=json_schema))
        
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

        Returns:
            A GenOut object with result, generated text, etc. The initialized dataclass object is in GenOut.value.
        """

        if is_dataclass(cls):
            schema = build_dataclass_object_json_schema(cls)
        else:
            raise TypeError("Only dataclass allowed for argument cls")

        out = self.gen_json(schema,
                            thread,
                            genconf,
                            massage_schema=True,
                            schemaconf=schemaconf)
    
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

        Returns:
            A GenOut object with result, generated text, etc. The initialized Pydantic BaseModel-derived object is in GenOut.value.
        """

        if is_subclass_of(cls, BaseModel):
            schema = json_schema_from_pydantic(cls)
        else:
            raise TypeError("Only pydantic BaseModel allowed for argument cls")

        out = self.gen_json(schema,
                            thread,
                            genconf,
                            massage_schema=True,
                            schemaconf=schemaconf)
    
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

        Returns:
            A GenOut object with model's results and instantiated type in the "value" field.
        """

        OUTPUT_KEY_NAME = "output"

        schema, created_output_key = build_root_json_schema(target, OUTPUT_KEY_NAME)
        
        final_type, is_list = get_final_type(target)

        if schemaconf is None:
            schemaconf = JSchemaConf()

        out = self.gen_json(schema,
                            thread,
                            genconf,
                            massage_schema=True,
                            schemaconf=schemaconf)
    
        if out.dic is not None:

            if created_output_key:
                val = out.dic[OUTPUT_KEY_NAME]
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





    # ======================================================== user generation points    


    def __call__(self,             
                 query: Union[str,Thread],
                 *,
                 inst: Optional[str] = None,

                 genconf: Optional[GenConf] = None,
                 ok_length_is_error: bool = False
                 ) -> str:
        """Text generation from a Thread or plain text, used by the other model generation methods.

        Args:
            query: Thread or an str with the text of a single IN message to use as model input.
            inst: Instruction message for model. Will override Thread's inst, if set. Defaults to None.
            genconf: Model generation configuration. Defaults to None, which uses model's default.
            ok_length_is_error: Should a result of GenRes.OK_LENGTH be considered an error and raise?

        Raises:
            GenError: If an error occurred. This can be a model error, or an invalid JSON output error.

        Returns:
            Text generated by model.
        """
        
        thread = Thread.ensure(query, inst)

        out = self.gen(thread=thread, 
                       genconf=genconf)
        
        GenError.raise_if_error(out,
                                ok_length_is_error=ok_length_is_error)

        return out.text




    def json(self,             
             json_schema: Union[dict,str,None],
             
             query: Union[str,Thread],
             *,
             inst: Optional[str] = None,

             genconf: Optional[GenConf] = None,
             massage_schema: bool = True,
             schemaconf: Optional[JSchemaConf] = None,
             ) -> dict:
        """JSON/JSON-schema constrained generation, returning a Python dict of values, constrained or not by a JSON schema.
        Raises GenError if unable to get a valid/schema-validated JSON.

        Args:
            json_schema: A JSON schema describing the dict fields that will be output. None means no schema (free JSON output).
            query: Thread or an str with the text of a single IN message to use as model input.
            inst: Instruction message for model. Will override Thread's inst, if set. Defaults to None.
            genconf: Model generation configuration. Defaults to None, which uses model's default.
            massage_schema: Simplify schema. Defaults to True.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to None, which uses model's default.

        Raises:
            GenError: If an error occurred, for example an invalid JSON schema output error. See GenError.

        Returns:
            A dict from model's JSON response, following genconf.jsonschema, if provided.
        """        

        thread = Thread.ensure(query, inst)

        out = self.gen_json(json_schema,                            
                            thread,
                            genconf,
                            massage_schema,
                            schemaconf)
        
        GenError.raise_if_error(out,
                                ok_length_is_error=False) # as valid JSON can still be produced

        return out.dic # type: ignore[return-value]




    def dataclass(self, # noqa: E811
                  cls: Any, # a dataclass definition

                  query: Union[str,Thread],
                  *,
                  inst: Optional[str] = None,

                  genconf: Optional[GenConf] = None,
                  schemaconf: Optional[JSchemaConf] = None
                  ) -> Any: # a dataclass object
        """Constrained generation after a dataclass definition, resulting in an object initialized with the model's response.
        Raises GenError if unable to get a valid response that follows the dataclass definition.

        Args:
            cls: A dataclass definition.
            query: Thread or an str with the text of a single IN message to use as model input.
            inst: Instruction message for model. Will override Thread's inst, if set. Defaults to None.
            genconf: Model generation configuration. Defaults to None, which uses model's default.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to None, which uses model's default.

        Raises:
            GenError: If an error occurred, for example invalid object initialization. See GenError.

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






    def pydantic(self,
                 cls: Any, # a Pydantic BaseModel class

                 query: Union[str,Thread],
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
            query: Thread or an str with the text of a single IN message to use as model input.
            inst: Instruction message for model. Will override Thread's inst, if set. Defaults to None.
            genconf: Model generation configuration. Defaults to None, which uses model's default.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to None, which uses model's default.

        Raises:
            GenError: If an error occurred, for example an invalid BaseModel object. See GenError.

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







    def extract(self,
                target: Any,

                query: Union[str,Thread],
                *,
                inst: Optional[str] = None,

                genconf: Optional[GenConf] = None,
                schemaconf: Optional[JSchemaConf] = None
                ) -> Any:
        
        """Free type constrained generation: an instance of the given type will be initialized with the model's output.
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
            query: Thread or an str with the text of a single IN message to use as model input.
            inst: Instruction message for model. Will override Thread's inst, if set. Defaults to None.
            genconf: Model generation configuration. Defaults to None, which uses model's default.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to None, which uses model's default.

        Raises:
            GenError: If an error occurred, for example invalid object initialization. See GenError.

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




    def classify(self,
                 labels: Any,

                 query: Union[str,Thread],
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
            query: Thread or an str with the text of a single IN message to use as model input.
            inst: Instruction message for model. Will override Thread's inst, if set. Defaults to None.
            genconf: Model generation configuration. Defaults to None, which uses model's default.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to None, which uses model's default.

        Raises:
            GenError: If an error occurred. See GenError.

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











    
    # ======================================================== properties

     

    @abstractmethod
    def token_len(self,
                  thread: Thread) -> int:
        """Calculate token length for a Thread.

        Args:
            thread: For token length calculation.

        Returns:
            Number of tokens the thread will use.
        """
        ...

    
    @property
    def ctx_len(self) -> int:
        """Maximum context length, shared for input + output.
        We assume a common in+out context where total token length must always be less than this number."""
        return self._ctx_len


    @classmethod
    def version(cls) -> str:
        """Sibila version + provider version
        Ex: sibila='0.2.3' provider='llama-cpp-python 0.2.44'        
        """        
        from .__init__ import __version__  # type: ignore[import-not-found]
        return f"sibila='{__version__}' provider='{cls.provider_version()}'"
        
    @classmethod
    @abstractmethod
    def provider_version(cls) -> str:
        """Provider library version: provider x.y.z
        Ex. llama-cpp-python 0.2.44
        """
        ...


    @property
    def desc(self) -> str:
        """Model description."""
        return "Unknown model desc"
    
    def info(self) -> str:
        """Model object information."""
        return f"desc='{self.desc}'," \
               f"ctx_len={self.ctx_len},\n" \
               f"genconf={self.genconf}"
               
    
    def __str__(self):
        return self.info()





    
    
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = internal
    
    def _prepare_gen_in(self,
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
            raise ValueError("Cannot generate from empty thread")
        
        if thread.last_kind != MsgKind.IN:
            logger.warn("Last thread message is not an IN message.")

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

                    first_text = thread.get_text(0)
                    first_text = thread.join_text(first_text, text, inst["sep_count"]) # will separate by join_sep * sep_count
                    thread.set_text(0, first_text)
                    
                    logger.debug(f'Appended {"json" if genconf.json_schema is None else "json_schema"} instruction to first message.')

            
        logger.debug(f"Prepare: {str(thread)}")
        
        return thread


    
    
    def _prepare_gen_out(self,
                         text: str, 
                         finish: str,
                         genconf: GenConf) -> GenOut:
        """Perform common operations over the generated model response.

        Args:
            text: Text obtained from model.
            finish: Model finish reason - only "stop", "length" are used, others will map as GenRes.ERROR_MODEL.

        Returns:
            A GenOut object with result, generated text, etc. 
        """

        logger.debug(f"Response {finish}: █{text}█")
        
        if text is None:
            text = ''
        else:
            text = text.strip()

        if genconf.format == "json":
            if "\\u" in text:
                # dumps(with default ensure_ascii=False) -> ascii (subset of utf-8) -> text.encode("latin1") -> latin1 ->
                #   decode("unicode-escape") -> utf-8
                text = text.encode("latin1").decode("unicode-escape")
                
            try:
                dic = json.loads(text)

                if genconf.json_schema is not None:
                    if isinstance(genconf.json_schema, str):
                        schema_dic = json.loads(genconf.json_schema)
                    else:
                        schema_dic = genconf.json_schema
                        
                    json_schema_validate(dic, schema_dic)

                out = GenOut(res=GenRes.from_finish_reason(finish),
                             text=text,
                             dic=dic)
            
            except json.JSONDecodeError:
                out = GenOut(res=GenRes.from_finish_reason("!json"),
                             text=text)
                
            except json_schema_ValidationError as err:
                logger.info(f"JSON schema validation error: {err.message}")
                out = GenOut(res=GenRes.from_finish_reason("!json_schema_val"),
                             text=text,
                             dic=dic)
                
            except json_schema_SchemaError as err:
                logger.info(f"JSON schema error: {err.message}")
                out = GenOut(res=GenRes.from_finish_reason("!json_schema_error"),
                             text=text,
                             dic=dic)

        else:
            out = GenOut(res=GenRes.from_finish_reason(finish),
                         text=text)
            
       
        return out
    










class TextModel(Model, ABC):
    """Model with text-based input/output."""

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

    
    def token_len(self,
                  thread: Thread,
                  _: Optional[GenConf] = None) -> int:
        """Calculate token length for a Thread.

        Args:
            thread: For token length calculation.

        Returns:
            Number of tokens the thread will use.
        """

        text = self.text_from_thread(thread)
        return self.tokenizer.token_len(text)

   
    
    @abstractmethod
    def text_from_thread(self,
                         thread: Thread) -> str:
        ...
    







from .models import Models


class FormattedTextModel(TextModel, ABC):
    """Model that uses formatted text (chat templates) for input/output."""
    
    format: Union[dict,None]
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

        self.format = None
        self._jinja_compiled_template = None
        
    
    def init_format(self,
                    format: Union[str,dict,None],
                    format_search_order: list,
                    info: dict):
        """
        format_search_order is a flags list:
            "name: match by name
            "meta_template": use model file metadata's template
            ex: ["name","meta_template"]
        
        info fields: {
            "name": name, # ex: filename
            "meta_template_name": "chat_template"
        }
        """

        # handle chat template format
        def search_format() -> dict:

            for order in format_search_order:

                if order == "name":
                    name = info["name"]
                    fmt = Models.search_format(name)
                    if fmt is not None:
                        return fmt
                        
                elif order == "meta_template":
                    md = self.get_metadata()
                    key_name = info["meta_template_name"]
                    if key_name in md:
                        logger.debug(f"Format from model file metadata ('{key_name}') template='{md[key_name]}'")
                        fmt = {
                            "template": md[key_name]
                        }
                        return fmt
            
            raise ValueError("Could not find a suitable format (chat template) for this model. Without a format, fine-tuned models (models with chat, instruct, etc in their names) cannot properly answer queries. You can pass a template string in the format arg when creating this model. It is a Jinja template which you can locate in the internet: try searching by the model name + 'chat template'.")

        
        if format is not None:
            if isinstance(format, str):
                if '{{' in format: # an str with a jinja template
                    self.format = {"template": format}
                    
                else: # a format name
                    self.format = Models.get_format(format)

            elif (isinstance(format, dict) and
                  "template" in format and 
                  '{{' in format["template"]): # a dict with at least a template key with a jinja template
                self.format = format.copy()
                
            else:
                raise TypeError("format arg can only be str, dict or None")
                
        if self.format is None:
            self.format = search_format() # will raise if unable to find

        # setup jinja template
        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        jinja_env.globals["raise_exception"] = jinja_raise_exception
        self._jinja_compiled_template = jinja_env.from_string(self.format["template"])



    
    def gen(self, 
            thread: Thread,
            genconf: Optional[GenConf] = None,
            ) -> GenOut:
        """Text generation from a Thread, used by the other model generation methods.
        Doesn't raise an exception if an error occurs, always returns GenOut.

        Args:
            thread: The Thread object to use as model input.
            genconf: Model generation configuration. Defaults to None, which uses model's default.

        Returns:
            A GenOut object with result, generated text, etc. 
        """

        if genconf is None:
            genconf = self.genconf

        thread = self._prepare_gen_in(thread, genconf)

        prompt = self.text_from_thread(thread)
        
        logger.debug(f"Prompt: █{prompt}█")

        text,finish = self._gen_text(prompt, genconf)

        out = self._prepare_gen_out(text, finish, genconf)

        return out


    
    
    @abstractmethod
    def _gen_text(self,
                  text: str,
                  genconf: GenConf) -> tuple[str,str]:
        """Generate from formatted text.

        Args:
            text: Formatted text (from input Thread).
            genconf: Model generation configuration.

        Returns:
            Tuple of strings: generated_text, finish_reason.
        """
        ...


    
    def text_from_thread(self,
                         thread: Thread) -> str:

        messages = thread.as_chatml()
        text = self._jinja_compiled_template.render(messages=messages, # type: ignore[union-attr]
                                                    add_generation_prompt=True,
                                                    **self.tokenizer.special_tokens_map())
        return text
        

    
    
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
                

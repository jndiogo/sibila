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


from jsonschema import (
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

from .json_utils import (
    JSchemaConf,
    json_schema_massage,
    json_schema_from_pydantic,
    pydantic_obj_from_json,
    get_type,
    get_type_list,
    build_type_json_schema,
    build_array_type_json_schema,
    build_object_type_json_schema
)

from .dictype import json_schema_from_dictype

from .utils import is_subclass_of



    



class Tokenizer(ABC):
    """Base tokenizer class to encode and decode tokens, measure text length in tokens, track special tokens."""

    bos_token_id: int # beginning of sentence
    bos_token: str

    eos_token_id: int # end of sentence
    eos_token: str

    pad_token_id: int # padding
    pad_token: str

    unk_token_id: int # unknown token
    unk_token: str

    
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
                  text: str) -> list[int]:
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
                 tokenizer: Tokenizer):
        """_summary_

        Args:
            is_local_model: Is the model running locally?
            genconf: Generation configuration options used in gen().
            tokenizer: Tokenizer used to encode text (even for message-based models).
        """
        
        self.is_local_model = is_local_model
        
        self._ctx_len = 0

        self.tokenizer = tokenizer

        if genconf is None:
            self.genconf = GenConf()
        else:
            self.genconf = genconf.clone()
        

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
            genconf: Model generation configuration. Defaults to None.

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

                 massage_schema: Optional[bool] = True,
                 schemaconf: Optional[JSchemaConf] = None,
                 ) -> GenOut:
        """JSON/JSON-schema grammar-constrained generation, returning a Python dict of values, constrained or not by a JSON schema.
        Doesn't raise an exception if an error occurs, always returns GenOut.

        Args:
            json_schema: A JSON schema describing the dict fields that will be output. None means no schema (free JSON output).
            thread: The Thread to use as model input.
            genconf: Model generation configuration. Defaults to None.
            massage_schema: Simplify schema. Defaults to True.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to None.

        Returns:
            A GenOut object with result, generated text, etc. The output dict is in GenOut.dic.
        """

        if genconf is None:
            genconf = self.genconf

        if genconf.json_schema is not None and json_schema is not None:
            logger.warn("Both json_schema and genconf.json_schema are set: using json_schema")

        if json_schema is not None and massage_schema:
            if schemaconf is None:
                schemaconf = JSchemaConf()
            json_schema = json_schema_massage(json_schema, schemaconf)

        out = self.gen(thread, 
                       genconf(format="json", 
                               json_schema=json_schema))
        
        return out        
        


    
    
    def gen_dictype(self,
                    dictype: dict,
                    thread: Thread,
                    genconf: Optional[GenConf] = None,
                    schemaconf: Optional[JSchemaConf] = None
                    ) -> GenOut:
        """Grammar-constrained generation after a dictype definition (see dictype.py), returning a Python dict of values, 
        Doesn't raise an exception if an error occurs, always returns GenOut.

        Args:
            dictype: A dictype defining the layout of the returned dict.
            thread: The Thread to use as model input.
            genconf: Model generation configuration. Defaults to None.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to None.

        Returns:
            A GenOut object with result, generated text, etc, with the output dict in GenOut.dic.
        """

        if isinstance(dictype, dict):
            params_schema = json_schema_from_dictype(dictype)
        else:
            raise TypeError("Arg dictype must be of dict type")

        if genconf is None:
            genconf = self.genconf

        out = self.gen_json(params_schema,
                            thread,
                            genconf,
                            massage_schema=True,
                            schemaconf=schemaconf) 
        return out


    
    
    
    def gen_pydantic(self,
                     cls: Any, # a Pydantic BaseModel class
                     thread: Thread,
                     genconf: Optional[GenConf] = None,
                     schemaconf: Optional[JSchemaConf] = None
                     ) -> GenOut:
        """Grammar-constrained generation after a Pydantic BaseModel-derived class definition.
        An initialized Pydantic BaseModel object is returned in the "obj" field of return dict.
        Doesn't raise an exception if an error occurs, always returns GenOut containing the created object.

        Args:
            cls: A class derived from a Pydantic BaseModel class.
            thread: The Thread to use as model input.
            genconf: Model generation configuration. Defaults to None.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to None.

        Returns:
            A GenOut object with result, generated text, etc. The initialized Pydantic BaseModel-derived object is in GenOut.obj.
        """

        if is_subclass_of(cls, BaseModel):
            schema = json_schema_from_pydantic(cls)
        else:
            raise TypeError("Only pydantic BaseModel allowed for argument cls")


        if genconf is None:
            genconf = self.genconf

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
                out.obj = obj
                
            except TypeError as e:
                out.res = GenRes.ERROR_JSON_SCHEMA_VAL # error validating for object (by Pydantic), but JSON is valid for its schema
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
                 ok_length_is_error: Optional[bool] = False
                 ) -> str:
        """Text generation from a Thread, used by the other model generation methods.

        Args:
            query: Thread or an str with the text of a single IN message to use as model input.
            inst: Instruction message for model. Will override Thread's inst, if set. Defaults to None.
            genconf: Model generation configuration. Defaults to None.
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
             massage_schema: Optional[bool] = True,
             schemaconf: Optional[JSchemaConf] = None,
             ) -> dict:
        """JSON/JSON-schema grammar-constrained generation, returning a Python dict of values, constrained or not by a JSON schema.
        Raises GenError if unable to get a valid/schema-validated JSON.

        Args:
            json_schema: A JSON schema describing the dict fields that will be output. None means no schema (free JSON output).
            query: Thread or an str with the text of a single IN message to use as model input.
            inst: Instruction message for model. Will override Thread's inst, if set. Defaults to None.
            genconf: Model generation configuration. Defaults to None.
            massage_schema: Simplify schema. Defaults to True.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to None.

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

        return out.dic





    def dictype(self,
                dictype: dict,

                query: Union[str,Thread],
                *,
                inst: Optional[str] = None,

                genconf: Optional[GenConf] = None,
                schemaconf: Optional[JSchemaConf] = None
                ) -> dict:
        """Grammar-constrained generation after a dictype definition (see dictype.py), returning a Python dict of values, 
        Raises GenError if unable to get a valid dict that follows the dictype definition.

        Args:
            dictype: A dictype definition stating the types of the generated dict fields.
            query: Thread or an str with the text of a single IN message to use as model input.
            inst: Instruction message for model. Will override Thread's inst, if set. Defaults to None.
            genconf: Model generation configuration. Defaults to None.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to None.

        Raises:
            GenError: If an error occurred, for example an invalid JSON schema output error. See GenError.

        Returns:
            A dict following the dictype definition.
        """

        thread = Thread.ensure(query, inst)

        out = self.gen_dictype(dictype,
                               thread,
                               genconf,
                               schemaconf)
        
        GenError.raise_if_error(out,
                                ok_length_is_error=False) # as valid JSON can still be produced

        return out.dic





    def pydantic(self,
                 cls: Any, # a Pydantic BaseModel class

                 query: Union[str,Thread],
                 *,
                 inst: Optional[str] = None,

                 genconf: Optional[GenConf] = None,
                 schemaconf: Optional[JSchemaConf] = None
                 ) -> Any: # a Pydantic BaseModel object
        """Grammar-constrained generation after a Pydantic BaseModel-derived class definition.
        An initialized Pydantic BaseModel object is returned in the "obj" field of return dict.
        Raises GenError if unable to get a valid dict that follows the BaseModel class definition.

        Args:
            cls: A class derived from a Pydantic BaseModel class.
            query: Thread or an str with the text of a single IN message to use as model input.
            inst: Instruction message for model. Will override Thread's inst, if set. Defaults to None.
            genconf: Model generation configuration. Defaults to None.
            schemaconf: JSchemaConf object that controls schema simplification. Defaults to None.

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

        return out.obj




    def extract(self,
                target: Any,

                query: Union[str,Thread],
                *,
                inst: Optional[str] = None,

                genconf: Optional[GenConf] = None,
                schemaconf: Optional[JSchemaConf] = None
                ) -> Any:
        
        """
        prim_type: can be Annotated[T, "Desc"]
            bool
            int
            float
            str
            
        enums: can be Annotated[T, "Desc"]
            Literal['year', 'name'] - all items of the same type:  int, float, str
            Enum, EnumInt, EnumStr - int, float, str
            [1, 2, 3] or ["a","b"] - all items of the same type:  int, float, str

        Datetime        

        list of values of a prim_type, BaseModel, dictype: 
            list[prim_type] - for example list[int]
            can be Annotated[list[T], "List desc"] and/or list[Annotated[T, "Item desc"]]

        dict -> dictype

        Pydantic -> BaseModel obj
            
        """

        OUTPUT_KEY = "output"

        thread = Thread.ensure(query, inst)


        # type list
        type_, list_desc, item_desc = get_type_list(target)
        if type_ is not None:
            
            # build json schema for list of type_
            items_repr = build_simple_type_json_schema(type_, item_desc)

            array_repr = build_array_type_json_schema(items_repr, list_desc)

            json_schema = build_object_type_json_schema({OUTPUT_KEY: array_repr})

        else:

            type_, item_desc, enum_list = get_type(target, 
                                                   allow_enums=True,
                                                   allow_BaseModel=True,
                                                   allow_dictype=True)
            if type_ is not None:
                item_repr = build_type_json_schema(type_, 
                                                   item_desc,
                                                   enum_list)

                json_schema = build_object_type_json_schema({OUTPUT_KEY: item_repr})

            else:
                raise TypeError(f"Unknown target type '{target}'")


        # arriving here, json_schema was built: gen with it
        out = self.json(json_schema,
                        query,
                        inst=inst,
                        genconf=genconf,
                        massage_schema=True,
                        schemaconf=schemaconf)
        
        value = out[OUTPUT_KEY]

        return value



    
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
        We assume a common in+out context where total token length must always be less than this number.
        """
        return self._ctx_len


    @classmethod
    def version(cls) -> str:
        """Sibila version + provider version
        Ex: sibila='0.2.3' provider='llama-cpp-python 0.2.44'        
        """        
        from .__init__ import __version__
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
                 genconf: GenConf,
                 tokenizer: Tokenizer):

        super().__init__(is_local_model,
                         genconf,
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
    







from .formatdir import (
    FormatDir
)

class FormattedTextModel(TextModel, ABC):
    """Model that uses formatted text (chat templates) for input/output."""
    
    def __init__(self,
                 is_local_model: bool,
                 genconf: GenConf,
                 tokenizer: Tokenizer):

        super().__init__(is_local_model,
                         genconf,
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
                    fmt = FormatDir.search(name)
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
                    self.format = FormatDir.get(format)

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
        Doesn't raise an exception if an error occurs, always returns a result dict.

        Args:
            thread: The Thread object to use as model input.
            genconf: Model generation configuration. Defaults to None.

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
        text = self._jinja_compiled_template.render(messages=messages, 
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
                 genconf: GenConf,
                 tokenizer: Tokenizer):

        super().__init__(is_local_model,
                         genconf,
                         tokenizer)
        
        self.is_message_model = True
                



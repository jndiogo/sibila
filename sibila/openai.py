"""OpenAI remote model access.
Using the chat completion API:
https://platform.openai.com/docs/api-reference/chat/create

- OpenAIModel: Access OpenAI models.
- OpenAITokenizer: Tokenizer for OpenAI models.
"""


from typing import Any, Optional, Union
import sys, json
from time import time 
from copy import copy

import logging
logger = logging.getLogger(__name__)


from .gen import (
    GenConf,
    GenOut
)

from .thread import (
    Thread,
    Msg
)

from .model import (
    MessagesModel,
    Tokenizer
)

from .json_schema import JSchemaConf

try:
    import openai
    has_openai = True
except ImportError:
    has_openai = False
    
try:
    import tiktoken
    has_tiktoken = True
except ImportError:
    has_tiktoken = False




class OpenAIModel(MessagesModel):
    """Access an OpenAI model.

    Supports constrained JSON output, via the OpenAI API tools mechanism.

    Ref:
        https://platform.openai.com/docs/api-reference/chat/create
    """

    PROVIDER_NAME:str = "openai"
    """Provider prefix that this class handles."""

    _token_estimation_factor: float
    """Multiplication factor to estimate token usage: multiplies text length to obtain token length."""

    DEFAULT_TOKEN_ESTIMATION_FACTOR: float = 0.4
    """Default factor for token_estimation_factor."""



    def __init__(self,
                 name: str,
                 *,
                 
                 # common base model args
                 genconf: Optional[GenConf] = None,
                 schemaconf: Optional[JSchemaConf] = None,
                 ctx_len: Optional[int] = None,
                 max_tokens_limit: Optional[int] = None,
                 tokenizer: Optional[Tokenizer] = None,
                
                 # most important OpenAI-specific args
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 overhead_per_msg: Optional[int] = None,
                 token_estimation_factor: Optional[float] = None,
                 create_tokenizer: bool = False,
                 
                 # other OpenAI-specific args
                 other_init_kwargs: dict = {},
                 ):
        """Create an OpenAI remote model.

        Args:
            name: Model name to resolve into an existing model.
            genconf: Model generation configuration. Defaults to None.
            schemaconf: Default configuration for JSON schema validation, used if generation call doesn't supply one. Defaults to None.
            ctx_len: Maximum context length to be used (shared for input and output). None for model's default.
            max_tokens_limit: Maximum output tokens limit. None for model's default.
            tokenizer: An external initialized tokenizer to use instead of the created from the GGUF file. Defaults to None.
            api_key: OpenAI API key. Defaults to None, which will use env variable OPENAI_API_KEY.
            base_url: Base location for API access. Defaults to None, which will use env variable OPENAI_BASE_URL or a default.
            overhead_per_msg: Overhead tokens to account for when calculating token length. None for model's default.
            token_estimation_factor: Used when no tokenizer is available. Multiplication factor to estimate token usage: multiplies total text length to obtain token length.
            create_tokenizer: When no tokenizer is passed, should try to create one?
            other_init_kwargs: Extra args for OpenAI.OpenAI() initialization. Defaults to {}.

        Raises:
            ImportError: If OpenAI API is not installed.
            NameError: If model name was not found or there's an API or authentication problem.
        """


        if not has_openai:
            raise ImportError("Please install openai by running: pip install openai")

        self._client = self._client_async = None


        # also accept "provider:name" for ease of use
        provider_name = self.PROVIDER_NAME + ":"
        if name.startswith(provider_name):
            name = name[len(provider_name):]

        super().__init__(False,
                         genconf,
                         schemaconf,
                         tokenizer
                         )

        if (ctx_len is not None and
            max_tokens_limit is not None and
            overhead_per_msg is not None and
            token_estimation_factor is not None): # all elements given: probably created via Models.create()

            self._model_name = name
            default_ctx_len = ctx_len
            default_max_tokens_limit = max_tokens_limit
            default_overhead_per_msg = overhead_per_msg
            default_token_estimation_factor = token_estimation_factor
        
        else: # need to resolve
            settings = self.resolve_settings(self.PROVIDER_NAME,
                                             name,
                                             ["name", 
                                              "ctx_len", 
                                              "max_tokens_limit", 
                                              "overhead_per_msg",
                                              "token_estimation_factor"])
            self._model_name = settings.get("name") or name
            default_ctx_len = settings.get("ctx_len") # type: ignore[assignment]
            default_max_tokens_limit = settings.get("max_tokens_limit") # type: ignore[assignment]
            default_overhead_per_msg = settings.get("overhead_per_msg") # type: ignore[assignment]
            default_token_estimation_factor = settings.get("token_estimation_factor") # type: ignore[assignment]
            
            # all defaults are conservative values
            if ctx_len is None and default_ctx_len is None:
                default_ctx_len = 4096
                logger.warning(f"Model '{self._model_name}': unknown ctx_len, assuming {default_ctx_len}")

            if max_tokens_limit is None and default_max_tokens_limit is None:
                default_max_tokens_limit = ctx_len or default_ctx_len                
                # don't warn: assume equal to ctx_len: logger.warning(f"Model '{self._model_name}': unknown max_tokens_limit, assuming {default_max_tokens_limit}")

            if overhead_per_msg is None and default_overhead_per_msg is None:
                default_overhead_per_msg = 3
                # don't warn for this setting due to derived model classes (none uses it)

            if token_estimation_factor is None and default_token_estimation_factor is None:
                default_token_estimation_factor = self.DEFAULT_TOKEN_ESTIMATION_FACTOR
                logger.warning(f"Model '{self._model_name}': unknown token_estimation_factor, assuming {default_token_estimation_factor}")


        self.ctx_len = ctx_len or default_ctx_len
        
        self.max_tokens_limit = max_tokens_limit or default_max_tokens_limit
        self.max_tokens_limit = min(self.max_tokens_limit, self.ctx_len)

        self._overhead_per_msg = overhead_per_msg or default_overhead_per_msg

        self._token_estimation_factor = token_estimation_factor or default_token_estimation_factor

        self.maybe_image_input = True # True means maybe - always check model specs

        # only check for "json" text presence as json schema (including field descriptions) is requested with the tools facility.
        self.json_format_instructors["json_schema"] = self.json_format_instructors["json"]


        if self.tokenizer is None and create_tokenizer:
            try:
                self.tokenizer = OpenAITokenizer(self._model_name)
            except Exception as e:
                logger.warning(f"Could not create a local tokenizer for model '{self._model_name}' - "
                               "token length calculation will be disabled and assume defaults. "
                               "To support recent OpenAI models, install the latest tiktoken version with 'pip install -U tiktoken'. "
                               f"Internal error: {e}")


        self._client_init_kwargs = other_init_kwargs
        if api_key is not None:
            self._client_init_kwargs["api_key"] = api_key
        if base_url is not None:
            self._client_init_kwargs["base_url"] = base_url



    def close(self):
        """Close model, release resources like memory or net connections."""
        self._client = self._client_async = self.tokenizer = None



    def _ensure_client(self,
                       is_async: bool):
        if is_async and self._client_async is not None:
            return
        elif not is_async and self._client is not None:
            return
            
        try:
            logger.debug(f"Creating inner OpenAI with ctx_len={self.ctx_len}, max_tokens_limit={self.max_tokens_limit}, "
                         f"_overhead_per_msg={self._overhead_per_msg}, _token_estimation_factor={self._token_estimation_factor}, "
                         f"init_kwargs={self._client_init_kwargs}")

            if is_async:
                self._client_async = openai.AsyncOpenAI(**self._client_init_kwargs) # type: ignore[assignment]
            else:
                self._client = openai.OpenAI(**self._client_init_kwargs) # type: ignore[assignment]

        except Exception as e:
            raise NameError(f"Could not create {'async' if is_async else ''} model '{self._model_name}' with error: {e}")
                








    def _gen_pre(self, 
                 thread: Thread,
                 genconf: Union[GenConf, None]
                 ) -> tuple:

        if genconf is None:
            genconf = self.genconf

        thread = self._prepare_gen_thread(thread, genconf)

        # This provider doesn't require "max_tokens" but will error on excess.
        if genconf.max_tokens != 0:
            token_len = self.token_len(thread, genconf)
            resolved_max_tokens = self.resolve_genconf_max_tokens(token_len, genconf)
            # for reference: next commented-out line is a bad idea, as endpoint will error if max_tokens is greater than limit:
            # resolved_max_tokens = self.resolve_genconf_max_tokens(0, genconf)
        else: # for genconf.max_tokens=0, this provider doesn't require "max_tokens", so we don't send below
            resolved_max_tokens = 0


        json_kwargs: dict = {}
        if genconf.format == "json":
            
            json_kwargs["response_format"] = {"type": "json_object"}

            if genconf.json_schema is not None:
                # use json_schema in OpenAi's tool API
                json_kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": self.output_fn_name},
                }

                if isinstance(genconf.json_schema, str):
                    params = json.loads(genconf.json_schema)
                else:
                    params = genconf.json_schema
                
                json_kwargs["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": self.output_fn_name,
                            "parameters": params
                        }
                    }
                ]

        # seed config is disabled, remote models and some hardware accelerated local models don't support it.
        # seed = genconf.seed
        # if seed == -1:
        #    seed = int(time())
        #    logger.debug(f"OpenAI random seed={seed}")

        
        msgs = thread.as_chatml()

        kwargs = {"model": self._model_name,
                  "messages": msgs, # type: ignore[arg-type]                  
                  "temperature": genconf.temperature,
                  "top_p": genconf.top_p,
                  # "seed": seed,
                  "n": 1,
                  **json_kwargs}
        
        if genconf.stop: # OpenAI API: empty stop errors when generating from images
            kwargs["stop"] = genconf.stop

        if resolved_max_tokens:
            kwargs["max_tokens"] = resolved_max_tokens

        # inject model-specific args, if any
        kwargs.update(genconf.resolve_special(self.PROVIDER_NAME))

        logger.debug(f"{type(self).__name__} gen args: {kwargs}")

        return (kwargs, genconf)
        

    def _gen_post(self, 
                  response: Any,
                  pre_kwargs: dict,
                  genconf: GenConf
                  ) -> GenOut:
            
        logger.debug(f"OpenAI response: {response}")

        choice = response.choices[0]
        finish = choice.finish_reason
        # OpenAI-compatible provider endpoints can give non-standard finish_reason values. Map as needed:
        if finish in ["eos", "tool_calls"]: finish = "stop"
        message = choice.message

        if "tool_choice" in pre_kwargs:
            
            # json schema generation via the tools API:
            if message.tool_calls is not None:
                if len(message.tool_calls) != 1:
                    logger.warn(f"OpenAIModel: expecting single message.tool_calls, but received {len(message.tool_calls)} - using first.")

                fn = message.tool_calls[0].function
                if fn.name != self.output_fn_name:
                    logger.warn(f"OpenAIModel: expecting '{self.output_fn_name}' function name, received ({fn.name})")

                text = fn.arguments

            else: # use content instead
                logger.warn("OpenAIModel: expecting message.tool_calls, but none received - using text content")
                text = message.content # type: ignore[assignment]
        
        else:
            # text or simple json format
            text = message.content # type: ignore[assignment]

        return self._prepare_gen_out(text, finish, genconf)







    
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
            RuntimeError: If unable to generate.

        Returns:
            A GenOut object with result, generated text, etc.
            The output text is in GenOut.text.
        """

        genconf2: GenConf
        kwargs, genconf2 = self._gen_pre(thread, genconf)

        self._ensure_client(False)

        try:
            # https://platform.openai.com/docs/api-reference/chat/create
            response = self._client.chat.completions.create(**kwargs) # type: ignore[attr-defined]

        except Exception as e:
            raise RuntimeError(f"Cannot generate. Internal error: {e}")


        return self._gen_post(response,
                              kwargs,
                              genconf2)
    



    async def gen_async(self, 
                        thread: Thread,
                        genconf: Optional[GenConf] = None,
                        ) -> GenOut:
        """Async text generation from a Thread, used by the other (async) generation methods.
        Doesn't raise an exception if an error occurs, always returns GenOut.

        Args:
            thread: The Thread to use as model input.
            genconf: Model generation configuration. Defaults to None.

        Raises:
            RuntimeError: If unable to generate.

        Returns:
            A GenOut object with result, generated text, etc.
            The output text is in GenOut.text.
        """

        genconf2: GenConf
        kwargs, genconf2 = self._gen_pre(thread, genconf)

        self._ensure_client(True)

        try:
            # https://platform.openai.com/docs/api-reference/chat/create
            response = await self._client_async.chat.completions.create(**kwargs) # type: ignore[attr-defined]

        except Exception as e:
            raise RuntimeError(f"Cannot generate. Internal error: {e}")

        return self._gen_post(response,
                              kwargs,
                              genconf2)
    

   





    def token_len(self,
                  thread_or_text: Union[Thread,str],
                  genconf: Optional[GenConf] = None) -> int:
        """Calculate or estimate the token length for a Thread or a plain text string.
        In some cases where it's not possible to calculate the exact token count, 
        this function should give a conservative (upper bound) estimate.
        It's up to the implementation whether to account for side information like JSON Schema,
        but it must reflect the model's context token accounting.
        Thread or text must be the final text which will passed to model.

        If a json_schema is provided in genconf, we use its string's token_len as upper bound for the extra prompt tokens.
        
        From https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        
        More info on calculating function_call (and tools?) tokens:

        https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/24
        
        https://gist.github.com/CGamesPlay/dd4f108f27e2eec145eedf5c717318f5

        Args:
            thread_or_text: For token length calculation.
            genconf: Model generation configuration. Defaults to None.

        Returns:
            Estimated number of tokens used.
        """

        if isinstance(thread_or_text, Thread):
            thread = thread_or_text            
        else:
            thread = Thread.make_IN(thread_or_text)

        num_tokens = 0

        if self.tokenizer is None: # no tokenizer was found, so we'll have to do a conservative estimate

            OVERHEAD_PER_MSG = 3
            for msg in thread.get_iter(True): # True for system message
                message = msg.as_chatml()
                msg_tokens = len(str(message["content"])) * self._token_estimation_factor + OVERHEAD_PER_MSG
                # str(message["content"]): hacky way to deal with dict "content" key
                num_tokens += int(msg_tokens)

            if genconf is not None and genconf.json_schema is not None:
                if isinstance(genconf.json_schema, str):
                    js_str = genconf.json_schema
                else:
                    js_str = json.dumps(genconf.json_schema)

                tools_num_tokens = len(js_str) * self._token_estimation_factor
                num_tokens += int(tools_num_tokens)
                # print("tools_num_tokens", tools_num_tokens)

        else: # do an "informed" token estimation from what is known of the OpenAI model's tokenization
            
            for msg in thread.get_iter(True): # True for system message
                message = msg.as_chatml()
                # print(message)
                num_tokens += self._overhead_per_msg
                for key, value in message.items():
                    num_tokens += len(self.tokenizer.encode(str(value))) # str(value): hacky way to deal with dict "content" key
            
            # add extras + every reply is primed with <|start|>assistant<|message|>
            num_tokens += 32

            # print("text token_len", num_tokens)

            if genconf is not None and genconf.json_schema is not None:
                TOOLS_TOKEN_LEN_FACTOR = 1.2

                if isinstance(genconf.json_schema, str):
                    js_str = genconf.json_schema
                else:
                    js_str = json.dumps(genconf.json_schema)

                tools_num_tokens = self.tokenizer.token_len(js_str)

                # this is an upper bound, as empirically tested with the api.
                tools_num_tokens = int(tools_num_tokens * TOOLS_TOKEN_LEN_FACTOR)
                # print("tools token_len", tools_num_tokens)

                num_tokens += tools_num_tokens

        
        return num_tokens



    @classmethod
    def known_models(cls,
                     api_key: Optional[str] = None) -> Union[list[str], None]:
        """List of model names that can be used. Some of the models are not chat models and cannot be used,
        for example embedding models.
        
        Args:
            api_key: Requires OpenAI API key, passed as this arg or set in env variable OPENAI_API_KEY.

        Returns:
            Returns a list of known models.
        """
 
        client = openai.OpenAI(api_key=api_key)
        model_list = client.models.list()

        out = []
        for model in model_list.data:
            out.append(model.id)
        return sorted(out)





    def name(self) -> str:
        """Model (short) name."""
        return self._model_name
    
    def desc(self) -> str:
        """Model description."""
        return f"{type(self).__name__}: '{self._model_name}'"


    @classmethod
    def provider_version(_) -> str:
        """Provider library version: provider x.y.z
        Ex. openai-1.3.6
        """
        try:        
            import openai
            ver = openai.__version__
        except Exception:
            raise ImportError("Please install openai by running: pip install openai")
            
        return f"openai-{ver}"
    
    








class OpenAITokenizer(Tokenizer):
    """Tokenizer for OpenAI models."""

    def __init__(self, 
                 model: str
                 ):

        if not has_tiktoken:
            raise Exception("Please install tiktoken by running: pip install tiktoken")
        
        self._tok = tiktoken.encoding_for_model(model)

        self.vocab_size = self._tok.n_vocab
        
        self.bos_token_id = None
        self.bos_token = None

        self.eos_token_id = None
        self.eos_token = None

        self.pad_token_id = None
        self.pad_token = None

        self.unk_token_id = None
        self.unk_token = None
  

    
    def encode(self, 
               text: str) -> list[int]:
        """Encode text into model tokens. Inverse of Decode().

        Args:
            text: Text to be encoded.

        Returns:
            A list of ints with the encoded tokens.
        """
        return self._tok.encode(text)

        
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
        assert skip_special, "OpenAITokenizer only supports skip_special=True"

        return self._tok.decode(token_ids)

   





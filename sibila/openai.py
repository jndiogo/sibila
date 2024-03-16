"""OpenAI remote model access.

- OpenAIModel: Use OpenAI API to access their models.
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
    MsgKind
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
    Ref: https://platform.openai.com/docs/api-reference/chat/create

    Attributes:
        ctx_len: Maximum context length, shared for input + output.
        desc: Model information.
    """

    TOOLS_TOKEN_LEN_FACTOR: float
    """Multiplication factor to use for tools section token length estimation."""

    DEFAULT_TOOLS_TOKEN_LEN_FACTOR: float = 1.2

    def __init__(self,
                 name: str,
                 *,
                 
                 # common base model args
                 genconf: Optional[GenConf] = None,
                 schemaconf: Optional[JSchemaConf] = None,
                 tokenizer: Optional[Tokenizer] = None,
                 ctx_len: Optional[int] = None,
                 max_tokens_limit: Optional[int] = None,
                
                 # most important OpenAI-specific args
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 overhead_per_msg: Optional[int] = None,
                 
                 # OpenAI-specific args
                 openai_init_kwargs: dict = {},
                 ):
        """Create an OpenAI remote model.
        Name resolution depends on unknown_name_mask and will keep removing letters 
        from the end of name and searching existing entries in penAIModel.known_models().

        Args:
            name: Model name to resolve into an existing model.
            genconf: Model generation configuration. Defaults to None.
            schemaconf: Default configuration for JSON schema validation, used if generation call doesn't supply one. Defaults to None.
            tokenizer: An external initialized tokenizer to use instead of the created from the GGUF file. Defaults to None.
            ctx_len: Maximum context length to be used (shared for input and output). None for model's default.
            max_tokens_limit: Maximum output tokens limit. None for model's default.
            api_key: OpenAI API key. Defaults to None, which will use env variable OPENAI_API_KEY.
            base_url: Base location for API access. Defaults to None, which will use env variable OPENAI_BASE_URL.
            overhead_per_msg: Overhead tokens to account for when calculating token length. None for model's default.
            openai_init_kwargs: Extra args for OpenAI.OpenAI() initialization. Defaults to {}.

        Raises:
            ImportError: If OpenAI API is not installed.
            NameError: If model name was not found or there's an API or authentication problem.
        """


        if not has_openai:
            raise ImportError("Please install openai by running: pip install openai")

        super().__init__(False,
                         genconf,
                         schemaconf,
                         tokenizer
                         )

        if (ctx_len is not None and
            max_tokens_limit is not None and
            overhead_per_msg is not None): # all elements given

            self._model_name = name
            default_ctx_len = ctx_len
            default_max_tokens_limit = max_tokens_limit
            default_overhead_per_msg = overhead_per_msg
        
        else: # need to resolve
            (self._model_name, 
             default_ctx_len, default_max_tokens_limit,
             default_overhead_per_msg) = self.resolve_settings("openai", name) # type: ignore[assignment]
            
            # all defaults are conservative values
            if default_ctx_len is None:
                default_ctx_len = 4096
                logger.warning(f"Model '{self._model_name}': unknown ctx_len, assuming {default_ctx_len}")
            if default_max_tokens_limit is None:
                default_max_tokens_limit = default_ctx_len
                logger.warning(f"Model '{self._model_name}': unknown max_tokens_limit, assuming {default_max_tokens_limit}")
            if default_overhead_per_msg is None:
                default_overhead_per_msg = 3
                logger.warning(f"Model '{self._model_name}': unknown overhead_per_msg, assuming {default_overhead_per_msg}")


        if not ctx_len: # None or 0
            self.ctx_len = default_ctx_len
        else:
            self.ctx_len = ctx_len
        
        if not max_tokens_limit:
            self.max_tokens_limit = default_max_tokens_limit
        else:
            self.max_tokens_limit = max_tokens_limit

        self.max_tokens_limit = min(self.max_tokens_limit, self.ctx_len)

        if overhead_per_msg is None:
            self._overhead_per_msg = default_overhead_per_msg
        else:
            self._overhead_per_msg = overhead_per_msg

        self.TOOLS_TOKEN_LEN_FACTOR = self.DEFAULT_TOOLS_TOKEN_LEN_FACTOR


        # only check for "json" text presence as json schema is requested with the tools facility.
        self.json_format_instructors["json_schema"] = self.json_format_instructors["json"]


        logger.debug(f"Creating inner OpenAI with ctx_len={self.ctx_len}, max_tokens_limit={self.max_tokens_limit}, _overhead_per_msg={self._overhead_per_msg}, base_url={base_url}, openai_init_kwargs={openai_init_kwargs}")


        try:
            if self.tokenizer is None:
                self.tokenizer = OpenAITokenizer(self._model_name)
        except Exception as e:
            raise NameError(f"Model not found for '{self._model_name}'. "
                            f"Internal error: {e}")


        try:
            self._client = openai.OpenAI(api_key=api_key,
                                         base_url=base_url,
                                         **openai_init_kwargs)
        except Exception as e:
            raise NameError(f"Could not create model '{self._model_name}' with error: {e}")
    





    
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

        if genconf is None:
            genconf = self.genconf

        thread = self._prepare_gen_in(thread, genconf)
        token_len = self.token_len(thread, genconf)


        # resolve max_tokens size
        resolved_max_tokens = self.resolve_genconf_max_tokens(token_len, genconf)


        fn_name = "json_out"

        json_kwargs: dict = {}
        format = genconf.format
        if format == "json":
            
            if genconf.json_schema is None:
                json_kwargs["response_format"] = {"type": "json_object"}

            else:
                # use json_schema in OpenAi's tool API
                json_kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": fn_name},
                }

                if isinstance(genconf.json_schema, str):
                    params = json.loads(genconf.json_schema)
                else:
                    params = genconf.json_schema
                
                json_kwargs["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "parameters": params
                        }
                    }
                ]

            logger.debug(f"OpenAI json args: {json_kwargs}")

        # seed config is disabled, has remote models and some hardware accelerated local models don't support it.
        # seed = genconf.seed
        # if seed == -1:
        #    seed = int(time())
        #    logger.debug(f"OpenAI random seed={seed}")

        
        msgs = thread.as_chatml()

        try:
            # https://platform.openai.com/docs/api-reference/chat/create
            response = self._client.chat.completions.create(model=self._model_name,
                                                            messages=msgs, # type: ignore[arg-type]
                                                            
                                                            max_tokens=resolved_max_tokens,
                                                            stop=genconf.stop,
                                                            temperature=genconf.temperature,
                                                            top_p=genconf.top_p,
                                                            # seed=seed,
                                                            **json_kwargs,
                                                
                                                            n=1
                                                            )
        except Exception as e:
            raise RuntimeError(f"Cannot generate. Internal error: {e}")
        
            
        logger.debug(f"OpenAI response: {response}")

        choice = response.choices[0]
        finish = choice.finish_reason
        message = choice.message

        if "tool_choice" in json_kwargs:
            
            # json schema generation via the tools API:
            if message.tool_calls is not None:
                fn = message.tool_calls[0].function
                if fn.name != fn_name:
                    logger.debug(f"OpenAIModel: different returned JSON function name ({fn.name})")

                text = fn.arguments
            else: # use content instead
                text = message.content # type: ignore[assignment]
        
        else:
            # text or simple json format
            text = message.content # type: ignore[assignment]

        out = self._prepare_gen_out(text, finish, genconf)

        return out

        


    


    def token_len(self,
                  thread: Thread,
                  genconf: Optional[GenConf] = None) -> int:
        """Calculate the number of tokens used by a list of messages.
        If a json_schema is provided in genconf, we use its string's token_len as upper bound for the extra prompt tokens.
        
        From https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        
        More info on calculating function_call (and tools?) tokens:

        https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/24
        
        https://gist.github.com/CGamesPlay/dd4f108f27e2eec145eedf5c717318f5

        Args:
            thread: For token length calculation.
            genconf: Model generation configuration. Defaults to None.

        Returns:
            Estimated number of tokens the thread will use.
        """

        # name = self._model_name

        num_tokens = 0
        for index in range(-1, len(thread)): # -1 for system message
            message = thread.msg_as_chatml(index)
            # print(message)
            num_tokens += self._overhead_per_msg
            for key, value in message.items():
                num_tokens += len(self.tokenizer.encode(value))
                # if key == "name":
                #     num_tokens += self._tokens_per_name
        
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        num_tokens += 10 # match API return counts

        # print("text token_len", num_tokens)

        if genconf is not None and genconf.json_schema is not None:
            if isinstance(genconf.json_schema, str):
                js_str = genconf.json_schema
            else:
                js_str = json.dumps(genconf.json_schema)

            tools_num_tokens = self.tokenizer.token_len(js_str)

            # this is an upper bound, as empirically tested with the api.
            tools_num_tokens = int(tools_num_tokens * self.TOOLS_TOKEN_LEN_FACTOR)

            # print("tools token_len", tools_num_tokens)

            num_tokens += tools_num_tokens
        
        return num_tokens



    
    @property
    def desc(self) -> str:
        """Model description."""
        return f"OpenAIModel: {self._model_name}"


    @classmethod
    def provider_version(_) -> str:
        """Provider library version: provider x.y.z
        Ex. openai 1.3.6
        """
        try:        
            import openai
            ver = openai.__version__
        except Exception:
            raise ImportError("Please install openai by running: pip install openai")
            
        return f"openai {ver}"
    
    @classmethod
    def known_models(cls) -> Union[list[str], None]:
        """If the model can only use a fixed set of models, return their names. Otherwise, return None.

        Returns:
            Returns a list of known models or None if it can accept any model.
        """
        return None






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

   





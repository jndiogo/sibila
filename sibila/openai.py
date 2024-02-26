"""OpenAI remote model access.

- OpenAIModel: Use OpenAI API to access their models.
- OpenAITokenizer: Tokenizer for OpenAI models.
"""


from typing import Any, Optional, Union
import sys, json
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



"""
Locate model and token counting data.
More recent models inside each family (GPT-4, GPT-3.5) first
Str values are links to a versioned model.
https://platform.openai.com/docs/models
"""
KNOWN_MODELS: dict = { # name: link | (ctx_len, n_msg_tokens, n_name_tokens)

    # ------------------------------ GPT 4
    "gpt-4-0613":     (8192, 3, 1),
    "gpt-4-32k-0613": (32768, 3, 1),
    "gpt-4-0314":     (8192, 3, 1),
    "gpt-4-32k-0314": (32768, 3, 1),
    
    "gpt-4-1106-preview": (4096, 3, 1), # aka gpt-4-turbo max input: 128000
    
    "gpt-4": "gpt-4-0613",
    "gpt-4-32k": "gpt-4-32k-0613",

    # ------------------------------ GPT 3.5
    "gpt-3.5-turbo-1106": (4096, 3, 1), # max input: 16385
    "gpt-3.5-turbo-0613": (4096, 3, 1),
    "gpt-3.5-turbo-16k-0613": (16385, 3, 1),
    "gpt-3.5-turbo-0301":  (4096, 4, -1),

    "gpt-3.5-turbo": "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k": "gpt-3.5-turbo-16k-0613",

    "gpt-3": "gpt-3.5-turbo-1106",
    "gpt-3.5": "gpt-3.5-turbo-1106",

}

def resolve_model(name: str, 
                  unknown_name_mask: int) -> tuple[str, int, int, int]:
    """Resolve a name string into an existing model from KNOWN_MODELS.

    Args:
        name: The model name to resolve.
        unknown_name_mask: How to deal with unmatched names, a mask of:

            - 2: Raise NameError if exact name not found.
            - 1: Only allow versioned names - raise NameError if generic non-versioned model name used.
            - 0: Accept any name, use first in list if necessary.

    Raises:
        NameError: If not found, according to unknown_name_mask.

    Returns:
        Existing model name from KNOWN_MODELS.
    """
    
    if name not in KNOWN_MODELS:
        
        if unknown_name_mask & 2:
            raise NameError(f"Unknown model '{name}'. Please provide a correct model name (see openai.KNOWN_MODELS) or use a more permissive unknown_name_mask")
        
        # 1) search first entry that starts with name
        nam = name
        found = None        

        while nam:
            for k in KNOWN_MODELS:
                if k.startswith(nam):
                    found = k
                    break
            if found is not None:
                break

            nam = nam[:-1] # remove last letter and try again
            
        if found is None: # 2) pick first entry in list
            found = list(KNOWN_MODELS.keys())[0]

        logger.warn(f"'{name}' not found: assuming '{found}' model")
        name = found
            
    v = KNOWN_MODELS[name]
    if isinstance(v, str):
        if unknown_name_mask & 1:
            raise NameError(f"Unknown model '{name}': would substitute with '{v}'. Please provide a versioned model name (see openai.KNOWN_MODELS) or use a more permissive unknown_name_mask")
            
        logger.info(f"'{name}' is a generic model name that may be updated over time: assuming '{v}' model")
        name = v

    t = KNOWN_MODELS[name]
    return name, t[0], t[1], t[2]







class OpenAIModel(MessagesModel):
    """Access an OpenAI model.

    Supports constrained JSON output, via the OpenAI API tools mechanism.
    Ref: https://platform.openai.com/docs/api-reference/chat/create

    Attributes:
        ctx_len: Maximum context length, shared for input + output.
        desc: Model information.
    """

    def __init__(self,
                 name: str,
                 unknown_name_mask: int = 0,
                 *,
                 
                 # common base model args
                 genconf: Optional[GenConf] = None,
                 schemaconf: Optional[JSchemaConf] = None,
                 tokenizer: Optional[Tokenizer] = None,
                 ctx_len: int = 0,
                
                 # most important OpenAI-specific args
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 
                 # OpenAI-specific args
                 openai_init_kwargs: dict = {},
                 ):
        """
        Args:
            name: Model name.
            unknown_name_mask: How to deal with unmatched names, a mask of:

                - 2: Raise NameError if exact name not found.
                - 1: Only allow versioned names - raise NameError if generic non-versioned model name used.
                - 0: (default) Accept any name, use first in list if necessary.
            
            genconf: Model generation configuration. Defaults to None.
            tokenizer: An external initialized tokenizer to use instead of the created from the GGUF file. Defaults to None.
            ctx_len: Maximum context length to be used (shared for input and output). Defaults to 0 which means model's maximum.
            api_key: OpenAI API key. Defaults to None, which will use env variable OPENAI_API_KEY.
            base_url: Base location for API access. Defaults to None, which will use env variable OPENAI_BASE_URL.
            openai_init_kwargs: Extra args for OpenAI.OpenAI() initialization. Defaults to {}.

        Raises:
            ImportError: If OpenAI API is not installed.
        """


        if not has_openai:
            raise ImportError("Please install openai by running: pip install openai")

        self._model_name, max_ctx_len, self._tokens_per_message, self._tokens_per_name = resolve_model(
            name,
            unknown_name_mask
        )

           
        super().__init__(False,
                         genconf,
                         schemaconf,
                         tokenizer
                         )

        # only check for "json" text presence as json schema is requested with the tools facility.
        self.json_format_instructors["json_schema"] = self.json_format_instructors["json"]
        
        logger.debug(f"Creating OpenAI with base_url={base_url}, openai_init_kwargs={openai_init_kwargs}")

        self._client = openai.OpenAI(api_key=api_key,
                                     base_url=base_url,
                        
                                     **openai_init_kwargs
                                     )

        
        # correct super __init__ values
        if self.tokenizer is None:
            self.tokenizer = OpenAITokenizer(self._model_name)

        if ctx_len == 0:
            self._ctx_len = max_ctx_len
        else:
            self._ctx_len = ctx_len
        

    
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

        if genconf is None:
            genconf = self.genconf

        token_len = self.token_len(thread, genconf)
        if genconf.max_tokens == 0:
            genconf = genconf(max_tokens=self.ctx_len - token_len)
            
        elif token_len + genconf.max_tokens > self.ctx_len:
            # this is not true for all models: 1106 models have 128k max input and 4k max output (in and out ctx are not shared)
            # so we assume the smaller max ctx length for the model
            logger.warn(f"Token length + genconf.max_tokens ({token_len + genconf.max_tokens}) is greater than model's context window length ({self.ctx_len})")

        
        thread = self._prepare_gen_in(thread, genconf)
        
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
        
        msgs = thread.as_chatml()

        # https://platform.openai.com/docs/api-reference/chat/create
        response = self._client.chat.completions.create(model=self._model_name,
                                                        messages=msgs, # type: ignore[arg-type]
                                                        
                                                        max_tokens=genconf.max_tokens,
                                                        stop=genconf.stop,
                                                        temperature=genconf.temperature,
                                                        top_p=genconf.top_p,
                                                        **json_kwargs,
                                            
                                                        n=1
                                                        )

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
            num_tokens += self._tokens_per_message
            for key, value in message.items():
                num_tokens += len(self.tokenizer.encode(value))
                # if key == "name":
                #     num_tokens += self._tokens_per_name
        
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

        if genconf is not None and genconf.json_schema is not None:
            if isinstance(genconf.json_schema, str):
                js_str = genconf.json_schema
            else:
                js_str = json.dumps(genconf.json_schema)
            # this is an upper bound, as empirically tested with the api.
            num_tokens += self.tokenizer.token_len(js_str)                
        
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

   





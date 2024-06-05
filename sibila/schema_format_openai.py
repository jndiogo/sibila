"""OpenAI API model access for providers that allow passing a JSON Schema in the response_format.
Instead of using the tools functionality to receive JSON matching a schema, some providers accept
a direct JSON Schema in the response_format entry.
Where OpenAI is mentioned, its referring to API use, not the company.

- SchemaFormatOpenAIModel: Base class for providers with his functionality.
"""


from typing import Any, Optional, Union
import os, json
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

from .openai import OpenAIModel


try:
    import openai
    has_openai = True
except ImportError:
    has_openai = False
    



class SchemaFormatOpenAIModel(OpenAIModel):
    """Access a model that allows JSON Schema passed in response_format.
    """

    PROVIDER_NAME:str = "to be set by derived class"
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
                
                 # most important OpenAI API specific args
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 token_estimation_factor: Optional[float] = None,
                 
                 # other Open API specific args
                 other_init_kwargs: dict = {},
                 ):
        """Create a remote model.
        Name resolution depends on unknown_name_mask and will keep removing letters 
        from the end of name and searching existing entries in penAIModel.known_models().

        Args:
            name: Model name to resolve into an existing model.
            genconf: Model generation configuration. Defaults to None.
            schemaconf: Default configuration for JSON schema validation, used if generation call doesn't supply one. Defaults to None.
            ctx_len: Maximum context length to be used (shared for input and output). None for model's default.
            max_tokens_limit: Maximum output tokens limit. None for model's default.
            tokenizer: An external initialized tokenizer to use instead of the created from the GGUF file. Defaults to None.
            api_key: API key. Defaults to None, which will use env variable *_API_KEY.
            base_url: Base location for API access. Defaults to None, which will use env variable *_BASE_URL or a default.
            token_estimation_factor: Used when no tokenizer is available. Multiplication factor to estimate token usage: multiplies total text length to obtain token length.
            other_init_kwargs: Extra args for OpenAI.OpenAI() initialization. Defaults to {}.

        Raises:
            ImportError: If OpenAI API is not installed.
            NameError: If model name was not found or there's an API or authentication problem.
        """

        super().__init__(name,
                            
                         # common base model args
                         genconf=genconf,
                         schemaconf=schemaconf,
                         ctx_len=ctx_len,
                         max_tokens_limit=max_tokens_limit,
                         tokenizer=tokenizer,
                            
                         # most important OpenAI API specific args
                         api_key=api_key,
                         base_url=base_url,
                         overhead_per_msg=0, # won't be used
                         token_estimation_factor=token_estimation_factor,
                         create_tokenizer=False,
                            
                         # other OpenAI API specific args
                         other_init_kwargs=other_init_kwargs)





    def _gen_pre(self, 
                 thread: Thread,
                 genconf: Union[GenConf, None]
                 ) -> tuple:

        if genconf is None:
            genconf = self.genconf

        thread = self._prepare_gen_thread(thread, genconf)

        # Known providers do not require "max_tokens" and may error on excess.
        if genconf.max_tokens != 0:
            token_len = self.token_len(thread, genconf)
            resolved_max_tokens = self.resolve_genconf_max_tokens(token_len, genconf)
        else: # for genconf.max_tokens=0, supported providers don't require "max_tokens", so we don't send below
            resolved_max_tokens = 0


        json_kwargs: dict = {}
        if genconf.format == "json":
            
            json_kwargs["response_format"] = {"type": "json_object"}

            if genconf.json_schema is not None:

                if isinstance(genconf.json_schema, str):
                    schema = json.loads(genconf.json_schema)
                else:
                    schema = genconf.json_schema

                json_kwargs["response_format"]["schema"] = schema

        # seed config is disabled, remote models and some hardware accelerated local models don't support it.
        # seed = genconf.seed
        # if seed == -1:
        #    seed = int(time())
        #    logger.debug(f"SchemaFormatOpenAIModel random seed={seed}")

        
        msgs = thread.as_chatml()

        kwargs = {"model": self._model_name,
                  "messages": msgs, # type: ignore[arg-type]
                  "stop": genconf.stop,
                  "temperature": genconf.temperature,
                  "top_p": genconf.top_p,
                  # "seed": seed,
                  "n": 1,
                  **json_kwargs}

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
            
        logger.debug(f"SchemaFormatOpenAIModel response: {response}")

        choice = response.choices[0]
        finish = choice.finish_reason
        # OpenAI-compatible provider endpoints can give non-standard finish_reason values. Map as needed:
        if finish in ["eos", "tool_calls"]: finish = "stop"
        message = choice.message

        text = message.content # type: ignore[assignment]

        return self._prepare_gen_out(text, finish, genconf)



    @classmethod
    def known_models(cls,
                     api_key: Optional[str] = None) -> Union[list[str], None]:
        """List of model names that can be used. Some of the models are not chat models and cannot be used,
        for example embedding models.
        
        Args:
            api_key: If the model provider requires an API key, pass it here or set it in the respective env variable.

        Returns:
            Returns a list of known models or None if unable to fetch it.
        """
        return None









class TogetherModel(SchemaFormatOpenAIModel):
    """Access a together.ai model with the OpenAI API.
    Supports constrained JSON output, via the response_format JSON Schema mechanism.

    Ref:
        https://docs.together.ai/docs/json-mode
        
        https://docs.together.ai/reference/chat-completions
    """

    PROVIDER_NAME:str = "together"
    """Provider prefix that this class handles."""

    DEFAULT_BASE_URL: str = "https://api.together.xyz/v1"
    """Default API access URL"""

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
                
                 # most important OpenAI API specific args
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 token_estimation_factor: Optional[float] = None,
                 
                 # other OpenAI API specific args
                 other_init_kwargs: dict = {},
                 ):
        """Create a together.ai remote model.

        Args:
            name: Model name to resolve into an existing model.
            genconf: Model generation configuration. Defaults to None.
            schemaconf: Default configuration for JSON schema validation, used if generation call doesn't supply one. Defaults to None.
            ctx_len: Maximum context length to be used (shared for input and output). None for model's default.
            max_tokens_limit: Maximum output tokens limit. None for model's default.
            tokenizer: An external initialized tokenizer to use instead of the created from the GGUF file. Defaults to None.
            api_key: API key. Defaults to None, which will use env variable TOGETHER_API_KEY.
            base_url: Base location for API access. Defaults to None, which will use env variable TOGETHER_BASE_URL or a default.
            token_estimation_factor: Used when no tokenizer is available. Multiplication factor to estimate token usage: multiplies total text length to obtain token length.
            other_init_kwargs: Extra args for OpenAI.OpenAI() initialization. Defaults to {}.

        Raises:
            ImportError: If OpenAI API is not installed.
            NameError: If model name was not found or there's an API or authentication problem.
        """

        if api_key is None:
            api_key = os.environ.get("TOGETHER_API_KEY")
        if base_url is None:
            base_url = os.environ.get("TOGETHER_BASE_URL", self.DEFAULT_BASE_URL)

        super().__init__(name,
                         # common base model args
                         genconf=genconf,
                         schemaconf=schemaconf,
                         ctx_len=ctx_len,
                         max_tokens_limit=max_tokens_limit,
                         tokenizer=tokenizer,
                            
                         # most important OpenAI API specific args
                         api_key=api_key,
                         base_url=base_url,
                         token_estimation_factor=token_estimation_factor,
                            
                         # other OpenAI API specific args
                         other_init_kwargs=other_init_kwargs)

        self.maybe_image_input = False # no together.ai models currently support image input - always check model specs











class FireworksModel(SchemaFormatOpenAIModel):
    """Access a Fireworks AI model with the OpenAI API.
    Supports constrained JSON output, via the response_format JSON Schema mechanism.

    Ref:
        https://readme.fireworks.ai/docs/structured-response-formatting

        https://readme.fireworks.ai/reference/createchatcompletion
    """

    PROVIDER_NAME:str = "fireworks"
    """Provider prefix that this class handles."""

    DEFAULT_BASE_URL: str = "https://api.fireworks.ai/inference/v1"
    """Default API access URL"""

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
                
                 # most important OpenAI API specific args
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 token_estimation_factor: Optional[float] = None,
                 
                 # other OpenAI API specific args
                 other_init_kwargs: dict = {},
                 ):
        """Create a Fireworks AI remote model.

        Args:
            name: Model name to resolve into an existing model.
            genconf: Model generation configuration. Defaults to None.
            schemaconf: Default configuration for JSON schema validation, used if generation call doesn't supply one. Defaults to None.
            ctx_len: Maximum context length to be used (shared for input and output). None for model's default.
            max_tokens_limit: Maximum output tokens limit. None for model's default.
            tokenizer: An external initialized tokenizer to use instead of the created from the GGUF file. Defaults to None.
            api_key: API key. Defaults to None, which will use env variable FIREWORKS_API_KEY.
            base_url: Base location for API access. Defaults to None, which will use env variable FIREWORKS_BASE_URL or a default.
            token_estimation_factor: Used when no tokenizer is available. Multiplication factor to estimate token usage: multiplies total text length to obtain token length.
            other_init_kwargs: Extra args for OpenAI.OpenAI() initialization. Defaults to {}.

        Raises:
            ImportError: If OpenAI API is not installed.
            NameError: If model name was not found or there's an API or authentication problem.
        """

        if api_key is None:
            api_key = os.environ.get("FIREWORKS_API_KEY")
        if base_url is None:
            base_url = os.environ.get("FIREWORKS_BASE_URL", self.DEFAULT_BASE_URL)

        super().__init__(name,
                         # common base model args
                         genconf=genconf,
                         schemaconf=schemaconf,
                         ctx_len=ctx_len,
                         max_tokens_limit=max_tokens_limit,
                         tokenizer=tokenizer,
                            
                         # most important OpenAI API specific args
                         api_key=api_key,
                         base_url=base_url,
                         token_estimation_factor=token_estimation_factor,
                            
                         # other OpenAI API specific args
                         other_init_kwargs=other_init_kwargs)

        self.maybe_image_input = False # no Fireworks models currently support image input - always check model specs







class GroqModel(SchemaFormatOpenAIModel):
    """Access a Groq model with the OpenAI API.
    Supports constrained JSON output, via the response_format JSON Schema mechanism.

    Ref:
        https://console.groq.com/docs/tool-use

        https://github.com/groq/groq-api-cookbook/blob/main/parallel-tool-use/parallel-tool-use.ipynb
    """

    PROVIDER_NAME:str = "groq"
    """Provider prefix that this class handles."""

    DEFAULT_BASE_URL: str = "https://api.groq.com/openai/v1"
    """Default API access URL"""

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
                
                 # most important OpenAI API specific args
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 token_estimation_factor: Optional[float] = None,
                 
                 # other OpenAI API specific args
                 other_init_kwargs: dict = {},
                 ):
        """Create a Groq remote model.

        Args:
            name: Model name to resolve into an existing model.
            genconf: Model generation configuration. Defaults to None.
            schemaconf: Default configuration for JSON schema validation, used if generation call doesn't supply one. Defaults to None.
            ctx_len: Maximum context length to be used (shared for input and output). None for model's default.
            max_tokens_limit: Maximum output tokens limit. None for model's default.
            tokenizer: An external initialized tokenizer to use instead of the created from the GGUF file. Defaults to None.
            api_key: API key. Defaults to None, which will use env variable GROQ_API_KEY.
            base_url: Base location for API access. Defaults to None, which will use env variable GROQ_BASE_URL or a default.
            token_estimation_factor: Used when no tokenizer is available. Multiplication factor to estimate token usage: multiplies total text length to obtain token length.
            other_init_kwargs: Extra args for OpenAI.OpenAI() initialization. Defaults to {}.

        Raises:
            ImportError: If OpenAI API is not installed.
            NameError: If model name was not found or there's an API or authentication problem.
        """

        if api_key is None:
            api_key = os.environ.get("GROQ_API_KEY")
        if base_url is None:
            base_url = os.environ.get("GROQ_BASE_URL", self.DEFAULT_BASE_URL)

        super().__init__(name,
                         # common base model args
                         genconf=genconf,
                         schemaconf=schemaconf,
                         ctx_len=ctx_len,
                         max_tokens_limit=max_tokens_limit,
                         tokenizer=tokenizer,
                            
                         # most important OpenAI API specific args
                         api_key=api_key,
                         base_url=base_url,
                         token_estimation_factor=token_estimation_factor,
                            
                         # other OpenAI API specific args
                         other_init_kwargs=other_init_kwargs)

        self.maybe_image_input = False # no Groq models currently support image input - always check model specs



    def _gen_pre(self, 
                 thread: Thread,
                 genconf: Union[GenConf, None]
                 ) -> tuple:

        if genconf is None:
            genconf = self.genconf

        thread = self._prepare_gen_thread(thread, genconf)

        # This provider doesn't require "max_tokens" and will error on excess.
        if genconf.max_tokens != 0:
            token_len = self.token_len(thread, genconf)
            resolved_max_tokens = self.resolve_genconf_max_tokens(token_len, genconf)
        else: # for genconf.max_tokens=0, this provider doesn't require "max_tokens", so we don't send below
            resolved_max_tokens = 0


        json_kwargs: dict = {}
        if genconf.format == "json":
            
            if genconf.json_schema is not None:
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

            else: # free json output
                json_kwargs["response_format"] = {"type": "json_object"}


        # seed config is disabled, remote models and some hardware accelerated local models don't support it.
        # seed = genconf.seed
        # if seed == -1:
        #    seed = int(time())
        #    logger.debug(f"SchemaFormatOpenAIModel random seed={seed}")

        
        msgs = thread.as_chatml()

        kwargs = {"model": self._model_name,
                  "messages": msgs, # type: ignore[arg-type]
                  "stop": genconf.stop,
                  "temperature": genconf.temperature,
                  "top_p": genconf.top_p,
                  # "seed": seed,
                  "n": 1,
                  **json_kwargs}

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
            
        logger.debug(f"SchemaFormatOpenAIModel response: {response}")

        choice = response.choices[0]
        finish = choice.finish_reason
        # OpenAI-compatible provider endpoints can give non-standard finish_reason values. Map as needed:
        if finish in ["eos", "tool_calls"]: finish = "stop"
        message = choice.message

        if "tool_choice" in pre_kwargs:
            
            # json schema generation via the tools API:
            if message.tool_calls is not None:
                if len(message.tool_calls) != 1:
                    logger.warn(f"SchemaFormatOpenAIModel: expecting single message.tool_calls, but received {len(message.tool_calls)} - using first.")

                fn = message.tool_calls[0].function
                if fn.name != self.output_fn_name:
                    logger.warn(f"SchemaFormatOpenAIModel: expecting '{self.output_fn_name}' function name, received ({fn.name})")

                text = fn.arguments

            else: # use content instead
                logger.warn("SchemaFormatOpenAIModel: expecting message.tool_calls, but none received - using text content")
                text = message.content # type: ignore[assignment]
        
        else:
            # text or simple json format
            text = message.content # type: ignore[assignment]

        return self._prepare_gen_out(text, finish, genconf)

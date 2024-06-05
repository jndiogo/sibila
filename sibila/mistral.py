"""Mistral remote model access.

- MistralModel: Access Mistral AI models.
"""


from typing import Any, Optional, Union
import os, sys, json
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
    from mistralai.client import MistralClient
    from mistralai.async_client import MistralAsyncClient    
    has_mistral = True
except ImportError:
    has_mistral = False




class MistralModel(MessagesModel):
    """Access a Mistral AI model.
    Supports constrained JSON output, via the Mistral API function calling mechanism.

    Ref:
        https://docs.mistral.ai/guides/function-calling/
    """

    PROVIDER_NAME:str = "mistral"
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
                
                 # most important Mistral-specific args
                 api_key: Optional[str] = None,
                 token_estimation_factor: Optional[float] = None,
                 
                 # other Mistral-specific args
                 mistral_init_kwargs: dict = {},
                 ):
        """Create a Mistral AI remote model.

        Args:
            name: Model name to resolve into an existing model.
            genconf: Model generation configuration. Defaults to None.
            schemaconf: Default configuration for JSON schema validation, used if generation call doesn't supply one. Defaults to None.
            ctx_len: Maximum context length to be used (shared for input and output). None for model's default.
            max_tokens_limit: Maximum output tokens limit. None for model's default.
            api_key: Mistral API key. Defaults to None, which will use env variable MISTRAL_API_KEY.
            token_estimation_factor: Multiplication factor to estimate token usage: multiplies total text length to obtain token length.
            mistral_init_kwargs: Extra args for mistral.MistralClient() initialization. Defaults to {}.

        Raises:
            ImportError: If Mistral API is not installed.
            NameError: If model name was not found or there's an API or authentication problem.
        """


        if not has_mistral:
            raise ImportError("Please install mistral by running: pip install mistralai")

        self._client = self._client_async = None


        # also accept "provider:name" for ease of use
        provider_name = self.PROVIDER_NAME + ":"
        if name.startswith(provider_name):
            name = name[len(provider_name):]

        super().__init__(False,
                         genconf,
                         schemaconf,
                         None
                         )

        if (ctx_len is not None and
            max_tokens_limit is not None and
            token_estimation_factor is not None): # all elements given: probably created via Models.create()

            self._model_name = name
            default_ctx_len = ctx_len
            default_max_tokens_limit = max_tokens_limit
            default_token_estimation_factor = token_estimation_factor
        
        else: # need to resolve
            settings = self.resolve_settings(self.PROVIDER_NAME,
                                             name,
                                             ["name", 
                                              "ctx_len", 
                                              "max_tokens_limit", 
                                              "token_estimation_factor"])
            self._model_name = settings.get("name") or name
            default_ctx_len = settings.get("ctx_len") # type: ignore[assignment]
            default_max_tokens_limit = settings.get("max_tokens_limit") or default_ctx_len
            default_token_estimation_factor = settings.get("token_estimation_factor") # type: ignore[assignment]

            # all defaults are conservative values
            if default_ctx_len is None:
                default_ctx_len = 32768
                logger.warning(f"Model '{self._model_name}': unknown ctx_len, assuming {default_ctx_len}")
            if default_max_tokens_limit is None:
                default_max_tokens_limit = default_ctx_len
                logger.warning(f"Model '{self._model_name}': unknown max_tokens_limit, assuming {default_max_tokens_limit}")
            if default_token_estimation_factor is None:
                default_token_estimation_factor = self.DEFAULT_TOKEN_ESTIMATION_FACTOR
                logger.warning(f"Model '{self._model_name}': unknown token_estimation_factor, assuming {default_token_estimation_factor}")


        self.ctx_len = ctx_len or default_ctx_len
        
        self.max_tokens_limit = max_tokens_limit or default_max_tokens_limit

        self.max_tokens_limit = min(self.max_tokens_limit, self.ctx_len)

        self._token_estimation_factor = token_estimation_factor or default_token_estimation_factor

        self.maybe_image_input = False # no Mistral models currently support image input - always check model specs

        # only check for "json" text presence as json schema (including field descriptions) is requested with the tools facility.
        self.json_format_instructors["json_schema"] = self.json_format_instructors["json"]

        self._client_init_kwargs = mistral_init_kwargs

        if api_key is not None:
            self._client_init_kwargs["api_key"] = api_key    
        elif "api_key" not in self._client_init_kwargs and "MISTRAL_API_KEY" in os.environ:
            # "MISTRAL_API_KEY" env key is ignored in pytest?
            self._client_init_kwargs["api_key"] = os.environ["MISTRAL_API_KEY"]



    def close(self):
        """Close model, release resources like memory or net connections."""
        self._client = self._client_async = None



    def _ensure_client(self,
                       is_async: bool):
        if is_async and self._client_async is not None:
            return
        elif not is_async and self._client is not None:
            return
            
        try:
            logger.debug(f"Creating inner MistralClient with ctx_len={self.ctx_len}, max_tokens_limit={self.max_tokens_limit}, "
                         f"_token_estimation_factor={self._token_estimation_factor}, init_kwargs={self._client_init_kwargs}")

            if is_async:
                self._client_async = MistralAsyncClient(**self._client_init_kwargs) # type: ignore[assignment]
            else:
                self._client = MistralClient(**self._client_init_kwargs) # type: ignore[assignment]

        except Exception as e:
            raise NameError(f"Could not create {'async' if is_async else ''} model '{self._model_name}' with error: {e}")
                








    def _gen_pre(self, 
                 thread: Thread,
                 genconf: Union[GenConf, None]
                 ) -> tuple:

        if genconf is None:
            genconf = self.genconf

        thread = self._prepare_gen_thread(thread, genconf)

        if genconf.max_tokens != 0:
            """
            This provider doesn't require "max_tokens" and won't error on excess.

            Since we can only have an estimate of token length, we don't use it when generating:
            token_len = self.token_len(thread, genconf)
            resolved_max_tokens = self.resolve_genconf_max_tokens(token_len, genconf)

            Instead we allow all available output length. This is only possible because endpoint
            doesn't error when max_tokens is larger than possible:"""
            resolved_max_tokens = self.resolve_genconf_max_tokens(0, genconf)
        else:
            resolved_max_tokens = 0


        # https://docs.mistral.ai/api/#operation/createChatCompletion

        json_kwargs: dict = {}
        format = genconf.format
        if format == "json":
            
            if genconf.json_schema is None:
                json_kwargs["response_format"] = {"type": "json_object"}

            else:
                # use json_schema in Mistral's functions API
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

                json_kwargs["tool_choice"] = "any"

            logger.debug(f"Mistral json args: {json_kwargs}")

        # seed config is disabled, remote models and some hardware accelerated local models don't support it.
        # seed = genconf.seed
        # if seed == -1:
        #    seed = int(time())
        #    logger.debug(f"Mistral random seed={seed}")

        
        msgs = thread.as_chatml()

        kwargs = {"model": self._model_name,
                  "messages": msgs, # type: ignore[arg-type]
                  "temperature": genconf.temperature,
                  "top_p": 1. if genconf.temperature == 0 else genconf.top_p,
                  # mistral API has no "stop" arg
                  # "random_seed": seed,
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
            
        logger.debug(f"Mistral response: {response}")

        choice = response.choices[0]
        
        finish = choice.finish_reason
        if finish == "tool_calls": finish = "stop"
        elif finish == "model_length": finish = "length"

        message = choice.message

        if "tool_choice" in pre_kwargs:
            
            # json schema generation via the tools API:
            if message.tool_calls is not None:
                if len(message.tool_calls) != 1:
                    logger.warn(f"Mistral: expecting single message.tool_calls, but received {len(message.tool_calls)} - using first.")

                fn = message.tool_calls[0].function
                if fn.name != self.output_fn_name:
                    logger.warn(f"Mistral: expecting '{self.output_fn_name}' function name, received ({fn.name})")

                text = fn.arguments

            else: # use content instead
                logger.warn("Mistral: expecting message.tool_calls, but none received - using text content")
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
            response = self._client.chat(**kwargs) # type: ignore[attr-defined]

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
            response = await self._client_async.chat(**kwargs) # type: ignore[attr-defined]

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

        Args:
            thread_or_text: For token length calculation.
            genconf: Model generation configuration. Defaults to None.

        Returns:
            Estimated number of tokens occupied.
        """

        if isinstance(thread_or_text, Thread):
            thread = thread_or_text            
        else:
            thread = Thread.make_IN(thread_or_text)

        OVERHEAD_PER_MSG = 3
        num_tokens = 0
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
        
        # print(num_tokens)
        return num_tokens

    
    @classmethod
    def known_models(cls,
                     api_key: Optional[str] = None) -> Union[list[str], None]:
        """If the model can only use a fixed set of models, return their names. Otherwise, return None.

        Args:
            api_key: If the model provider requires an API key, pass it here or set it in the respective env variable.

        Returns:
            Returns a list of known models or None if unable to fetch it.
        """

        args = {}
        if api_key is not None:
            args["api_key"] = api_key
        model = MistralClient(**args) # type: ignore[arg-type]

        model_list = model.list_models()
        del model

        out = []
        for mod in model_list.data:
            out.append(mod.id)

        return sorted(out)





    def name(self) -> str:
        """Model (short) name."""
        return self._model_name
        
    def desc(self) -> str:
        """Model description."""
        return f"MistralModel: {self._model_name}"


    @classmethod
    def provider_version(_) -> str:
        """Provider library version: provider x.y.z
        Ex. mistralai-0.1.8
        """
        try:
            ver = MistralClient()._version
        except Exception:
            raise ImportError("Please install mistralai by running: pip install mistralai")
            
        return f"mistralai-{ver}"
    

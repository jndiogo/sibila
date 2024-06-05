"""Anthropic remote model access.

- AnthropicModel: Access Anthropic models.
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
    from anthropic import Anthropic, AsyncAnthropic
    has_anthropic = True
except ImportError:
    has_anthropic = False




class AnthropicModel(MessagesModel):
    """Access an Anthropic model.
    Supports constrained JSON output, via the Anthropic API function calling mechanism.

    Ref:
        https://docs.anthropic.com/claude/docs/intro-to-claude
    """

    PROVIDER_NAME:str = "anthropic"
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
                
                 # most important Anthropic-specific args
                 api_key: Optional[str] = None,
                 token_estimation_factor: Optional[float] = None,
                 
                 # other Anthropic-specific args
                 anthropic_init_kwargs: dict = {},
                 ):
        """Create an Anthropic remote model.

        Args:
            name: Model name to resolve into an existing model.
            genconf: Model generation configuration. Defaults to None.
            schemaconf: Default configuration for JSON schema validation, used if generation call doesn't supply one. Defaults to None.
            ctx_len: Maximum context length to be used (shared for input and output). None for model's default.
            max_tokens_limit: Maximum output tokens limit. None for model's default.
            api_key: Anthropic API key. Defaults to None, which will use env variable ANTHROPIC_API_KEY.
            token_estimation_factor: Multiplication factor to estimate token usage: multiplies total text length to obtain token length.
            anthropic_init_kwargs: Extra args for Anthropic() initialization. Defaults to {}.

        Raises:
            ImportError: If Anthropic API is not installed.
            NameError: If model name was not found or there's an API or authentication problem.
        """


        if not has_anthropic:
            raise ImportError("Please install anthropic API by running: pip install anthropic")

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
                default_ctx_len = 200000
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

        self.maybe_image_input = True # currently all Anthropic models support image input - always check model specs

        # only check for "json" text presence as json schema (including field descriptions) is requested with the tools facility.
        self.json_format_instructors["json_schema"] = self.json_format_instructors["json"]

        self._client_init_kwargs = anthropic_init_kwargs

        if api_key is not None:
            self._client_init_kwargs["api_key"] = api_key    



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
            logger.debug(f"Creating inner Anthropic client with ctx_len={self.ctx_len}, max_tokens_limit={self.max_tokens_limit}, "
                         f"_token_estimation_factor={self._token_estimation_factor}, init_kwargs={self._client_init_kwargs}")

            if is_async:
                self._client_async = AsyncAnthropic(**self._client_init_kwargs) # type: ignore[assignment]
            else:
                self._client = Anthropic(**self._client_init_kwargs) # type: ignore[assignment]

        except Exception as e:
            raise NameError(f"Could not create {'async' if is_async else ''} model '{self._model_name}' with error: {e}")
                








    def _gen_pre(self, 
                 thread: Thread,
                 genconf: Union[GenConf, None]
                 ) -> tuple:

        if genconf is None:
            genconf = self.genconf

        thread = self._prepare_gen_thread(thread, genconf)

        """
        This provider requires "max_tokens" and doesn't error on excess.

        Since we can only have an estimate of token length, we don't use it when generating:
            token_len = self.token_len(thread, genconf)
            resolved_max_tokens = self.resolve_genconf_max_tokens(token_len, genconf)

        Instead we allow all available output length. This is only possible because endpoint
        doesn't error when max_tokens is larger than possible:"""
        resolved_max_tokens = self.resolve_genconf_max_tokens(0, genconf)


        # https://docs.anthropic.com/claude/docs/tool-use

        json_kwargs: dict = {}
        format = genconf.format
        if format == "json":
            
            if genconf.json_schema is not None:
                # use json_schema in Anthropic's functions API
                if isinstance(genconf.json_schema, str):
                    params = json.loads(genconf.json_schema)
                else:
                    params = genconf.json_schema
                
                json_kwargs["tools"] = [ # description is optional
                    {
                        "name": self.output_fn_name,
                        "input_schema": params
                    }
                ]

            logger.debug(f"Anthropic json args: {json_kwargs}")
            
        has_images = thread.has_images
        if has_images: # download any remote url images to data: urls
            thread = thread.clone()
            thread.download_images_as_data()

        msgs = thread.as_chatml(include_INST=False)

        if has_images: # massage images from the ChatML format into Anthropic's format
            for msg in msgs:
                content = msg["content"]
                if isinstance(content, list):
                    for cont in content:
                        if cont["type"] == "image_url":
                            image_url = cont["image_url"]["url"]
                            
                            image_url = image_url[5:]
                            mime = image_url.split(";")
                            if len(mime) != 2:
                                raise ValueError(f"Error decoding image data: '{image_url[:16]}'")
                            base64 = mime[1].split(",")
                            if len(base64) != 2 or base64[0].lower() != "base64":
                                raise ValueError(f"Error decoding image data base64: '{image_url[:32]}'")
                            mime_str = mime[0]
                            base64_str = base64[1]

                            # rewrite keys
                            cont["type"] = "image"
                            del cont["image_url"]
                            cont["source"] = {
                                "type": "base64",
                                "media_type": mime_str,
                                "data": base64_str
                            }


        if format == "json" and "tools" not in json_kwargs: 
            # json non-schema request: prefill format as an assistant message
            msgs.append({"role": "assistant", "content": "{"})

        # Anthropic API has no support for seed parameter
        
        kwargs = {"model": self._model_name,
                  "messages": msgs, # type: ignore[arg-type]
                  "stop_sequences": genconf.stop,
                  "max_tokens": resolved_max_tokens,
                  "temperature": genconf.temperature,
                  "top_p": genconf.top_p,
                  **json_kwargs}

        if thread.inst:
            kwargs["system"] = thread.inst.text

        # inject model-specific args, if any
        kwargs.update(genconf.resolve_special(self.PROVIDER_NAME))

        logger.debug(f"{type(self).__name__} gen args: {kwargs}")

        return (kwargs, genconf)
        

    def _gen_post(self, 
                  response: Any,
                  pre_kwargs: dict,
                  genconf: GenConf
                  ) -> GenOut:
            
        logger.debug(f"Anthropic response: {response}")

        finish = "length" if response.stop_reason == "max_tokens" else "stop"

        for message in response.content:

            if "tools" in pre_kwargs:
                if message.type == "tool_use":                
                    if message.name != self.output_fn_name:
                        logger.warn(f"Anthropic: expecting '{self.output_fn_name}' function name, received ({message.name})")
                    response = message.input # a dict
                    break

            elif message.type == "text":
                response = message.text
                if genconf.format == "json": # prepend previous prefill
                    response = "{" + response
                break

        return self._prepare_gen_out(response, finish, genconf)







    
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
            if "tools" in kwargs:
                response = self._client.beta.tools.messages.create(**kwargs) # type: ignore[attr-defined]
            else:
                response = self._client.messages.create(**kwargs) # type: ignore[attr-defined]

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
            if "tools" in kwargs:
                response = await self._client_async.beta.tools.messages.create(**kwargs) # type: ignore[attr-defined]
            else:
                response = await self._client_async.messages.create(**kwargs) # type: ignore[attr-defined]

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


 

    def name(self) -> str:
        """Model (short) name."""
        return self._model_name
        
    def desc(self) -> str:
        """Model description."""
        return f"AnthropicModel: {self._model_name}"


    @classmethod
    def provider_version(_) -> str:
        """Provider library version: provider x.y.z
        Ex. anthropic-0.25.6
        """
        try:
            import anthropic
            ver = anthropic.__version__
        except Exception:
            raise ImportError("Please install anthropic API by running: pip install anthropic")
            
        return f"anthropic-{ver}"
    
    

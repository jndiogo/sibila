"""Local model access via the llama.cpp engine.

- LlamaCppModel: Use local GGUF format models via llama.cpp engine.
- LlamaCppTokenizer: Tokenizer for llama.cpp loaded GGUF models.
"""


from typing import Optional, Union, BinaryIO, Literal

import sys, os, ctypes, struct, re

import logging
logger = logging.getLogger(__name__)


from .gen import (
    GenConf
)

from .thread import (
    Thread
)

from .model import (
    FormattedTextModel,
    Tokenizer
)

from .models import Models

from .json_schema import JSchemaConf

from .json_grammar import (
    gbnf_from_json_schema,
    JSON_GBNF
)

from .utils import get_bytes_from_url


try:
    from llama_cpp import (
        Llama,
        llama_cpp,
        llama_token_get_text,
        llama_grammar,
    )
    has_llama_cpp = True
except ImportError:
    has_llama_cpp = False

try:
    from llama_cpp import (
        llava_cpp,
    )
    has_llava_cpp = True
except ImportError:
    has_llava_cpp = False





class LlamaCppModel(FormattedTextModel):
    """Use local GGUF format models via llama.cpp engine.
    
    Supports grammar-constrained JSON output following a JSON schema.
    """

    PROVIDER_NAME:str = "llamacpp"
    """Provider prefix that this class handles."""

    _llama: Llama
    """LlamaCpp instance"""

    def __init__(self,
                 path: str,

                 format: Optional[str] = None,                 
                 format_search_order: list[str] = ["name", "meta_template", "folder_json"],

                 *,

                 # common base model args
                 genconf: Optional[GenConf] = None,
                 schemaconf: Optional[JSchemaConf] = None,
                 ctx_len: Optional[int] = None,
                 max_tokens_limit: Optional[int] = None,
                 tokenizer: Optional[Tokenizer] = None,

                 # important LlamaCpp-specific args
                 n_gpu_layers: int = -1,
                 main_gpu: int = 0,
                 n_batch: int = 512,
                 seed: int = 4294967295,
                 verbose: bool = False,

                 # other LlamaCpp-specific args
                 **llamacpp_kwargs
                 ):
        """
        Args:
            path: File path to the GGUF file.
            format: Chat template format to use with model. Leave as None for auto-detection.
            format_search_order: Search order for auto-detecting format, "name" searches in the filename, "meta_template" looks in the model's metadata, "folder_json" looks for configs in file's folder. Defaults to ["name","meta_template", "folder_json"].
            genconf: Default generation configuration, which can be used in gen() and related. Defaults to None.
            schemaconf: Default configuration for JSON schema validation, used if generation call doesn't supply one. Defaults to None.
            ctx_len: Maximum context length to be used. Use 0 for maximum possible size, which may raise an out of memory error. None will use a default from the 'llamacpp' provider's '_default' entry at 'res/base_models.json'.
            max_tokens_limit: Maximum output tokens limit. None for no limit.
            tokenizer: An external initialized tokenizer to use instead of the created from the GGUF file. Defaults to None.
            n_gpu_layers: Number of model layers to run in a GPU. Defaults to -1 for all.
            main_gpu: Index of the GPU to use. Defaults to 0.
            n_batch: Prompt processing batch size. Defaults to 512.
            seed: Random number generation seed, for non zero temperature inference. Defaults to 4294967295.
            verbose: Emit (very) verbose llama.cpp output. Defaults to False.

        Raises:
            ImportError: If llama-cpp-python is not installed.
            ValueError: For arguments or settings problems.
            NameError: If the model was not found or the file is corrupt.
            AttributeError: If a suitable template format was not found.
            MemoryError: If an out of memory situation arises.
        """

        self._llama = None # type: ignore[assignment]
        self._llava_ctx = None

        self.tokenizer = None # type: ignore[assignment]
        self._own_tokenizer = False

        if not has_llama_cpp:
            raise ImportError("Please install llama-cpp-python by running: pip install llama-cpp-python")

        # also accept "provider:path" for ease of use
        provider_name = self.PROVIDER_NAME + ":"
        if path.startswith(provider_name):
            path = path[len(provider_name):]

        sub_paths = extract_sub_paths(path)
        for sp in sub_paths:
            if not os.path.isfile(sp):
                raise NameError(f"Model file not found at '{sp}'")

        llama_path = sub_paths[0]
        llava_proj_path = sub_paths[1] if len(sub_paths) > 1 else None

        # find ctx_len from metadata --and-- check file format
        max_ctx_len = 0
        try:
            md = load_gguf_metadata(llama_path)
            if md is not None:
                for key in md:
                    if key.endswith('.context_length'):
                        max_ctx_len = int(md[key])
                        break
        except Exception as e:
            raise NameError(f"Error loading file '{llama_path}': {e}")


        if ctx_len is None: # find a default in Models _default dict
            defaults = Models.resolve_provider_defaults("llamacpp", ["ctx_len"], 2)
            if defaults["ctx_len"] is not None:
                ctx_len = defaults["ctx_len"]
                logger.debug(f"Defaulting ctx_len={ctx_len} from Models '_default' entry")

        if ctx_len == 0: # default to maximum ctx_len - this can be dangerous, as big ctx_len will probably out of memory
            if max_ctx_len != 0:
                ctx_len = max_ctx_len
            else:
                raise ValueError("Cannot find model's maximum ctx_len information. Please provide a non-zero ctx_len arg")

        if max_ctx_len != 0:
            if ctx_len > max_ctx_len: # type: ignore[operator]
                raise ValueError(f"Arg ctx_len ({ctx_len}) is greater than model's maximum ({max_ctx_len})")


        super().__init__(True,
                         genconf,
                         schemaconf,
                         tokenizer
                         )

        # update kwargs from important args
        llamacpp_kwargs.update(n_ctx=ctx_len,
                               n_batch=n_batch,
                               n_gpu_layers=n_gpu_layers,
                               main_gpu=main_gpu,
                               seed=seed,
                               verbose=verbose
                               )
        
        logger.debug(f"Creating inner Llama model with path='{llama_path}', llamacpp_kwargs={llamacpp_kwargs}")


        try:
            with llamacpp_verbosity_manager(verbose):
                self._llama = Llama(model_path=llama_path, **llamacpp_kwargs)

        except Exception as e:
            raise MemoryError(f"Could not load model file '{llama_path}'. "
                              "This is usually an out of memory situation but could also be due to a corrupt file. "
                              f"Internal error: {e}.")


        self._model_path = llama_path
       

        # correct super __init__ values
        self.ctx_len = self._llama.n_ctx()
        
        if max_tokens_limit is not None:
            self.max_tokens_limit = max_tokens_limit

        self.max_tokens_limit = min(self.max_tokens_limit, self.ctx_len)



        try:
            self.init_format(format,
                             format_search_order,
                             {"name": os.path.basename(path), # note: the multiple filename with '*'
                              "path": path, # note: full path of the multiple filename with '*'
                              "meta_template_name": "tokenizer.chat_template"}
                             )
        except Exception as e:
            del self.tokenizer
            del self._llama
            raise AttributeError(str(e))
            

        # llava projector setup
        if llava_proj_path is not None:

            if not has_llava_cpp:
                raise ImportError("Llava is not available in this installation of llama-cpp-python")
    
            logger.debug(f"Creating inner Llava projector with path='{llava_proj_path}'")

            with llamacpp_verbosity_manager(verbose):
                self._llava_ctx = llava_cpp.clip_model_load(llava_proj_path.encode(encoding='utf-8'), 
                                                            0) # verbosity

            if self._llava_ctx is None:
                raise ValueError(f"Failed to load llava projector: {llava_proj_path}")

            self._model_path += "*" + llava_proj_path

            """
            self._llava_exit_stack = ExitStack()
            def llava_free():
                with llamacpp_verbosity_manager(verbose):
                    llava_cpp.clip_free(self._llava_ctx)
                    
            self._llava_exit_stack.callback(llava_free)
            """

        self.maybe_image_input = self._llava_ctx is not None



        if self.tokenizer is None:
            self.tokenizer = LlamaCppTokenizer(self._llama)
            self._own_tokenizer = True
        else:
            self._own_tokenizer = False



    def close(self):
        """Close model, release resources like memory or net connections."""
        
        if hasattr(self, "tokenizer") and self.tokenizer:
            if hasattr(self, "_own_tokenizer") and self._own_tokenizer:
                del self.tokenizer
            self.tokenizer = None

        if hasattr(self, "_llava_ctx") and self._llava_ctx: # only happens if llama_cpp was loaded
            llava_cpp.clip_free(self._llava_ctx)
            del self._llava_ctx
            self._llava_ctx = None

        if hasattr(self, "_llama") and self._llama:
            del self._llama
            self._llama = None


    def __del__(self):
        """__del__ is important to release llama.cpp memory"""
        # print("LlamaCppModel.__del__")
        self.close()
    
    
    def _gen_thread(self,
                    thread: Thread,
                    genconf: GenConf) -> tuple[str,str]:
        """Generate from formatted text.

        Args:
            thread: Input Thread object.
            genconf: Model generation configuration.

        Raises:
            ValueError: If trying to generate from an empty prompt.
            RuntimeError: If unable to generate.
            
        Returns:
            Tuple of strings: generated_text, finish_reason.
        """

        thread = self._prepare_gen_thread(thread, genconf)

        token_ids = self.tokens_llava_embeddings_from_thread(thread,
                                                             create_embeddings=True)

        # This provider doesn't require "max_tokens" (but defaults to a low value) and will not error on excess.
        if genconf.max_tokens != 0:
            token_len = len(token_ids)
            resolved_max_tokens = self.resolve_genconf_max_tokens(token_len, genconf)
        else: # for genconf.max_tokens=0, this provider doesn't require "max_tokens", so we don't send below
            resolved_max_tokens = 0

        # prepare llamaCpp args:
        genconf_kwargs = genconf.as_dict()
        genconf_kwargs["max_tokens"] = resolved_max_tokens

        format = genconf_kwargs.pop("format")
        if format == "json":
            if genconf_kwargs["json_schema"] is None:
                grammar = llama_grammar.LlamaGrammar.from_string(JSON_GBNF, 
                                                                 logger.getEffectiveLevel() == logging.DEBUG)
                    
            else: # translate json_schema to a llama grammar
                jsg = gbnf_from_json_schema(genconf_kwargs["json_schema"])
                logger.debug(f"JSON schema GBNF grammar:\n█{jsg}█")
                
                grammar = llama_grammar.LlamaGrammar.from_string(jsg, 
                                                                 logger.getEffectiveLevel() == logging.DEBUG)

            genconf_kwargs["grammar"] = grammar
            
        # clean keys unknown to llama.cpp
        genconf_kwargs.pop("json_schema")

        # inject model-specific args, if any
        genconf_kwargs.update(genconf.resolve_special(self.PROVIDER_NAME))
        genconf_kwargs.pop("special")


        # seed config is disabled, has remote models and some hardware accelerated local models don't support it.
        # if "seed" in genconf_kwargs:
        #    if genconf_kwargs["seed"] == -1:
        #        genconf_kwargs["seed"] = int(time())


        logger.debug(f"LlamaCpp args: {genconf_kwargs}")


        try:
            # Llamacpp.create_completion() never returns special tokens because it uses its own detokenize()
            response = self._llama.create_completion(token_ids,
                                                     echo=False,
                                                     stream=False,
                                                     **genconf_kwargs)
        except Exception as e:
            raise RuntimeError(f"Cannot generate. Internal error: {e}")
        
        logger.debug(f"LlamaCpp response: {response}")

        choice = response["choices"][0] # type: ignore[index]

        text = choice["text"]
        finish_reason = choice["finish_reason"]

        return text, finish_reason # type: ignore[return-value]




    async def _gen_thread_async(self,
                                thread: Thread,
                                genconf: GenConf) -> tuple[str,str]:
        """Generate from formatted text. Please note that the llama.cpp engine 
        cannot currently benefit from async: calls will be generated sequentially.

        Args:
            thread: Input Thread object.
            genconf: Model generation configuration.

        Raises:
            ValueError: If trying to generate from an empty prompt.
            RuntimeError: If unable to generate.

        Returns:
            Tuple of strings: generated_text, finish_reason.
        """

        return self._gen_thread(thread, genconf)





    def tokens_from_thread(self,
                           thread: Thread) -> list[int]:
        """Encode a thread to tokens. 
        Thread must be the final thread which will passed to model.
            
        Args:
            thread: Final thread to be passed to model.

        Returns:
            Tokens list.
        """

        return self.tokens_llava_embeddings_from_thread(thread,
                                                        create_embeddings=False)
    



    # based on llamacpp-python's llama_chat_format.py Llava15ChatHandler
    def tokens_llava_embeddings_from_thread(self,
                                            thread: Thread,
                                            create_embeddings: bool) -> list[int]:
        """Encode a thread to tokens and setup llava embeddings. 
        Thread must be the final thread which will passed to model.
        Tokens resulting from image embedding are all set to -1, 
        the embeddings are cached inside llava/llama, so generation should follow.
            
        Args:
            thread: Final thread to be passed to model.
            create_embeddings: True will create image embeddings (if any), besides tokens. False only extracts tokens.

        Returns:
            Tokens list.
        """

        if self.tokenizer is None:
            raise ValueError("A LlamaCppModel object requires a tokenizer")

        text = self.text_from_thread(thread)
        if not text:
            raise ValueError("Cannot generate from an empty prompt")

        if logger.isEnabledFor(logging.DEBUG):
            filtered_text = re.sub(r'(data:[^,]+)\S+', r'\1(...)', text)
            logger.debug(f"Prompt: █{filtered_text}█")


        if not thread.has_images:
            token_ids = self.tokenizer.encode(text)
            logger.debug(f"Prompt len(token_ids): {len(token_ids)}")
            return token_ids


        # there are images in the thread: image embeddings must be taken
        
        if self._llava_ctx is None:
            raise ValueError("Can't embed images: llava_cpp is not enabled for model")

        # extract all image urls:
        image_urls = []
        for msg in thread:
            if msg.has_images:
                for image in msg.images:
                    image_urls.append(image["url"])

        # split prompt into text and image urls
        split_text = split_text_on_image_urls(text, image_urls)
        # print(split_text)


        if create_embeddings:
            # eval must be done with _llama, image embeddings will be cached and used for generation
            self._llama.reset()
            self._llama._ctx.kv_cache_clear()

            for type_, value in split_text:
                if type_ == "text":
                    tokens = self.tokenizer.encode(value)

                    total = self._llama.n_tokens + len(tokens)
                    if total > self._llama.n_ctx():
                        raise ValueError(f"Token length ({total}) exceeds n_ctx ({self._llama.n_ctx()})")

                    self._llama.eval(tokens)

                else:
                    image_bytes = get_bytes_from_url(value)

                    try:
                        embed = None
                        with llamacpp_verbosity_manager(self._llama.verbose):
                            # create embeddings
                            embed = llava_cpp.llava_image_embed_make_with_bytes(
                                self._llava_ctx,
                                self._llama.context_params.n_threads_batch,
                                (ctypes.c_uint8 * len(image_bytes)).from_buffer(bytearray(image_bytes)),
                                len(image_bytes))

                            total = self._llama.n_tokens + embed.contents.n_image_pos
                            if total > self._llama.n_ctx():
                                raise ValueError(f"Token length ({total}) exceeds n_ctx ({self._llama.n_ctx()})")
                                            
                            # eval embeddings
                            n_past = ctypes.c_int(self._llama.n_tokens)
                            n_past_p = ctypes.pointer(n_past)
                            llava_cpp.llava_eval_image_embed(
                                self._llama.ctx,
                                embed,
                                self._llama.n_batch,
                                n_past_p)

                    finally:
                        if embed is not None:
                            llava_cpp.llava_image_embed_free(embed)

                    # avoid issues with hf tokenizer: sets image tokens to -1, but embeddings were stored
                    self._llama.input_ids[self._llama.n_tokens:n_past.value] = -1
                    self._llama.n_tokens = n_past.value

            token_ids = self._llama.input_ids[:self._llama.n_tokens].tolist()

        else: # image embeddings need not be evaluated: emit text tokens and fake -1 tokens for embeddings

            token_ids = []

            for type_, value in split_text:
                if type_ == "text":
                    tokens = self.tokenizer.encode(value)

                    total = len(token_ids) + len(tokens)
                    if total > self._llama.n_ctx():
                        raise ValueError(f"Token length ({total}) exceeds n_ctx ({self._llama.n_ctx()})")

                    token_ids += tokens

                else:
                    image_bytes = get_bytes_from_url(value)

                    try:
                        embed = None
                        with llamacpp_verbosity_manager(self._llama.verbose):
                            # create embeddings
                            embed = llava_cpp.llava_image_embed_make_with_bytes(
                                self._llava_ctx,
                                self._llama.context_params.n_threads_batch,
                                (ctypes.c_uint8 * len(image_bytes)).from_buffer(bytearray(image_bytes)),
                                len(image_bytes))
                            
                            embed_token_count = embed.contents.n_image_pos

                            total = len(token_ids) + embed_token_count
                            if total > self._llama.n_ctx():
                                raise ValueError(f"Token length ({total}) exceeds n_ctx ({self._llama.n_ctx()})")

                    finally:
                        if embed is not None:
                            llava_cpp.llava_image_embed_free(embed)

                    # set image tokens to -1
                    token_ids += [-1] * embed_token_count

        logger.debug(f"Prompt len(token_ids): {len(token_ids)}")

        return token_ids





    def text_from_thread(self,
                         thread: Thread) -> str:
        """Apply template format to transform a Thread into raw text.

        Args:
            thread: Input Thread object.

        Returns:
            Text resulting from chat template expansion.
        """

        if self.tokenizer is None:
            raise ValueError("A FormattedTextModel object requires a tokenizer")

        messages = thread.as_chatml()
        text = self._jinja_compiled_template.render(messages=messages, # type: ignore[union-attr]
                                                    add_generation_prompt=True,
                                                    **self.tokenizer.special_tokens_map())
        return text
        


    def token_len(self,
                  thread_or_text: Union[Thread,str],
                  _: Optional[GenConf] = None) -> int:
        """Calculate or estimate the token length for a Thread or a plain text string.
        In some cases where it's not possible to calculate the exact token count, 
        this function should give a conservative (upper bound) estimate.
        It's up to the implementation whether to account for side information like JSON Schema,
        but it must reflect the model's context token accounting.
        Thread or text must be the final text which will passed to model.
         
        Args:
            thread_or_text: Final thread or text to be passed to model.

        Returns:
            Number of tokens used.
        """

        if isinstance(thread_or_text, Thread):
            thread = thread_or_text            
        else:
            thread = Thread.make_IN(thread_or_text)

        token_ids = self.tokens_from_thread(thread)
        return len(token_ids)










   
    def name(self) -> str:
        """Model (short) name."""
        return os.path.basename(self._model_path)

    def desc(self) -> str:
        """Model description."""
        return f"{type(self).__name__}: '{self._model_path}' - '{self._llama._model.desc()}'"

        
    @classmethod
    def provider_version(_) -> str:
        """Provider library version: provider x.y.z
        Ex. llama-cpp-python-0.2.44
        """
        try:        
            import llama_cpp
            ver = llama_cpp.__version__
        except Exception:
            raise ImportError("Please install llama-cpp-python by running: pip install llama-cpp-python")
            
        return f"llama-cpp-python-{ver}"





    @property
    def n_embd(self) -> int:
        """Embedding size of model."""
        return self._llama.n_embd()

    @property
    def n_params(self) -> int:
        """Total number of model parameters."""
        return self._llama._model.n_params()



    def get_metadata(self):
        """Returns model metadata."""
        out = {}
        buf = bytes(16 * 1024)
        lmodel = self._llama.model
        count = llama_cpp.llama_model_meta_count(lmodel)
        for i in range(count):
            res = llama_cpp.llama_model_meta_key_by_index(lmodel, i, buf,len(buf))
            if res >= 0:
                key = buf[:res].decode('utf-8')
                res = llama_cpp.llama_model_meta_val_str_by_index(lmodel, i, buf,len(buf))
                if res >= 0:
                    value = buf[:res].decode('utf-8')
                    out[key] = value
        return out


   
    
    


class LlamaCppTokenizer(Tokenizer):
    """Tokenizer for llama.cpp loaded GGUF models."""

    def __init__(self, 
                 llama: Llama):
        self._llama = llama

        self.vocab_size = self._llama.n_vocab()

        self.bos_token_id = self._llama.token_bos()
        self.bos_token = llama_token_get_text(self._llama.model, self.bos_token_id).decode("utf-8")

        self.eos_token_id = self._llama.token_eos()
        self.eos_token = llama_token_get_text(self._llama.model, self.eos_token_id).decode("utf-8")

        self.pad_token_id = None
        self.pad_token = None

        self.unk_token_id = None # ? fill by taking a look at id 0?
        self.unk_token = None

    
    def encode(self, 
               text: str) -> list[int]:
        """Encode text into model tokens. Inverse of Decode().

        Args:
            text: Text to be encoded.

        Returns:
            A list of ints with the encoded tokens.
        """

        # str -> bytes
        btext = text.encode("utf-8", errors="ignore")

        return self._llama.tokenize(btext, add_bos=False, special=True)



    def decode(self,
               token_ids: list[int],
               skip_special: bool = True) -> str:
        """Decode model tokens to text. Inverse of Encode().

        Using instead of llama-cpp-python's to fix error: remove first character after a bos only if it's a space.

        Args:
            token_ids: List of model tokens.
            skip_special: Don't decode special tokens like bos and eos. Defaults to True.

        Returns:
            Decoded text.
        """

        if not len(token_ids):
            return ""
        
        output = b""
        size = 32
        buffer = (ctypes.c_char * size)()

        if not skip_special:
            special_toks = {self.bos_token_id: self.bos_token.encode("utf-8"), # type: ignore[union-attr]
                            self.eos_token_id: self.eos_token.encode("utf-8")} # type: ignore[union-attr]

            for token in token_ids:
                if token == self.bos_token_id:
                    output += special_toks[token]
                elif token == self.eos_token_id:
                    output += special_toks[token]
                else:
                    n = llama_cpp.llama_token_to_piece(
                        self._llama.model, llama_cpp.llama_token(token), buffer, size
                    )
                    output += bytes(buffer[:n]) # type: ignore[arg-type]

        else: # skip special
            for token in token_ids:
                if token != self.bos_token_id and token != self.eos_token_id:
                    n = llama_cpp.llama_token_to_piece(
                        self._llama.model, llama_cpp.llama_token(token), buffer, size
                    )
                    output += bytes(buffer[:n]) # type: ignore[arg-type]
            

        # "User code is responsible for removing the leading whitespace of the first non-BOS token when decoding multiple tokens."
        if (# token_ids[0] != self.bos_token_id and # we also try cutting if first is bos to approximate HF tokenizer
           len(output) and output[0] <= 32 # 32 = ord(' ')
           ):
            output = output[1:]
                
        return output.decode("utf-8", errors="ignore")
    
        






        


# Avoid "LookupError: unknown encoding: ascii" when open() called in a destructor
outnull_file = open(os.devnull, "w")
errnull_file = open(os.devnull, "w")

class llamacpp_verbosity_manager():
    '''Based on guidance._utils.py (MIT License):
       Remaps stdout and stderr back to their normal selves from what ipykernel did to them.
       Also based on: 
        https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/_utils.py
        https://github.com/ipython/ipykernel/issues/795
       '''
    
    # NOTE: these must be "saved" here to avoid exceptions when using
    #       this context manager inside of a __del__ method
    sys = sys
    os = os

    def __init__(self, 
                 verbose: bool):
        self._verbose = verbose

    def __enter__(self):
        if not self._verbose:

            """avoids most llama.cpp stdout/err output"""
            normal_stdout = self.sys.__stdout__.fileno()
            self.restore_stdout = None            
            if getattr(self.sys.stdout, "_original_stdstream_copy", normal_stdout) != normal_stdout:
                # print("stdout _original_stdstream_copy")
                self.restore_stdout = self.sys.stdout._original_stdstream_copy
                self.sys.stdout._original_stdstream_copy = normal_stdout

            normal_stderr = self.sys.__stderr__.fileno()
            self.restore_stderr = None
            if getattr(self.sys.stderr, "_original_stdstream_copy", normal_stderr) != normal_stderr:
                # print("stderr _original_stdstream_copy")
                self.restore_stderr = self.sys.stderr._original_stdstream_copy
                self.sys.stderr._original_stdstream_copy = normal_stderr

            """avoids stderr output like:
            ggml_backend_cuda_buffer_type_alloc_buffer: allocating 4095.06 MiB on device 0: cudaMalloc failed: out of memory"""
            # check if sys.stdout and sys.stderr have fileno method
            if hasattr(self.sys.stdout, 'fileno') and hasattr(self.sys.stderr, 'fileno'):
                # print("fileno")
                self.old_stdout_fileno_undup = self.sys.stdout.fileno()
                self.old_stderr_fileno_undup = self.sys.stderr.fileno()

                self.old_stdout_fileno = self.os.dup(self.old_stdout_fileno_undup)
                self.old_stderr_fileno = self.os.dup(self.old_stderr_fileno_undup)

                self.old_stdout = self.sys.stdout
                self.old_stderr = self.sys.stderr

                self.os.dup2(outnull_file.fileno(), self.old_stdout_fileno_undup)
                self.os.dup2(errnull_file.fileno(), self.old_stderr_fileno_undup)

                self.sys.stdout = outnull_file
                self.sys.stderr = errnull_file


    def __exit__(self, exc_type, exc_value, traceback):
        if not self._verbose:

            if self.restore_stdout is not None:
                self.sys.stderr._original_stdstream_copy = self.restore_stdout
            if self.restore_stderr is not None:
                self.sys.stderr._original_stdstream_copy = self.restore_stderr

            # check if sys.stdout and sys.stderr have fileno method
            if hasattr(self.sys.stdout, 'fileno') and hasattr(self.sys.stderr, 'fileno'):
                self.sys.stdout = self.old_stdout
                self.sys.stderr = self.old_stderr

                self.os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
                self.os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

                self.os.close(self.old_stdout_fileno)
                self.os.close(self.old_stderr_fileno)









"""Read metadata (like ctx_len value which is in "*.context_length" key) from a GGUF file's metadata.
Directly parses from binary file, without loading model via Llama. After model is loaded these values are available through get_metadata().
Based on llama.cpp's source and https://github.com/oobabooga/text-generation-webui/blob/main/modules/metadata_gguf.py
"""

GGUF_MAGIC = 0x46554747 # define GGUF_MAGIC "GGUF" => 0x46554747

GGUF_UNPACK_FORMAT = {
    0: ("<B", 1), # uint8
    1: ("<b", 1), # int8
    2: ("<H", 2), # uint16
    3: ("<h", 2), # int16
    4: ("<I", 4), # uint32
    5: ("<i", 4), # int32
    6: ("<f", 4), # float32
    7: ("?",  1), # bool
    # 8=str, 9=array
    10: ("<Q", 8), # uint64 
    11: ("<q", 8), # int64
    12: ("<d", 8), # float64
}
def read_gguf_item(item_type: int, 
                   f: BinaryIO):
    
    if item_type == 8: # str
        item_len = struct.unpack("<Q", f.read(8))[0]
        item_bytes = f.read(item_len)
        item = item_bytes.decode("utf-8", "ignore")
            
    else:
        format, byte_size = GGUF_UNPACK_FORMAT.get(item_type) # type: ignore
        item = struct.unpack(format, f.read(byte_size))[0]

    return item


def load_gguf_metadata(path: str,
                       include_arrays: bool = False) -> Union[dict,None]:
    
    out = {}

    with open(path, "rb") as f:

        magic_marker = struct.unpack("<I", f.read(4))[0]
        if magic_marker != GGUF_MAGIC:
            raise RuntimeError("Not a GGUF file")
        
        version = struct.unpack("<I", f.read(4))[0]
        if version < 2:
            raise RuntimeError("Unsupported older GGUF format")
        
        f.read(8) # count skip
        
        kv_count = struct.unpack("<Q", f.read(8))[0]

        for i in range(kv_count):
            key_len = struct.unpack("<Q", f.read(8))[0]
            key_bytes = f.read(key_len)
            key = key_bytes.decode("utf-8", "ignore")

            item_type = struct.unpack("<I", f.read(4))[0]
            
            if item_type == 9: # array of type
                item_type = struct.unpack("<I", f.read(4))[0]
                item_len = struct.unpack("<Q", f.read(8))[0]

                arr = []
                for i in range(item_len):
                    item = read_gguf_item(item_type, f)
                    arr.append(item)
    
                if include_arrays:
                    out[key] = arr
                
            else: # non-array type
                item = read_gguf_item(item_type, f)
                out[key] = item
                
            # print(key, item_type)

    return out




def extract_sub_paths(in_path: str) -> list[str]:
    """Extract multiple paths in filename1*filename2"""

    out = []
    path_list = in_path.split("*")

    curr_dir = ""

    for path in path_list:
        dir_part = os.path.dirname(path)
        if dir_part: # has dir
            curr_dir = dir_part
        else: # filename only: append curr_dir
            path = os.path.join(curr_dir, path)
        out.append(path)
            
    return out


# Based on llamacpp-python's function of the same name in llama_chat_format.py
def split_text_on_image_urls(text: str, 
                             image_urls: list[str]):

    def find_first(s: str, substrs: list[str]):
        for i, substr in enumerate(substrs):
            pos = s.find(substr)
            if pos != -1:
                return pos, i
        return None, None

    split_text: list[tuple[Literal["text", "image_url"], str]] = []

    remaining = text
    while remaining:
        # find next image_url
        pos, i = find_first(remaining, image_urls)
        if pos is not None and i is not None:
            if pos > 0:
                split_text.append(("text", remaining[:pos]))
                
            split_text.append(("image_url", image_urls[i]))
            remaining = remaining[pos + len(image_urls[i]):]
        else:
            split_text.append(("text", remaining))
            remaining = ""

    return split_text


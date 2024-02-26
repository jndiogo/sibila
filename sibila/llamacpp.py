"""Local model access via the llama.cpp engine.

- LlamaCppModel: Use local GGUF format models via llama.cpp engine.
- LlamaCppTokenizer: Tokenizer for llama.cpp loaded GGUF models.
"""


from typing import Any, Optional, Union

import sys, os, json, ctypes
from copy import copy

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

from .json_schema import JSchemaConf

from .json_grammar import (
    gbnf_from_json_schema,
    JSON_GBNF
)

try:
    from llama_cpp import (
        Llama,
        llama_cpp,
        llama_token_get_text,
        llama_grammar
    )
    has_llama_cpp = True
except ImportError:
    has_llama_cpp = False






class LlamaCppModel(FormattedTextModel):
    """Use local GGUF format models via llama.cpp engine.
    
    Supports grammar-constrained JSON output following a JSON schema.

    Attributes:
        ctx_len: Maximum context length, shared for input + output.
        desc: Model information.
    """

    _llama: Llama
    """LlamaCpp instance"""

    def __init__(self,
                 path: str,

                 format: Optional[Union[str,dict]] = None,                 
                 format_search_order: list[str] = ["name","meta_template"],

                 *,

                 # common base model args
                 genconf: Optional[GenConf] = None,
                 schemaconf: Optional[JSchemaConf] = None,
                 tokenizer: Optional[Tokenizer] = None,
                 ctx_len: int = 2048,

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
            format_search_order: Search order for auto-detecting format, "name" searches in the filename, "meta_template" looks in the model's metadata. Defaults to ["name","meta_template"].
            genconf: Default generation configuration, which can be used in gen() and related. Defaults to None.
            tokenizer: An external initialized tokenizer to use instead of the created from the GGUF file. Defaults to None.
            ctx_len: Maximum context length to be used (shared for input and output). Defaults to 2048.
            n_gpu_layers: Number of model layers to run in a GPU. Defaults to -1 for all.
            main_gpu: Index of the GPU to use. Defaults to 0.
            n_batch: Prompt processing batch size. Defaults to 512.
            seed: Random number generation seed, for non zero temperature inference. Defaults to 4294967295.
            verbose: Emit (very) verbose output. Defaults to False.

        Raises:
            ImportError: If llama-cpp-python is not installed.
            ValueError: If ctx_len is 0 or larger than the values supported by model.
        """

        self._llama = None # type: ignore[assignment]
        self.tokenizer = None # type: ignore[assignment]

        if not has_llama_cpp:
            raise ImportError("Please install llama-cpp-python by running: pip install llama-cpp-python")

        if ctx_len == 0:
            raise ValueError("LlamaCppModel doesn't support ctx_len=0")
        
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
        
        logger.debug(f"Creating Llama with model_path='{path}', llamacpp_kwargs={llamacpp_kwargs}")

        with normalize_notebook_stdout_stderr(not verbose):
            self._llama = Llama(model_path=path, **llamacpp_kwargs)

        self._model_path = path
        
        # correct super __init__ values
        self._ctx_len = self._llama.n_ctx()
        
        n_ctx_train = self._llama._model.n_ctx_train()        
        if self.ctx_len > n_ctx_train:
            raise ValueError(f"ctx_len ({self.ctx_len}) is greater than n_ctx_train ({n_ctx_train})")

        
        if self.tokenizer is None:
            self.tokenizer = LlamaCppTokenizer(self._llama)

        try:
            self.init_format(format,
                             format_search_order,
                             {"name": os.path.basename(self._model_path),
                              "meta_template_name": "tokenizer.chat_template"}
                             )
        except Exception as e:
            del self.tokenizer
            del self._llama
            raise e
            
        
   
    

    def __del__(self):
        if self.tokenizer:
            del self.tokenizer
        if self._llama:
            del self._llama

    
    
    
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

        token_ids = self.tokenizer.encode(text)

        if genconf is None: 
            genconf = self.genconf

        if genconf.max_tokens == 0:
            genconf = genconf(max_tokens=self.ctx_len - len(token_ids))
            
        elif len(token_ids) + genconf.max_tokens > self.ctx_len:
            logger.warn(f"Token length + genconf.max_tokens ({len(token_ids) + genconf.max_tokens}) is greater than model's context window length ({self.ctx_len})")

        # prepare llamaCpp args:
        genconf_kwargs = genconf.asdict()
        
        format = genconf_kwargs.pop("format")
        if format == "json":
            if genconf_kwargs["json_schema"] is None:
                grammar = llama_grammar.LlamaGrammar.from_string(JSON_GBNF, 
                                                                 logger.getEffectiveLevel() == logging.DEBUG)
                    
            else: # translate json_schema to a llama grammar
                jsg = gbnf_from_json_schema(genconf_kwargs["json_schema"])
                logger.debug(f"JSON schema GBNF grammar:\n{jsg}")
                
                grammar = llama_grammar.LlamaGrammar.from_string(jsg, 
                                                                 logger.getEffectiveLevel() == logging.DEBUG)

            genconf_kwargs["grammar"] = grammar
            
        genconf_kwargs.pop("json_schema")

        logger.debug(f"LlamaCpp args: {genconf_kwargs}")
        
        # Llamacpp.create_completion() never returns special tokens because it uses its own detokenize()
        response = self._llama.create_completion(token_ids,
                                                 echo=False,
                                                 stream=False,
                                                 **genconf_kwargs)
        logger.debug(f"LlamaCpp response: {response}")

        choice = response["choices"][0] # type: ignore[index]
        return choice["text"], choice["finish_reason"] # type: ignore[return-value]


        
    @classmethod
    def provider_version(_) -> str:
        """Provider library version: provider x.y.z
        Ex. llama-cpp-python 0.2.44
        """
        try:        
            import llama_cpp
            ver = llama_cpp.__version__
        except Exception:
            raise ImportError("Please install llama-cpp-python by running: pip install llama-cpp-python")
            
        return f"llama-cpp-python {ver}"

    
    @property
    def desc(self) -> str:
        """Model description."""
        return f"LlamaCppModel: {self._model_path} - '{self._llama._model.desc()}'"

    
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
                 llama: Llama, 
                 reg_flags: Optional[str] = None):
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

        # workaround for https://github.com/ggerganov/llama.cpp/issues/4772
        self._workaround1 = reg_flags is not None and "llamacpp1" in reg_flags
            

    
    def encode(self, 
               text: str) -> list[int]:
        """Encode text into model tokens. Inverse of Decode().

        Args:
            text: Text to be encoded.

        Returns:
            A list of ints with the encoded tokens.
        """

        if self._workaround1:
            # append a space after each bos and eos, so that llama's tokenizer matches HF
            def space_post(text, s):
                out = ""
                while (index := text.find(s)) != -1:
                    after = index + len(s)
                    out += text[:after]
                    if text[after] != ' ':
                        out += ' '
                    text = text[after:]
                        
                out += text
                return out

            text = space_post(text, self.bos_token)
            text = space_post(text, self.eos_token)
            # print(text)
        
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
    
        




class normalize_notebook_stdout_stderr():
    '''
    Based on guidance._utils.py (MIT License):
        Remaps stdout and stderr back to their normal selves from what ipykernel did to them.
        Based on: https://github.com/ipython/ipykernel/issues/795
    '''
    def __init__(self, 
                 enabled: bool = True):
        self._enabled = enabled

    def __enter__(self):
        if self._enabled:
            normal_stdout = sys.__stdout__.fileno()
            self.restore_stdout = None
            if getattr(sys.stdout, "_original_stdstream_copy", normal_stdout) != normal_stdout:
                self.restore_stdout = sys.stdout._original_stdstream_copy
                sys.stdout._original_stdstream_copy = normal_stdout
    
            normal_stderr = sys.__stderr__.fileno()
            self.restore_stderr = None
            if getattr(sys.stderr, "_original_stdstream_copy", normal_stderr) != normal_stderr:
                self.restore_stderr = sys.stderr._original_stdstream_copy
                sys.stderr._original_stdstream_copy = normal_stderr

            self.old_stdout = sys.stdout
            self.old_stderr = sys.stderr
    
            # sys.stdout = open(os.devnull, 'w')
            # sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        if self._enabled:
            # sys.stdout = self.old_stdout
            # sys.stderr = self.old_stderr

            if self.restore_stdout is not None:
                sys.stderr._original_stdstream_copy = self.restore_stdout
            if self.restore_stderr is not None:
                sys.stderr._original_stdstream_copy = self.restore_stderr


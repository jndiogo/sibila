"""Model generation configurations and results.

- GenConf: Model generation configuration, used in Model.gen() and variants.
- GenRes: Generaion result enum.
- GenError: Generation exception.
- GenOut: Results of model generation.
"""

from typing import Any, Optional, Union, Callable
from typing_extensions import Self
from dataclasses import dataclass, field, asdict
from enum import IntEnum

import json
from copy import copy

import logging
logger = logging.getLogger(__name__)




@dataclass
class GenConf:
    """Model generation configuration, used in Model.gen() and variants."""
    
    max_tokens: int = 0 
    """Maximum output token length. Special value of 0 means all available context length, special values between -1 and -100 mean a -percentage of ctx_len. For example -20 allows output up to 20% of ctx_len."""
    
    stop: Union[str, list[str]] = field(default_factory=list)
    """List of generation stop text sequences"""
    
    temperature: float = 0.
    """Generation temperature. Use 0 to always pick the most probable output, without random sampling. Larger positive values will produce more random outputs."""

    top_p: float = 0.9
    """Nucleus sampling top_p value. Only applies if temperature > 0."""

    # seed config is disabled, has remote models and some hardware accelerated local models don't support it.
    # seed: Union[int,None] = None
    # """Numeric seed for token sampling. Special values: None for not setting it, -1 to pick a random seed. Only applies if temperature > 0."""

    format: str = "text"
    """Output format: "text" or "json". For JSON output, text is validated as in json.loads().
    Thread msgs must explicitly request JSON output or a warning will be emitted if string json not present
    (this is automatically done in Model.json() and related calls).
    """

    json_schema: Union[str,dict,None] = None
    """A JSON schema to validate the JSON output.
    Thread msgs must list the JSON schema and request its use; must also set the format to "json".
    """
    
    
    def __call__(self,
                 **kwargs: Any) -> Self:
        """Return a copy of the current GenConf updated with values in kwargs. Doesn't modify object.

        Args:
            **kwargs: update settings of the same names in the returned copy.

        Raises:
            KeyError: If key does not exist.

        Returns:
            A copy of the current object with kwargs values updated. Doesn't modify object.
        """

        ret = copy(self)

        for k,v in kwargs.items():
            if not hasattr(ret, k):
                raise KeyError(f"No such key '{k}'")
            setattr(ret, k,v)

        return ret


    def clone(self) -> Self:
        """Return a copy of this configuration."""
        return copy(self)
        
    def as_dict(self) -> dict:
        """Return GenConf as a dict."""
        return asdict(self)

    @staticmethod
    def from_dict(dic: dict) -> Any: # Any = GenConf
        return GenConf(**dic)

    def resolve_max_tokens(self,
                           ctx_len: int,
                           max_tokens_limit: Optional[int] = None) -> int:
        """Calculate actual max_tokens value for cases where it's zero or a percentage of model's ctx_len)

        Args:
            ctx_len: Model's context length.
            max_tokens_limit: Optional model's limit for max_tokens. Defaults to None.

        Returns:
            An actual model maximum number of output tokens.
        """

        max_tokens = self.max_tokens
        if max_tokens <= 0:
            if max_tokens == 0:
                max_tokens = ctx_len
            else:
                max_tokens = min(-max_tokens, 100)
                max_tokens = int(max_tokens / 100.0 * ctx_len)
                max_tokens = max(1,max_tokens)
        if max_tokens_limit is not None:
            max_tokens = min(max_tokens, max_tokens_limit)

        return max_tokens





class GenRes(IntEnum):
    """Model generation result."""

    OK_STOP = 1 
    """Generation complete without errors."""

    OK_LENGTH = 0
    """Generation stopped due to reaching max_tokens."""

    ERROR_JSON = -1 
    """Invalid JSON: this is often due to the model returning OK_LENGTH (finished due to max_tokens reached), which cuts off the JSON text."""

    ERROR_JSON_SCHEMA_VAL = -2
    """Failed JSON schema validation."""

    ERROR_JSON_SCHEMA_ERROR = -2
    """JSON schema itself is not valid."""

    ERROR_MODEL = -3
    """Other model internal error."""


    @staticmethod
    def from_finish_reason(finish: str) -> Any: # Any=GenRes
        """Convert a ChatCompletion finish result into a GenRes.

        Args:
            finish: ChatCompletion finish result.

        Returns:
            A GenRes result.
        """
        if finish == 'stop':
            return GenRes.OK_STOP
        elif finish == 'length':
            return GenRes.OK_LENGTH
        elif finish == '!json':
            return GenRes.ERROR_JSON
        elif finish == '!json_schema_val':
            return GenRes.ERROR_JSON_SCHEMA_VAL
        elif finish == '!json_schema_error':
            return GenRes.ERROR_JSON_SCHEMA_ERROR
        else:
            return GenRes.ERROR_MODEL
           
    @staticmethod
    def as_text(res: Any) -> str: # Any=GenRes
        """Returns a friendlier description of the result.

        Args:
            res: Model output result.

        Raises:
            ValueError: If unknown GenRes.

        Returns:
            A friendlier description of the GenRes.
        """

        if res == GenRes.OK_STOP:
            return "Stop"
        elif res == GenRes.OK_LENGTH:
            return "Length (output cut)"
        elif res == GenRes.ERROR_JSON:
            return "JSON decoding error"
            
        elif res == GenRes.ERROR_JSON_SCHEMA_VAL:
            return "JSON SCHEMA validation error"
        elif res == GenRes.ERROR_JSON_SCHEMA_ERROR:
            return "Error in JSON SCHEMA"
            
        elif res == GenRes.ERROR_MODEL:
            return "Model internal error"
        else:
            raise ValueError("Bad/unknow GenRes")



@dataclass
class GenOut:
    """Model output, returned by gen_extract(), gen_json() and other model calls that don't raise exceptions."""
    
    res: GenRes
    """Result of model generation."""
    
    text: str
    """Text generated by model."""
    
    dic: Union[dict,None] = None
    """Python dictionary, output by the structured calls like gen_json()."""

    value: Union[Any, None] = None # Any = accepted type instances, dataclass or Pydantic BaseModel object
    """Initialized instance value, dataclass or Pydantic BaseModel object, as returned in calls like extract()."""
    
    
    def as_dict(self):
        """Return GenOut as a dict."""
        return asdict(self)

    def __str__(self):
        out = f"Error={self.res.as_text(self.res)} text=█{self.text}█"
        if self.dic is not None:
            out += f" dic={self.dic}"
        if self.value is not None:
            out += f" value={self.value}"
        return out




class GenError(RuntimeError, GenOut):
    """Model generation exception, raised when the model was unable to return a response."""

    def __init__(self, 
                 out: GenOut):
        """An error has happened during model generation.

        Args:
            out: Model output
        """

        assert out.res != GenRes.OK_STOP, "OK_STOP is not an error"      

        super().__init__()

        self.res = out.res
        self.text = out.text
        self.dic = out.dic
        self.value = out.value


    @staticmethod
    def raise_if_error(out: GenOut,
                       ok_length_is_error: bool):
        """Raise an exception if the model returned an error

        Args:
            out: Model returned info.
            ok_length_is_error: Should a result of GenRes.OK_LENGTH be considered an error?

        Raises:
            GenError: If an error was returned by model.
        """
        
        if out.res != GenRes.OK_STOP:
            if out.res == GenRes.OK_LENGTH and not ok_length_is_error:
                return # OK_LENGTH to not be considered an error

            raise GenError(out)
            
        
    def __str__(self):
        out = f"Error={self.res.as_text(self.res)} text=█{self.text}█"
        if self.dic is not None:
            out += f" dic={self.dic}"
        if self.value is not None:
            out += f" value={self.value}"
        return out





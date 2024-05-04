"""Structured data from local or remote LLM models."""

__version__ = "0.4.2"

__all__ = [
    "Models",
    "Model", "TextModel", "MessagesModel", "Tokenizer",
    "GenConf", "GenRes", "GenError", "GenOut",
    "AnthropicModel",
    "FireworksModel",
    "LlamaCppModel", "LlamaCppTokenizer",
    "MistralModel",
    "OpenAIModel", "OpenAITokenizer",
    "TogetherModel",
    "Thread", "MsgKind",
    "Context", "Trim",
    "JSchemaConf",
    "TDesc"
]

__author__ = "Jorge Diogo"


from .gen import (
    GenConf,
    GenRes,
    GenError,
    GenOut
)

from .thread import (
    Thread,
    MsgKind,
)

from .model import (
    Model,
    TextModel,
    FormattedTextModel,
    MessagesModel,
    Tokenizer
)


from .anthropic import AnthropicModel

from .llamacpp import (
    LlamaCppModel,
    LlamaCppTokenizer
)

from .mistral import MistralModel

from .openai import (
    OpenAIModel,
    OpenAITokenizer
)

from .schema_format_openai import (
    TogetherModel,
    FireworksModel
)


from .context import (
    Context,
    Trim
)

from .models import Models

from .json_schema import JSchemaConf

""" A shorter 'Annotated', meant to be used as: TDesc[type, "description"] """
from typing import Annotated
TDesc = Annotated

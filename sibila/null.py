"""Null model(s) for testing.

Created later if  needed: NullFormattedTextModel, NullMessagesModel
"""


from typing import Any, Optional, Union

import sys, os, json, ctypes
from copy import copy

import logging
logger = logging.getLogger(__name__)


from .gen import (
    GenConf,
    GenOut
)

from .thread import (
    Thread
)

from .model import (
    Model,
    MessagesModel,
    FormattedTextModel,
    Tokenizer
)

from .json_schema import JSchemaConf




class NullModel(Model):

    def __init__(self,
                 is_local_model: bool = True,
                 is_message_model: bool = False,

                 genconf: Optional[GenConf] = None,
                 schemaconf: Optional[JSchemaConf] = None
                 ):

        super().__init__(is_local_model,
                         genconf,
                         schemaconf,
                         None)
        
        self.is_message_model = is_message_model

        self.response = {
            "text": "",
            "finish": "stop"
        }


    def set_response(self,
                     text: str,
                     finish="stop"):
        """
        finish: stop, length
        """
        self.response["text"] = text
        self.response["finish"] = finish



    def gen(self, 
            thread: Thread,
            genconf: Optional[GenConf] = None,
            ) -> GenOut:
        
        if genconf is None:
            genconf = self.genconf

        out = self._prepare_gen_out(self.response["text"],
                                    self.response["finish"], 
                                    genconf)

        return out



    def token_len(self,
                  thread: Thread,
                  _: Optional[GenConf] = None) -> int:
        assert False, "NullModel doesn't use tokens"


    @classmethod
    def provider_version(_) -> str:
        """Provider library version: provider x.y.z
        Ex. openai 1.3.6
        """
        return "NullModel 0.0.0"

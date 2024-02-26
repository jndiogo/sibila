from typing import Any, Optional, Union
from enum import IntFlag
from collections.abc import Sequence

import json
from copy import copy
from pprint import pformat

import logging
logger = logging.getLogger(__name__)

from .model import (
    GenConf,
    Model,    
)

from .thread import (
    Thread,
    MsgKind
)



class Trim(IntFlag):
    """Flags for Thread trimming."""

    NONE = 0
    """No trimming."""

    INST = 1
    """Can remove INST message."""

    IN = 2
    """Can remove IN messages."""

    OUT = 4
    """Can remove OUT messages."""
        
    KEEP_FIRST_IN = 1024
    """If trimming IN messages, never remove first one."""

    KEEP_FIRST_OUT = 2048
    """If trimming OUT messages, never remove first one."""



class Context(Thread):

    """A class based on Thread that manages total token length, so that it's kept under a certain value.
    Also supports a persistent inst (instructions) text."""

    
    def __init__(self,                 
                 t: Optional[Union[Thread,list,str,dict,tuple]] = None, 
                 max_token_len: Optional[int] = None,        
                 pinned_inst_text: str = "",
                 join_sep: str = "\n"):
        """
        Args:
            t: Can initialize from a Thread, from a list (containing messages in any format accepted in _parse_msg()) or a single message as an str, an (MsgKind,text) tuple or a dict. Defaults to None.
            max_token_len: Maximum token count to use when trimming. Defaults to None, which will use max model context length.
            pinned_inst_text: Pinned inst text which survives clear(). Defaults to "".
            join_sep: Separator used when message text needs to be joined. Defaults to "\\n".
        """

        super().__init__(t,
                         inst=pinned_inst_text,
                         join_sep=join_sep)
        
        self.max_token_len = max_token_len

        self.pinned_inst_text = pinned_inst_text
        
   
    
    def clear(self):
        """Delete all messages but reset inst to a pinned text if any."""
        super().clear()        
        if self.pinned_inst_text is not None:
            self.inst = self.pinned_inst_text
            


    def trim(self,
             trim_flags: Trim,
             model: Model,
             *,
             max_token_len: Optional[int] = None,
             ) -> bool:
        """Trim context by selectively removing older messages until thread fits max_token_len.

        Args:
            trim_flags: Flags to guide selection of which messages to remove.
            model: Model that will process the thread.
            max_token_len: Cut messages until size is lower than this number. Defaults to None.

        Raises:
            RuntimeError: If unable to trim anything.

        Returns:
            True if any context trimming occurred.
        """

        if max_token_len is None:
            max_token_len = self.max_token_len
            
        if max_token_len is None:
            max_token_len = model.ctx_len

        # if genconf is None:
        #     genconf = model.genconf            
        # assert max_token_len < model.ctx_len, f"max_token_len ({max_token_len}) must be < model's context size ({model.ctx_len}) - genconf.max_new_tokens"
        
        if trim_flags == Trim.NONE: # no trimming
            return False
        
        thread = self.clone()

        any_trim = False
        
        while True:

            curr_len = model.token_len(thread)

            if curr_len <= max_token_len:
                break

            logger.debug(f"len={curr_len} / max={max_token_len}")

            if self.inst and trim_flags & Trim.INST:
                self.inst = ''
                any_trim = True
                logger.debug(f"Cutting INST {self.inst[:80]} (...)")
                continue

            # cut first possible message, starting from oldest first ones
            trimmed = False
            in_index = out_index = 0

            for index,m in enumerate(thread):
                kind,text = m

                if kind == MsgKind.IN:
                    if trim_flags & Trim.IN:
                        if not (trim_flags & Trim.KEEP_FIRST_IN and in_index == 0):
                            del thread[index]
                            trimmed = True
                            logger.debug(f"Cutting IN {text[:80]} (...)")
                            break
                    in_index += 1

                elif kind == MsgKind.OUT:
                    if trim_flags & Trim.OUT:                        
                        if not (trim_flags & Trim.KEEP_FIRST_OUT and out_index == 0):
                            del thread[index]
                            trimmed = True
                            logger.debug(f"Cutting OUT {text[:80]} (...)")
                            break
                    out_index += 1
                
            if not trimmed:
                # all thread messages were cycled but not a single could be cut, so size remains the same
                # arriving here we did all we could for trim_flags but could not remove any more
                raise RuntimeError("Unable to trim anything out of thread")
            else:
                any_trim = True

        # while end
        

        if any_trim:
            self._msgs = thread._msgs
            
        return any_trim




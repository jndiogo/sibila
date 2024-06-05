"""Text chunk splitter for RAG-like applications."""


from typing import Any, Optional, Union, Callable
from copy import copy
import re

from .thread import Thread
from .gen import GenConf


class RecursiveTextSplitter:
    """
    When using a token_len len_fn, returned chunks might be sightly smaller than chunk size, if one of the separators is space, because tokens may already assume space as a prefix.
    """

    chunk_size:int 
    chunk_overlap: int
    seps: list[str]
    len_fn: Callable[[str], int]

    def __init__(self,
                 chunk_size: int,
                 chunk_overlap: int = 0,
                 seps: Optional[list[str]] = None,
                 len_fn: Optional[Callable[[str], int]] = None
                 ):
        """
        chunk_size and chunk_overlap are in whatever units len_fn returns (chars or tokens for example).
        """
        
        if chunk_overlap >= (chunk_size // 2):
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than half the chunk_size ({chunk_size})")

        if chunk_size < 3:
            raise ValueError("chunk_size must be at least 3")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.len_fn = len_fn or len # type: ignore[assignment]
        
        self.seps = seps or ["\n\n", "\n", " ", ""]
        if self.seps[-1] != "":
            self.seps.append("")
        


    def split(self,
              text: Optional[Union[str,list[str]]] = None,
              path: Optional[Union[str,list[str]]] = None,
              seps: Optional[list[str]] = None,
              len_fn: Optional[Callable[[str], int]] = None
              ) -> list[str]:

        if (text is not None) + (path is not None) != 1:
            raise ValueError("Only one of text or path can be given")
        
        if path is not None:
            if isinstance(path, str):
                path = [path]
            text = []
            for p in path:
                with open(p, "r", encoding="utf-8") as f:
                    text.append(f.read()) 
        elif isinstance(text, str):
            text = [text]
                
        out = []

        for t in text:  # type: ignore[union-attr]
            out += self._split(t, seps, len_fn)
    
        return out
    


    

    def _split(self,
               text: str,
               seps: Optional[list[str]] = None,
               len_fn: Optional[Callable[[str], int]] = None
               ) -> list[str]:
        # print(text, out)
        
        seps = seps or self.seps
        assert seps and seps[-1] == ""

        sep = seps[0]
        if sep != "":
            spl = text.split(sep)
        else:
            spl = list(text)

        len_fn = len_fn or self.len_fn
        sep_len = len_fn(sep)
        
        size, over = self.chunk_size, self.chunk_overlap

        out = []
        accum = "" # only accumulates up to size
        while spl:
            head = spl[0]
            if head: # avoid empty
           
                if len_fn(accum) + sep_len + len_fn(head) <= size: 
                    # fits in accum: merge
                    if accum:
                        accum += sep
                    accum += head
                    
                elif accum and len_fn(head) <= (size - over + sep_len): 
                    # can emit accum and set accum=overlap+head
                    out.append(accum)
                    if over: # get chunk overlap from accum
                        # backtrack along sep till len() >= over size
                        if sep != "":
                            over_text = ""
                            ar = accum.split(sep)
                            for r in ar[-1::-1]:
                                if len_fn(r) + (sep_len * int(bool(over_text))) + len_fn(over_text) >= over:
                                    break
                                else:
                                    if over_text:
                                        over_text = sep + over_text
                                    over_text = r + over_text
                        else:
                            over_text = accum[-over:]
                        
                        head = over_text + sep + head
                    
                    accum = head
                    
                else: # head doesn't fit: recurse into next separator
                    if accum: # emit accum if any
                        out.append(accum)
                        accum = ""
                        
                    out += self._split(head, seps[1:])
            
            spl = spl[1:]
            
        if accum: # emit accum if any
            out.append(accum)

        return out

            
            

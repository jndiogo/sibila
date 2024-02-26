"""Threads of messages are the way to communicate with models. 
For a quick extraction or classification, a thread with a simple IN message and possibly model instructions can be used.
For more sophisticated queries that depend on previous interactions, IN (user or input) and OUT (model response) messages can be added.

- Msg: A message of type IN (user query), OUT (assistant answer) or SYS (initial assistant instructions).
- Thread: A sequence of Msgs alternating between user and model.
"""

from typing import Any, Optional, Union, Callable
from typing_extensions import Self
from enum import IntEnum
from collections.abc import Sequence

from copy import copy, deepcopy
import json

import logging
logger = logging.getLogger(__name__)




class MsgKind(IntEnum):
    """Enumeration for kinds of messages in a Thread."""

    IN = 0
    """Input message, from user."""

    OUT = 1
    """Model output message."""

    INST = 2
    """Initial instructions for model."""

    @staticmethod
    def kind_from_chatml_role(role: str) -> Any: # Any=MsgKind
        KIND_FROM_CHATML: dict = {"user": MsgKind.IN, "assistant": MsgKind.OUT, "system": MsgKind.INST}
        kind = KIND_FROM_CHATML.get(role)
        if kind is None:
            raise ValueError(f"Unknown ChatML role '{role}'.")
        else:
            return kind

    @staticmethod
    def chatml_role_from_kind(kind: Any) -> str: # Any=MsgKind
        CHATML_FROM_KIND: dict = {MsgKind.IN: "user", MsgKind.OUT: "assistant", MsgKind.INST: "system"}
        return CHATML_FROM_KIND.get(kind) # type: ignore[return-value]




class Thread(Sequence):
    """A sequence of messages alternating between IN ("user" role) and OUT ("assistant" role).

    Stores a special initial INST information (known as "system" role in ChatML) providing instructions to the model.
    Some models don't use system instructions - in those cases it's prepended to first IN message.

    Messages are kept in a strict IN,OUT,IN,OUT,... order. To enforce this, if two IN messages are added, the second just appends to the text of the first.
    
    Attributes:
        inst: Text for system instructions.
    """

    inst: str
    """Text for system instructions, defaults to empty string"""

    join_sep: str
    """Separator used when message text needs to be joined. Defaults to '\\n'"""

    _msgs: list[str]
    """List of thread messages"""
    

    def __init__(self,
                 t: Optional[Union[Any,list,str,dict,tuple]] = None, # Any=Thread
                 inst: str = '',
                 join_sep: str = "\n"):
        """
        Examples:
            Creation with message list

            >>> from sibila import Thread, MsgKind
            >>> th = Thread([(MsgKind.IN, "Hello model!"), (MsgKind.OUT, "Hello there human!")],
            ...             inst="Be helpful.")
            >>> print(th)
            inst=█Be helpful.█, sep='\\n', len=2
            0: IN=█Hello model!█
            1: OUT=█Hello there human!█

            Adding messages

            >>> from sibila import Thread, MsgKind
            >>> th = Thread(inst="Be helpful.")
            >>> th.add(MsgKind.IN, "Can you teach me how to cook?")
            >>> th.add_IN("I mean really cook as a chef?") # gets appended
            >>> print(th)
            inst=█Be helpful.█, sep='\\n', len=1
            0: IN=█Can you teach me how to cook?\\nI mean really cook as a chef?█

            Another way to add a message

            >>> from sibila import Thread, MsgKind
            >>> th = Thread(inst="Be informative.")
            >>> th.add_IN("Tell me about kangaroos, please?")
            >>> th += "They are so impressive." # appends text to last message
            >>> print(th)
            inst=█Be informative.█, sep='\\n', len=1
            0: IN=█Tell me about kangaroos, please?\\nThey are so impressive.█
            
            As a ChatML message list
            >>> from sibila import Thread, MsgKind
            >>> th = Thread([(MsgKind.IN, "Hello model!"), (MsgKind.OUT, "Hello there human!")], 
            ...             inst="Be helpful.")
            >>> th.as_chatml()
            [{'role': 'system', 'content': 'Be helpful.'},
             {'role': 'user', 'content': 'Hello model!'},
             {'role': 'assistant', 'content': 'Hello there human!'}]

        Args:
            t: Can initialize from a Thread, from a list (containing messages in any format accepted in _parse_msg()) or a single message as an str, an (MsgKind,text) tuple or a dict. Defaults to None.
            join_sep: Separator used when message text needs to be joined. Defaults to "\\n".

        Raises:
            TypeError: On invalid args passed.
        """

        if isinstance(t, Thread):
            self._msgs = t._msgs.copy()
            self.inst = t.inst
            self.join_sep = t.join_sep
        else:
            self._msgs = []
            self.inst = inst
            self.join_sep = join_sep

            if t is not None:
                self.concat(t)



    def clear(self):
        """Delete all messages and clear inst."""
        self.inst = ""
        self._msgs = []
            
    

    @property
    def last_kind(self) -> MsgKind:
        """Get kind of last message in thread .

        Returns:
            Kind of last message or MsgKind.IN if empty.
        """
        length = len(self._msgs)
        if not length: # empty: assume IN
            return MsgKind.IN
        else:
            return Thread._kind_from_pos(length - 1)


    @property
    def last_text(self) -> str:
        """Get text of last message in thread .

        Returns:
            Last message text.

        Raises:
            IndexError: If thread is empty.
        """
        length = len(self._msgs)
        if not length: # empty
            raise IndexError("Thread is empty")
        else:
            return self._msgs[-1]


    def add(self, 
            t: Union[str,tuple,dict,MsgKind],
            text: Optional[str] = None):
        """Add a message to Thread by parsing a mix of types.

        Accepts any of these argument combinations:

        - t=MsgKind, text=str
        - t=str, text=None -> uses last thread message's MsgKind
        - (MsgKind, text)
        - {"kind": "...", text: "..."}
        - {"role": "...", content: "..."} - ChatML format

        Args:
            t: One of the accepted types listed above.
            text: Message text if first type is MsgKind. Defaults to None.
        """
        
        kind, text = self._parse_msg(t, text)

        if kind == MsgKind.INST:
            self.inst = self.join_text(self.inst, text)
        else:
            if kind == self.last_kind and len(self._msgs):
                self._msgs[-1] = self.join_text(self._msgs[-1], text)
            else:
                self._msgs.append(text) # in new kind


            
    def addx(self, 
             path: Optional[str] = None, 
             text: Optional[str] = None,
             kind: Optional[MsgKind] = None):
        """Add message with text from a supplied arg or loaded from a path.

        Args:
            path: If given, text is loaded from an UTF-8 file in this path. Defaults to None.
            text: If given, text is added. Defaults to None.
            kind: MsgKind of message. If not given or the same as last thread message, it's appended to it. Defaults to None.
        """

        assert (path is not None) ^ (text is not None), "Only one of path or text"

        if path is not None:
            with open(path, 'r', encoding="utf-8") as f:
                text = f.read()

        if kind is None: # use last message role, so that it gets appended
            kind = self.last_kind

        self.add(kind, text)



    def get_text(self,
                 index: int) -> str:
        """Return text for message at index.

        Args:
            index: Message index. Use -1 to get inst value.

        Returns:
            Message text at index.
        """        
        if index == -1:
            return self.inst
        else:
            return self._msgs[index]

    def set_text(self,
                 index: int,
                 text: str):        
        """Set text for message at index.

        Args:
            index: Message index. Use -1 to set inst value.
            text: Text to replace in message at index.
        """
        if index == -1:
            self.inst = text
        else:
            self._msgs[index] = text
            

    def concat(self,
               t: Optional[Union[Self,list,str,dict,tuple]]):
        """Concatenate a Thread or list of messages to the current Thread.

        Take care that the other list starts with an IN message, therefore, 
        if last message in self is also an IN kind, their text will be joined as in add().

        Args:
            t: A Thread or a list of messages. Otherwise a single message as in add().

        Raises:
            TypeError: If bad arg types provided.
        """
        if isinstance(t, Thread):
            for msg in t:
                self.add(msg)
            self.inst = self.join_text(self.inst, t.inst)

        elif isinstance(t, list): # message list
            for msg in t:
                self.add(msg)

        elif isinstance(t, str) or isinstance(t, dict) or isinstance(t, tuple): # single message
            self.add(t)

        else:
            raise TypeError("Arg t must be: Thread --or-- list[messages] --or-- an str, tuple or dict single message.")
    



    
    def load(self,
             path: str):
        """Load this Thread from a JSON file.

        Args:
            path: Path of file to load.
        """

        with open(path, 'r', encoding='utf-8') as f:
            js = f.read()
        state = json.loads(js)

        self._msgs = state["_msgs"]
        self.inst = state["inst"]
        self.join_sep = state["join_sep"]

    
    def save(self,
             path: str):
        """Serialize this Thread to JSON.

        Args:
            path: Path of file to save into.
        """
 
        state = {"_msgs": self._msgs,
                 "inst": self.inst,
                 "join_sep": self.join_sep
                 }
    
        json_str = json.dumps(state, indent=2, default=vars)
    
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json_str)

    


    
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = for convenience:
    
    def init_INST_IN(self,
                     inst_text: str,
                     in_text: str):
        """Initialize Thread with instructions and an IN message.

        Args:
            inst_text: Instructions text.
            in_text: Text for IN message.
        """
        self.clear()
        self += self.make_INST_IN(inst_text, in_text)


    def add_IN(self,
               in_text: str):
        """Appends an IN message to Thread.

        Args:
            in_text: Text for IN message.
        """
        self.add(MsgKind.IN, in_text)
    
    def add_OUT(self,
                out_text: str):
        """Appends an OUT message to Thread.

        Args:
            in_text: Text for OUT message.
        """
        self.add(MsgKind.OUT, out_text)

    
    def add_OUT_IN(self,
                   out_text: str,
                   in_text: str):
        """Appends an OUT message followed by an IN message.

        Args:
            out_text: Text for OUT message.
            in_text: Text for IN message.
        """        
        self.add(MsgKind.OUT, out_text)
        self.add(MsgKind.IN, in_text)



    @staticmethod
    def make_INST_IN(inst_text: str,
                     in_text: str) -> Any: # Any=Thread
        """Return an initialized Thread with instructions and an IN message.

        Args:
            inst_text: Instructions text.
            in_text: Text for IN message.
        """

        thread = Thread([(MsgKind.INST, inst_text),
                         (MsgKind.IN, in_text)]
                        )        
        return thread

    @staticmethod
    def make_IN(in_text: str) -> Any: # Any=Thread
        """Return an initialized Thread with an IN message.

        Args:
            in_text: Text for IN message.
        """

        thread = Thread([(MsgKind.IN, in_text)])        
        return thread

    @staticmethod
    def make_OUT_IN(out_text: str,
                    in_text: str) -> Any: # Any=Thread
        """Return an initialized Thread with an OUT message followed by an IN message.

        Args:
            out_text: Text for OUT message.
            in_text: Text for IN message.
        """
        thread = Thread([(MsgKind.OUT, out_text),
                         (MsgKind.IN, in_text)]
                        )
        return thread

    @staticmethod
    def ensure(query: Union[str,Any],
               inst: Optional[str] = None) -> Any: # Any=Thread
        """Utility to return a Thread from either a passed Thread or an str used as an IN message.

        Args:
            query: Thread or an str with the text of a single IN message to use as model input.
            inst: Instruction message for model. Will override Thread's inst. Defaults to None.

        Raises:
            TypeError: Arg query must be of type Thread or str.

        Returns:
            Initialized Thread object.
        """
        
        if isinstance(query, str):
            if inst is None:
                return Thread.make_IN(query)
            else:
                return Thread.make_INST_IN(inst, query)
        elif isinstance(query, Thread):
            if inst is not None:
                query = Thread(query, inst) # a clone
            return query
        else:
            raise TypeError("Arg query must be of type Thread or str")



    def msg_as_chatml(self,
                      index: int) -> dict:
        """Returns message in a ChatML dict.

        Args:
            index: Index of the message to return.

        Returns:
            A ChatML dict with "role" and "content" keys.
        """
        kind = Thread._kind_from_pos(index)
        role = MsgKind.chatml_role_from_kind(kind)
        text = self._msgs[index] if index >= 0 else self.inst
        return {"role": role, "content": text}
    

    def as_chatml(self) -> list[dict]:
        """Returns Thread as a list of ChatML messages.

        Returns:
            A list of ChatML dict elements with "role" and "content" keys.
        """
        msgs = []

        for index,msg in enumerate(self._msgs):
            if index == 0 and self.inst:
                msgs.append(self.msg_as_chatml(-1))
            msgs.append(self.msg_as_chatml(index))
            
        return msgs


    def has_text_lower(self,
                       text_lower: str) -> bool:
        """Can the lowercase text be found in one of the messages?

        Args:
            text_lower: The lowercase text to search for in messages.

        Returns:
            True if such text was found.
        """
        for msg in self._msgs:
            if text_lower in msg.lower():
                return True
            
        return False        
    

    
    def clone(self):
        """Return a copy of current Thread.

        Returns:
            A copy of this Thread.
        """
        return Thread(self)



    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = lower level
    def _parse_msg(self,
                   t: Union[str,tuple,dict,MsgKind],
                   text: Optional[str] = None) -> tuple[MsgKind,str]:
        
        """Parses a mix of types into (MsgKind, text).
        Accepts arg types:
            t:MsgKind, text:str
            t:str, text=None -> returns self.last_kind(), text
            (MsgKind, text)
            {"kind": "...", text: "..."}
            {"role": "...", content: "..."} - ChatML format

        Args:
            t: one of the accepted types.

        Returns:
            Tuple of parsed MsgKind, text
        """
        
        if text is not None: # t must be a kind
            if not isinstance(text, str):
                raise TypeError("If arg text is given, it must be an str.")
            if not isinstance(t, MsgKind):
                raise TypeError("If arg text is given, arg k must be of type MsgKind.")
            return t,text

        elif isinstance(t, str):
            return self.last_kind, t

        elif isinstance(t, tuple):
            if len(t) != 2 or not isinstance(t[0], MsgKind) or not isinstance(t[1], str):
                raise TypeError("Tuple must hold (MsgKind, str).")
            return t[0], t[1]
        
        elif isinstance(t, dict):
            if "role" in t:
                kind = MsgKind.kind_from_chatml_role(t["role"])
                if "content" not in t:
                    raise TypeError("A dict in format ChatML must include a content key.")
                return kind, t["content"]
            
            elif "kind" in t:
                if "text" not in t:
                    raise TypeError("Dict with 'kind' key must also include a 'text' key.")
                return t["kind"], t["text"]

            else:
                raise TypeError("Unknown dict format.")
        else:
            raise TypeError("""Args must be: MsgKind,str --or str,text=None --or-- (MsgKind, text) --or-- {"kind": "...", text: "..."} --or--  {"role": "...", content: "..."}.""")



    @staticmethod                     
    def _kind_from_pos(pos: int) -> MsgKind:
        if pos == -1:
            return MsgKind.INST
        else:
            return MsgKind.OUT if pos % 2 else MsgKind.IN


    def join_text(self,
                  a: str,
                  b: str,
                  sep_count: int = 1) -> str:        
        if b:
            if a:
                if a[-sep_count] != self.join_sep:
                    a = a + (self.join_sep * sep_count) + b
                else:
                    a += b
            else:
                a = b
        return a

            
    def __len__(self):
        return len(self._msgs)


    def __getitem__(self, # type: ignore[override]
                    index: int) -> tuple[MsgKind,str]:
        return Thread._kind_from_pos(index), self._msgs[index]


    def __delitem__(self, 
                    index: int):
        
        if index + 1 == len(self._msgs): # last: just delete
            del self._msgs[index]

        elif index == 0:
            raise IndexError("Can only delete at index 0 if len=1.")
        
        else:
            # augment next of the same kind with previous (of the same kind) text
            self._msgs[index + 1] = self.join_text(self._msgs[index - 1], 
                                                   self._msgs[index + 1])
            del self._msgs[index] # delete requested
            del self._msgs[index - 1] # delete previous


    def __iter__(self):
        class MsgIter:
            def __init__(self, thread):
                self.thread = thread
                self.curr = -1

            def __iter__(self):
                return self
            
            def __next__(self):
                self.curr += 1
                if self.curr < len(self.thread):
                    return self.thread[self.curr]
                else:
                    raise StopIteration

        return MsgIter(self)

    def __reversed__(self):
        return reversed(self._msgs)


    def __add__(self,
                other: Union[Self,list,str,dict,tuple]) -> Self:

        out = self.clone()
        out.concat(other)

        if isinstance(other, Thread):
            self.inst = self.join_text(self.inst, other.inst)

        return out


    def __str__(self):
        inst = self.inst.replace('\n', '\\n')
        sep = self.join_sep.replace('\n', '\\n')
        out = f"inst=█{inst}█, sep='{sep}', len={len(self._msgs)}"
        for index,text in enumerate(self._msgs):
            text = text.replace("\n", "\\n")
            kind = Thread._kind_from_pos(index)
            out += f"\n{index}: {kind.name}=█{text}█"
        return out

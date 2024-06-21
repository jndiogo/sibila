"""Threads of messages are the way to communicate with models. 
For a quick extraction or classification, a thread with a simple IN message and possibly model instructions is enough.
For more sophisticated queries that depend on previous interactions, IN (user or input) and OUT (model output or response) messages can be added to construct a conversation or thread.

- Msg: A message of type IN (user query), OUT (assistant answer) or INST (initial model instructions).
- Thread: A sequence of Msgs alternating between user and model.
"""

from typing import Optional, Union, Callable
from typing_extensions import Self
from enum import Enum, IntFlag
from collections.abc import Sequence, Iterable
from dataclasses import dataclass

import json

import logging
logger = logging.getLogger(__name__)

from .utils import (
    quote_text,
    join_text,
    is_url,
    load_image_as_base64_data_url,
    download_as_base64_data_url
)


@dataclass
class Msg():

    class Kind(Enum):
        """Enumeration for kinds of messages in a Thread."""

        IN = "IN"
        """Input message, from user."""

        OUT = "OUT"
        """Model output message."""

        INST = "INST"
        """Initial model instructions."""
        

        def as_chatml_role(self: Self) -> str:
            CHATML_FROM_KIND: dict = {Msg.Kind.IN: "user", Msg.Kind.OUT: "assistant", Msg.Kind.INST: "system"}
            return CHATML_FROM_KIND.get(self) # type: ignore[return-value]
                
        @staticmethod
        def from_chatml_role(role: str) -> 'Msg.Kind':
            KIND_FROM_CHATML: dict = {"user": Msg.Kind.IN, "assistant": Msg.Kind.OUT, "system": Msg.Kind.INST}
            kind = KIND_FROM_CHATML.get(role)
            if kind is None:
                raise ValueError(f"Unknown ChatML role '{role}'")
            else:
                return kind
    
        @staticmethod
        def flip(kind: 'Msg.Kind') -> 'Msg.Kind':
            return Msg.Kind.OUT if kind is Msg.Kind.IN else Msg.Kind.IN
    
        def __repr__(self):
            return repr(self.value)
        
        

    kind: Kind
    """Message kind."""

    text: str
    """Message text (mandatory)."""

    images: Optional[list[dict]] = None
    """List of images in message. An entry must have a 'url' key, but any other keys can be added.
    Key 'url' key must be a remote url (https,http) or a 'data:' base64-encoded url."""
    


    def __post_init__(self):
        self.set_images(self.images)


    @staticmethod
    def make_IN(text: str,
                images: Optional[Union[list,str,dict]] = None) -> 'Msg':
        return Msg(Msg.Kind.IN,
                   text,
                   images)

    @staticmethod
    def make_OUT(text: str,
                 images: Optional[Union[list,str,dict]] = None) -> 'Msg':
        return Msg(Msg.Kind.OUT, 
                   text, 
                   images)

    @staticmethod
    def make_INST(text: str,
                  images: Optional[Union[list,str,dict]] = None) -> 'Msg':
        return Msg(Msg.Kind.INST, 
                   text,
                   images)


    def clone(self) -> 'Msg':
        return Msg(self.kind, self.text, self.images)


    def as_dict(self) -> dict:
        """Return Msg as a dict."""
        return {"kind": self.kind.value, # kind as string
                "text": self.text,
                "images": self.images}

    @staticmethod
    def from_dict(dic: dict) -> 'Msg':
        return Msg(kind=Msg.Kind(dic["kind"]),
                   text=dic["text"],
                   images=dic["images"])


    
    def as_chatml(self) -> dict:
        """Returns message in a ChatML dict.

        Returns:
            A ChatML dict with "role" and "content" keys.
        """

        role = self.kind.as_chatml_role()

        if self.images:
            chatml_msg = {
                "role": role, 
                "content": [
                    {"type": "text", "text": self.text},
                ]}
            
            for image in self.images:
                if "url" not in image:
                    raise ValueError(f"Image without 'url' key at {image}")
                
                image_url = {"url": image["url"]}
                if "detail" in image:
                    image_url["detail"] = image["detail"]

                chatml_msg["content"].append( # type: ignore[attr-defined]
                    {"type": "image_url", "image_url": image_url}
                )
            return chatml_msg
        else:
            return {"role": role, "content": self.text}


    @staticmethod
    def from_chatml(dic: dict,
                    join_sep:str = "\n") -> 'Msg':
        
        role = dic.get("role")
        if role is None:
            raise ValueError(f"Key 'role' not found in {dic}")

        kind = Msg.Kind.from_chatml_role(role)

        content = dic.get("content")
        if content is None:
            raise ValueError(f"Bad 'content' key in {dic}")

        text = ''
        images = []
        if isinstance(content, list):
            for cont in content:
                if not isinstance(cont, dict) or "type" not in cont:
                    raise TypeError(f"ChatML list entries must be of type dict and include a 'type' key in {cont}")

                if cont["type"] == "text":
                    text = join_text(text, cont["text"], join_sep)

                elif cont["type"] == "image_url":
                    image = cont["image_url"]
                    if "url" not in image:
                        raise TypeError(f"ChatML image_url entries must include a 'url' key in {cont}")
                    images.append(image)
                    
        elif isinstance(content, str):
            text = content
            
        else:
            raise TypeError(f"ChatML content must have str or dict type in {content}")

        return Msg(kind, 
                   text,
                   images if images else None)


    def set_images(self,
                   images: Union[list,str,dict,None]):

        if images is None:
            self.images = None
            return
        
        elif not isinstance(images, list): # ensure input images to be a list
            images = [images]

        self.images = []

        for image in images:

            if isinstance(image, str): # an URL or local path to be loaded into a data: URL
                if not is_url(image):
                    image_url = load_image_as_base64_data_url(image)
                else:
                    image_url = image

                self.images.append({"url": image_url})

            elif isinstance(image, dict):
                if "url" not in image:
                    raise TypeError(f"Image entries must have an 'url' key at {image}")
                
                if not is_url(image["url"]): # a local image path
                    image_url = load_image_as_base64_data_url(image["url"])
                    image["url"] = image_url
                    
                self.images.append(image)

            else:
                raise ValueError("Unable to set images: expecting an str file path or a dict with 'url' key")


    @property
    def has_images(self) -> bool:
        return bool(self.images)


    def download_images_as_data(self):
        """Download any remote images to a 'data:' url."""

        if self.images:
            for image in self.images:
                if not image["url"].startswith("data:"):
                    image["url"] = download_as_base64_data_url(image["url"])


    def join_same_kind(self,
                       other: Self,
                       join_sep: str):
        
        assert self.kind == other.kind, f"Messages are not of the same kind: {self} and {other}."

        self.text = join_text(self.text, other.text,
                              join_sep)
        
        if other.images:
            if self.images is None:
                self.images = []

            self.images += other.images


    def __str__(self):
        text = self.text.replace("\n", "\\n")
        out = f"{self.kind.name}={quote_text(text)}"
        if self.images:
            out += " images=["
            for image in self.images:
                dic = image.copy()
                dic['url'] = dic['url'][:64] + "(...)" if len(dic['url']) > 64 else dic['url']
                out += str(dic) + ","
            out += "]"
                
        return out

    def __repr__(self):
        return self.__str__()
    







class Thread(Sequence):
    """A sequence of messages alternating between IN ("user" role) and OUT ("assistant" role).

    Stores a special initial INST information (known as "system" role in ChatML) providing instructions to the model.
    Some models don't use system instructions - in those cases it's prepended to first IN message.

    Messages are kept in a strict IN,OUT,IN,OUT,... order. 
    To enforce this, if two IN messages are added, the second just appends to the text of the first or to its image list.
    
    Attributes:
        inst: Text for system instructions.
    """

    _msgs: list[Msg]
    """List of IN and OUT Msg messages. Messages are made to alternate between IN and OUT kinds."""

    inst: Msg
    """System instructions in an Msg of kind INST, defaults to empty text."""

    join_sep: str
    """Separator used when message text needs to be joined. Defaults to '\\n'"""


    def __init__(self,
                 t: Optional[Union[Self,list,Msg,dict,tuple,str]] = None,
                 inst: str = "",
                 join_sep: str = "\n"):
        """
        Args:
            t: Optionally initialize from a Thread, list[Msg], list[ChatML format dict], list[tuple], list[str], Msg, ChatML format dict, tuple or str.
            inst: Instructions text. If inst arg is not set and t is a Thread, its inst will be used.
            join_sep: Separator used when message text needs to be joined. Defaults to "\\n".

        Raises:
            TypeError: On invalid args passed.
        """

        self._msgs = []
        self.inst = Msg.make_INST(inst)
        self.join_sep = join_sep

        if t is not None:
            self.concat(t)



    def clone(self) -> Self:
        """Return a copy of current Thread.

        Returns:
            A copy of this Thread.
        """
        return Thread(self)


    def clear(self,
              clear_inst: bool = True):
        """Delete all messages and clear inst."""
        self._msgs = []
        if clear_inst:
            self.inst.text = ""
                


    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = Msg add and concat:
    
    def init_INST_IN(self,
                     inst_text: str,
                     in_text: str,
                     in_images: Optional[Union[list,str,dict]] = None):
        """Initialize Thread with instructions and an IN message.

        Args:
            inst_text: Instructions text.
            in_text: Text for IN message.
            in_images: An array (or its first element) of either an str (a file path, will be loaded and converted to a data: URL) or a dict with "url" key and others. If url arg is not a valid URL, it will be loaded and converted to a data: URL.
        """
        self.clear()
        self.inst.text = inst_text
        self.add_IN(in_text, in_images)


    def add_IN(self,
               in_text: str,
               in_images: Optional[Union[list,str,dict]] = None):
        """Appends an IN message to Thread.

        Args:
            in_text: Text for IN message.
            in_images: An array (or its first element) of either an str (a file path, will be loaded and converted to a data: URL) or a dict with "url" key and others. If url arg is not a valid URL, it will be loaded and converted to a data: URL.
        """
        self.add(Msg.Kind.IN, in_text, in_images)
    

    def add_OUT(self,
                out_text: str,
                out_images: Optional[Union[list,str,dict]] = None):
        """Appends an OUT message to Thread.

        Args:
            out_text: Text for OUT message.
            out_images: An array (or its first element) of either an str (a file path, will be loaded and converted to a data: URL) or a dict with "url" key and others. If url arg is not a valid URL, it will be loaded and converted to a data: URL.
        """
        self.add(Msg.Kind.OUT, out_text, out_images)

    
    def add_OUT_IN(self,
                   out_text: str,
                   in_text: str,
                   *,
                   out_images: Optional[Union[list,str,dict]] = None,
                   in_images: Optional[Union[list,str,dict]] = None):
        """Appends an OUT message followed by an IN message.

        Args:
            out_text: Text for OUT message.
            in_text: Text for IN message.
            out_images: An array (or its first element) of either an str (a file path, will be loaded and converted to a data: URL) or a dict with "url" key and others. If url arg is not a valid URL, it will be loaded and converted to a data: URL.
            in_images: Optional list of IN message images.
        """        
        self.add(Msg.Kind.OUT, out_text, out_images)
        self.add(Msg.Kind.IN, in_text, in_images)



    @staticmethod
    def make_INST_IN(inst_text: str,
                     in_text: str,
                     in_images: Optional[Union[list,str,dict]] = None) -> 'Thread':
        """Return an initialized Thread with instructions and an IN message.

        Args:
            inst_text: Instructions text.
            in_text: Text for IN message.
            in_images: An array (or its first element) of either an str (a file path, will be loaded and converted to a data: URL) or a dict with "url" key and others. If url arg is not a valid URL, it will be loaded and converted to a data: URL.
        """

        thread = Thread(inst=inst_text)
        thread.add_IN(in_text, in_images)
        return thread

    @staticmethod
    def make_IN(in_text: str,
                in_images: Optional[Union[list,str,dict]] = None) -> 'Thread':
        """Return an initialized Thread with an IN message.

        Args:
            in_text: Text for IN message.
            in_images: An array (or its first element) of either an str (a file path, will be loaded and converted to a data: URL) or a dict with "url" key and others. If url arg is not a valid URL, it will be loaded and converted to a data: URL.
        """

        thread = Thread()
        thread.add_IN(in_text, in_images)
        return thread



    def add(self, 
            t: Union[Msg,dict,tuple,str,Msg.Kind],
            text: Optional[str] = None,
            images: Optional[Union[list,str,dict]] = None):
        
        """Add a message to Thread.

        Accepts any of these argument combinations:
            t=Msg, ChatML format dict, tuple or str
            --or--
            t=kind, text[, images]

        Args:
            t: One of Msg, ChatML format dict, tuple or str, or Msg.Kind.
            text: Message text, only if t=Msg.Kind.
            images: only if t=Msg.Kind or t=str-> an array (or its first element) of either an str (a file path, will be loaded and converted to a data: URL) or a dict with keys "url" and any other keys like "detail". If url arg is not a valid URL, it will be loaded and converted to a data URL.
        """

        if text is not None:
            if not isinstance(t, Msg.Kind):
                raise TypeError("When arg 'text' is given, first arg must be of type Msg.Kind")

            msg = Msg(t, text, images)

        else: # add from t arg
            if isinstance(t, dict): # ChatML formatted dict
                msg = Msg.from_chatml(t)


            elif isinstance(t, tuple):
                msg = Msg(self.next_kind,
                          *t)

            elif isinstance(t, str): # simple text
                msg = Msg(self.next_kind,
                          t,
                          images)
                
            elif isinstance(t, Msg):
                msg = t.clone()

            else:
                raise TypeError("Arg 't' must be one of: Msg, ChatML format dict, tuple or str")

           
        # now append to list
        if msg.kind == Msg.Kind.INST:
            self.inst.join_same_kind(msg, self.join_sep)

        else:
            if not len(self._msgs) or msg.kind == self.next_kind: # next different kind or empty
                self._msgs.append(msg)
            else: # new msg is of same kind as last existing message: join/append to it
                last = self._msgs[-1]
                last.join_same_kind(msg, self.join_sep)



    def concat(self,
               t: Union[Self,list,Msg,dict,tuple,str]):
        """Concatenate to current Thread: another Thread, list[Msg], list[ChatML format dict], list[str], Msg, ChatML format dict or str.

        if last message in self is the same kind of first in t, their text, images, etc will be joined.

        Args:
            t: A Thread, list[Msg], list[ChatML format dict], list[str], Msg, ChatML format dict or str.
        """
        if isinstance(t, Thread):
            for msg in t:
                self.add(msg)

            self.inst.join_same_kind(t.inst, self.join_sep)

        else:
            if not isinstance(t, list):
                t = [t]
            for msg in t:
                self.add(msg)



    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = To and from dict and ChatML, load-save

    @staticmethod
    def from_dict(state: dict) -> 'Thread':
        """Deserialize a Thread from a dict."""

        th = Thread()
        for dic in state["_msgs"]:
            th.add(Msg.from_dict(dic))
        th.inst = Msg.from_dict(state["inst"])
        th.join_sep = state["join_sep"]

        return th

    def as_dict(self) -> dict:
        """Serialize this Thread to a dict."""
 
        state = {"_msgs": [],
                 "inst": self.inst.as_dict(),
                 "join_sep": self.join_sep}
    
        for msg in self._msgs:
            state["_msgs"].append(msg.as_dict()) # type: ignore[attr-defined]

        return state


    
    def load(self,
             path: str,
             clear: bool):
        """Load this Thread from a JSON file.

        Args:
            path: Path of file to load.
            clear: Should thread be cleared of messages, including INST? If not will concatenate with existing ones.
        """

        with open(path, 'r', encoding='utf-8') as f:
            js = f.read()
        state = json.loads(js)

        if clear:
            self.clear()

        th = self.from_dict(state)
        self.concat(th)

    
    def save(self,
             path: str):
        """Serialize this Thread to a JSON file.

        Args:
            path: Path of file to save into.
        """
 
        state = self.as_dict()
    
        json_str = json.dumps(state, indent=2, default=vars)
    
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json_str)

    


    def as_chatml(self,
                  include_INST: bool = True) -> list[dict]:
        """Returns Thread as a list of ChatML messages.

        Returns:
            A list of ChatML dict elements with "role" and "content" keys.
        """
        msgs = []

        if self.inst.text and include_INST:
            msgs.append(self.inst.as_chatml())

        for msg in self._msgs:
            msgs.append(msg.as_chatml())
            
        return msgs




    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = Utils

    @property
    def has_images(self) -> bool:
        for msg in self.get_iter(True):
            if msg.has_images:
                return True
        return False


    def download_images_as_data(self):
        self.inst.download_images_as_data()

        for msg in self:
            if msg.has_images:
                msg.download_images_as_data()


    def get_iter(self,
                 include_set_inst: bool):
        """Return an iterator that can be used to cycle over messages.
        include_set_inst: If inst message is set, include it before all others.
        """
        class MsgIter:
            def __init__(self, 
                         thread: Thread,
                         include_inst: bool):
                self.thread = thread
                self.curr = -1 - int(include_inst)

            def __iter__(self):
                return self
            
            def __next__(self):
                self.curr += 1
                if self.curr == -1:
                    return self.thread.inst
                elif self.curr < len(self.thread):
                    return self.thread[self.curr]
                else:
                    raise StopIteration

        return MsgIter(self,
                       include_set_inst and bool(self.inst.text))


    @property
    def next_kind(self) -> Msg.Kind:
        """Get kind of next new message that can be added to thread .

        Returns:
            Kind of last message or Msg.Kind.IN if empty.
        """
        if not self._msgs: # empty
            return Msg.Kind.IN
        else:
            return Msg.Kind.flip(self._msgs[-1].kind)



    def has_text_lower(self,
                       text_lower: str) -> bool:
        """Can the lowercase text be found in one of the messages?

        Args:
            text_lower: The lowercase text to search for in messages.

        Returns:
            True if such text was found.
        """
        for msg in self._msgs:
            if text_lower in msg.text.lower():
                return True
            
        return False        
        


    @staticmethod
    def ensure(query: Union['Thread',Msg,tuple,str],
               inst: Optional[str] = None) -> 'Thread':
        """Utility to return a Thread from either a passed Thread or an str used as an IN message.

        Args:
            query: A Thread or a single IN message given as Msg, list, tuple or str. List and tuple should contain the same args as for creating Msg.
            inst: Instruction message for model. Will override Thread's inst. Defaults to None.

        Raises:
            TypeError: Arg query must be of type Thread or str.

        Returns:
            Initialized Thread object.
        """
        
        if isinstance(query, str):
            th = Thread.make_IN(query)
            if inst is not None:
                th.inst.text = inst

        elif isinstance(query, tuple):
            th = Thread.make_IN(*query)
            if inst is not None:
                th.inst.text = inst

        elif isinstance(query, Msg):
            if query.kind != Msg.Kind.IN:
                raise TypeError("Only 'IN' kind is allowed for Msg")
            th = Thread(query)
            if inst is not None:
                th.inst.text = inst

        elif isinstance(query, Thread):
            if inst is not None:
                th = Thread(query, inst) # a clone
            else:
                th = query

        else:
            raise TypeError("Arg query must be of type Thread, Msg, tuple or str")

        return th




    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = __lower_level__


    def __len__(self):
        return len(self._msgs)


    def __add__(self,
                other: Union[Self,list, Msg, dict, str]) -> Self:
        out = self.clone()
        out.concat(other)
        return out


    def __getitem__(self, # type: ignore[override]
                    index: int) -> Msg:
        
        if isinstance(index, int):
            if index < 0: # need to take care of negative index
                index = len(self._msgs) + index
            if index >= len(self._msgs):
                raise IndexError(f"Trying to access non-existent message at index {index}, allowed index: 0 to {len(self._msgs)-1}")

            return self._msgs[index]
        
        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self._msgs)) # negative index are taken care here
            return self._msgs[slice(start, stop, step)]
        else:
            raise TypeError("Arg index must be int or slice types")


    def __delitem__(self, 
                    index: int):

        if not isinstance(index, int):
            raise TypeError("Arg index must be of int type")

        if index < 0: # need to take care of negative index
            index = len(self._msgs) + index
        if index >= len(self._msgs):
            raise IndexError(f"Trying to access non-existent message at index {index}, allowed index: 0 to {len(self._msgs)-1}")

        if index + 1 == len(self._msgs): # last: just delete
            del self._msgs[index]

        elif index == 0: # delete first message: this will leave an OUT message as first
            del self._msgs[0]
        
        else:
            # augment next of the same kind with previous (of the same kind) text
            # merge text and images
            if self._msgs[index - 1].kind != self._msgs[index + 1].kind:
                raise ValueError(f"Messages at index {index-1} and index {index+1} should have the same kind")
            
            self._msgs[index - 1].join_same_kind(self._msgs[index + 1],  self.join_sep)

            del self._msgs[index + 1] # delete next, which was joined to index-1
            del self._msgs[index] # delete requested


    def __iter__(self):
        # Default iterator doesn't include inst message.
        return self.get_iter(False)


    def __reversed__(self):
        return reversed(self._msgs)




    def __eq__(self, other) -> bool:
        return (self._msgs == other._msgs and 
                self.inst == other.inst and 
                self.join_sep == other.join_sep)


    def __str__(self):
        inst = self.inst.text.replace('\n', '\\n')
        join_sep = self.join_sep.replace('\n', '\\n')
        out = f"Thread inst={quote_text(inst)}, join_sep='{join_sep}', len={len(self._msgs)}"
        for index,msg in enumerate(self._msgs):
            out += f"\n{index}: {msg}"
               
        return out

    def __repr__(self):
        return self.__str__()
    





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


    def trim(self,
             trim_flags: Trim,
             max_token_len: int,
             thread_token_len_fn: Callable
             ) -> int:
        """Trim context by selectively removing older messages until thread fits max_token_len.

        Args:
            trim_flags: Flags to guide selection of which messages to remove.
            max_token_len: Cut messages until size is lower than this number. Defaults to None.
            thread_token_len_fn: A function that returns token count for a passed Thread.

        Example of a thread_token_len_fn that counts 1 char = 1 token:
            def thread_token_len_fn(thread: Thread) -> int:
                total = len(thread.inst.text)
                for msg in thread:
                    total += len(msg.text)
                    if msg.images:
                        total += len(str(msg.images))
                return total

        Returns:
            Trimming result: 1=trimmed messages to max_token_len, 0: no trimming was needed, -1: Unable to trim to max_token_len.
        """

        if trim_flags == Thread.Trim.NONE: # no trimming
            return 0
        
        thread = self.clone()

        any_trim = False
        
        while True:

            curr_len = thread_token_len_fn(thread)

            if curr_len <= max_token_len:
                break

            logger.debug(f"len={curr_len} / max={max_token_len}")

            if thread.inst.text and trim_flags & Thread.Trim.INST:
                thread.inst.text = ""
                any_trim = True
                logger.debug(f"Cutting INST {thread.inst.text[:40]}")
                continue

            # cut first possible message, starting from oldest first ones
            trimmed = False
            in_index = out_index = 0

            for index,msg in enumerate(thread):

                if msg.kind == Msg.Kind.IN:
                    if trim_flags & Thread.Trim.IN:
                        if not (trim_flags & Thread.Trim.KEEP_FIRST_IN and in_index == 0):
                            del thread[index]
                            trimmed = True
                            logger.debug(f"Cutting IN {msg.text[:40]}")
                            break
                    in_index += 1

                elif msg.kind == Msg.Kind.OUT:
                    if trim_flags & Thread.Trim.OUT:                        
                        if not (trim_flags & Thread.Trim.KEEP_FIRST_OUT and out_index == 0):
                            del thread[index]
                            trimmed = True
                            logger.debug(f"Cutting OUT {msg.text[:40]}")
                            break
                    out_index += 1
                
            if not trimmed:
                # all thread messages were cycled but not a single could be cut, so size remains the same
                # arriving here we did all we could for trim_flags but could not remove any more
                return -1
            else:
                any_trim = True

        # while end
        

        if any_trim:
            self._msgs = thread._msgs
            self.inst = thread.inst
            
        return int(any_trim)



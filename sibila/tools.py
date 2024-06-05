"""Tools for model interaction, summarization, etc.
 
- interact(): Interact with model as in a chat, using input().
- loop(): Iteratively append inputs and generate model outputs.
- recursive_summarize(): Recursively summarize a (large) text or text file.
"""


from typing import Any, Optional, Union, Callable

import json
from pprint import pformat
from copy import copy

import logging
logger = logging.getLogger(__name__)


from .gen import (
    GenConf,
    GenRes,
    GenOut
)

from .thread import (
    Thread,
    Msg
)

from .model import (
    Model
)

from .text_splitter import RecursiveTextSplitter




TRIM_DEFAULT = Thread.Trim.IN | Thread.Trim.OUT | Thread.Trim.KEEP_FIRST_IN


def loop(callback: Callable[[Union[GenOut,None], Thread, Model, GenConf, int], bool],
         model: Model,
         *,
         inst_text: Optional[str] = None,
         in_text: Optional[str] = None,

         trim_flags: Thread.Trim = TRIM_DEFAULT,
         max_token_len: Optional[int] = None,
         thread: Optional[Thread] = None,

         genconf: Optional[GenConf] = None,
         ) -> Thread:
    """Iteratively append inputs and generate model outputs.
    
    Callback should call ctx.add_OUT(), ctx.add_IN() and return a bool to continue looping or not.
    
    If last Thread msg is not Msg.Kind.IN, callback() will be called with out_text=None.

    Args:
        callback: A function(out, ctx, model) that will be iteratively called with model's output.
        model: Model to use for generating.
        inst_text: text for Thread instructions. Defaults to None.
        in_text: Text for Thread's initial Msg.Kind.IN. Defaults to None.
        trim_flags: Thread trimming flags, when Thread is too long. Defaults to TRIM_DEFAULT.
        max_token_len: Maximum token count to use when trimming. Defaults to None.
        thread: Optional input Thread. Defaults to None.
        genconf: Model generation configuration. Defaults to None, which uses to model's genconf.
    """
    
    if thread is None:
        thread = Thread()
    else:
        thread = thread

    if inst_text is not None:
        thread.inst.text = inst_text
    if in_text is not None:
        thread.add_IN(in_text)
    
    if genconf is None:
        genconf = model.genconf

    if max_token_len is None:
        resolved_max_tokens = genconf.resolve_max_tokens(model.ctx_len, model.max_tokens_limit)
        max_token_len = model.ctx_len - resolved_max_tokens
        if max_token_len == 0:
            raise ValueError("Unable to calc max_token_len: either pass the value to this function or set GenConf.max_tokens to a non-zero value")

    while True:

        if len(thread) and thread[-1].kind == Msg.Kind.IN:
            # last is an IN message: we can trim and generate
        
            thread.trim(trim_flags,
                    max_token_len,
                    model.token_len_lambda)
                            
            out = model.gen(thread, genconf)
        else:
            out = None # first call
        
        res = callback(out, 
                       thread, 
                       model,
                       genconf,
                       max_token_len)

        if not res:
            break
            

    return thread
            


def interact(model: Model,
             *,
             th: Optional[Thread] = None,
             inst_text: Optional[str] = None,
             trim_flags: Thread.Trim = TRIM_DEFAULT,
             
             genconf: Optional[GenConf] = None,
             max_tokens_default: int = -20
             ) -> Thread:
    """Interact with model as in a chat, using input().

    Includes a list of commands: type !? to see help.

    Args:
        model: Model to use for generating.
        th: Optional input Thread. Defaults to None.
        inst_text: text for Thread instructions. Defaults to None.
        trim_flags: Thread trimming flags, when Thread is too long. Defaults to TRIM_DEFAULT.
        genconf: Model generation configuration. Defaults to None, which uses to model's genconf.
        max_tokens_default: Used if a non-zero genconf.max_tokens is not found.

    Returns:
        Thread after all the interactions.
    """

    def callback(out: Union[GenOut,None], 
                 th: Thread, 
                 model: Model,
                 genconf: GenConf,
                 max_token_len: int) -> bool:

        if out is not None:
            if out.res != GenRes.OK_STOP:
                print(f"***Result={GenRes.as_text(out.res)}***")

            if out.text:
                text = out.text
            else:
                text = "***No text out***"
                
            th.add_OUT(text)
            print(text)
            print()

        
        def print_thread_info():
            length = model.token_len(th, genconf)
            print(f"Thread token len={length}, max len before next gen={max_token_len}")
            

        
        # input loop ===============================================
        MARKER: str = '"""'
        multiline: str = ""

        while True:

            user = input('>').strip()
        
            if multiline:
                if user.endswith(MARKER):
                    user = multiline + "\n" + user[:-3]
                    multiline = ""
                else:
                    multiline += "\n" + user
                    continue
    
            else:
                if not user:
                    return False # terminate loop
                    
                elif user.startswith(MARKER):
                    multiline = user[3:]
                    continue
                    
                elif user.endswith("\\"):
                    user = user[:-1]
                    user = user.replace("\\n", "\n")
                    th.add_IN(user)
                    continue
                    
                elif user.startswith("!"): # a command
                    params = user[1:].split("=")
                    cmd = params[0]
                    params = params[1:]
    
                    if cmd == "inst":
                        th.clear()
                        if params:
                            text = params[0].replace("\\n", "\n")
                            th.inst.text = text
                            
                    elif cmd == "add" or cmd == "a":
                        if params:
                            try:
                                path = params[0]
                                with open(path, "r", encoding="utf-8") as f:
                                    text = f.read()
                                th.add_IN(text)
                                print(text[:500])
                            except FileNotFoundError:
                                print(f"Could not load '{path}'")
                        else:
                            print("Path needed")
                                                
                    elif cmd == 'c':
                        print_thread_info()
                        print(th)
                        
                    elif cmd == 'cl':
                        if not params:
                            params.append("thread.json")
                        try:
                            th.load(params[0], 
                                    clear=True)
                            print(f"Loaded context from {params[0]}")
                        except FileNotFoundError:
                            print(f"Could not load '{params[0]}'")
                        
                    elif cmd == 'cs':
                        if not params:
                            params.append("thread.json")
                        th.save(params[0])
                        print(f"Saved context to {params[0]}")

                    elif cmd == 'image':
                        if not params:
                            print("No image given, using a remote photo of two cats")
                            params.append("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/Singapura_Cats.jpg/320px-Singapura_Cats.jpg")
                        try:
                            th.add_IN("", params[0])
                            print(f"Added image '{params[0]}'.\nPlease enter your question:")
                        except FileNotFoundError:
                            print(f"Could not local image '{params[0]}'")

                    elif cmd == 'tl':
                        print_thread_info()
                        
                    elif cmd == 'info':
                        print(f"Model:\n{model.info()}")
                        print(f"GenConf:\n{genconf}\n")
                        
                        print_thread_info()

    
                    else:
                        print(f"Unknown command '!{cmd}' - known commands:\n"
                              " !inst[=text] - clear messages and add inst (system) message\n"
                              " !add=path - load file and add to last msg\n"
                              " !image=path/url - include a local or remote image. Local images must fit the context!\n"
                              " !c - list context msgs\n"
                              " !cl=path - load context (default=thread.json)\n"
                              " !cs=path - save context (default=thread.json)\n"
                              " !tl - thread's token length\n"
                              " !info - model and genconf info\n"
                              ' Delimit with """ for multiline begin/end or terminate line with \\ to continue into a new line\n'
                              " Empty line + enter to quit"
                              )
                        # " !p - show formatted prompt (if model supports it)\n"
                        # " !to - prompt's tokens\n"
    
                    print()
                    continue
    
            # we have a user prompt
            user = user.replace("\\n", "\n")
            break

        
        th.add_IN(user)
        
        return True # continue looping



    if genconf is None:
        genconf = model.genconf

    if genconf.max_tokens == 0:
        genconf = genconf(max_tokens=max_tokens_default)

    # start prompt loop
    th = loop(callback,
              model,
               
              thread=th,
              inst_text=inst_text,
              in_text=None, # call callback for first prompt
              trim_flags=trim_flags,
              genconf=genconf)

    return th







def recursive_summarize(model: Model,
                        text: Optional[str] = None,
                        path: Optional[str] = None,
                        overlap_size: int = 20,
                        max_token_len: Optional[int] = None,
                        genconf: Optional[GenConf] = None) -> str:
    
    """Recursively summarize a large text or text file, to fit in a Thread context.
     
    Works by:

    1. Breaking text into chunks that fit models context.
    2. Run model to summarize chunks.
    3. Join generated summaries and jump to 1. - do this until text size no longer decreases.

    Args:
        model: Model to use for summarizing.
        text: Initial text.
        path: --Or-- A path to an UTF-8 text file.
        overlap_size: Size in model tokens of the overlapping portions at beginning and end of chunks.

    Returns:
        The summarized text.
    """

    if (text is not None) + (path is not None) != 1:
        raise ValueError("Only one of text or path can be given")

    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

    inst_text = """Your task is to do short summaries of text."""
    in_text = "Summarize the following text:\n"
    th = Thread(inst=inst_text)

    if genconf is None:
        genconf = model.genconf

    if max_token_len is None:
        if model.genconf.max_tokens == 0:
            raise ValueError("Unable to calc max_token_len: make sure genconf.max_tokens is not zero")

        resolved_max_tokens = genconf.resolve_max_tokens(model.ctx_len, model.max_tokens_limit)

        thread = Thread.make_INST_IN(inst_text, in_text)
        token_len = model.token_len(thread)
        max_token_len = model.ctx_len - resolved_max_tokens - (token_len + 16) 
    

    # split initial text
    logger.debug(f"Max token len {max_token_len}")
    
    token_len_fn = model.token_len_lambda
    logger.debug(f"Initial text token_len {token_len_fn(text)}") # type: ignore[arg-type,call-arg]
    
    spl = RecursiveTextSplitter(max_token_len, overlap_size, len_fn=token_len_fn) # type: ignore[arg-type]

    round = 0
    while True: # summarization rounds
        logger.debug(f"Round {round} {'='*60}")
        
        in_list = spl.split(text=text)
        in_len = sum([len(t) for t in in_list])

        logger.debug(f"Split in {len(in_list)} parts, total len {in_len} chars")
        
        out_list = []
        for i,t in enumerate(in_list):
    
            logger.debug(f"{round}>{i} {'='*30}")
            
            th.clear(clear_inst=False)
            th.add_IN(in_text)
            th.add_IN(t)
    
            out = model.gen(th)        
            logger.debug(out)
    
            out_list.append(out.text)

        text = "\n".join(out_list)
        
        out_len = len(text) # sum([len(t) for t in out_list])
        if out_len >= in_len:
            break
        elif len(out_list) == 1:
            break
        else:
            round += 1
            
    return text



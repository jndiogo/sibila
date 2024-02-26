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
    MsgKind
)

from .context import (
    Trim,
    Context
)

from .model import (
    Model
)

from .text_splitter import RecursiveTextSplitter




TRIM_DEFAULT = Trim.IN | Trim.OUT | Trim.KEEP_FIRST_IN


def loop(callback: Callable[[Union[GenOut,None], Context, Model, GenConf], bool],
         model: Model,
         *,
         inst_text: Optional[str] = None,
         in_text: Optional[str] = None,

         trim_flags: Trim = TRIM_DEFAULT,
         ctx: Optional[Context] = None,

         genconf: Optional[GenConf] = None,
         ) -> Context:
    """Iteratively append inputs and generate model outputs.
    
    Callback should call ctx.add_OUT(), ctx.add_IN() and return a bool to continue looping or not.
    
    If last Thread msg is not MsgKind.IN, callback() will be called with out_text=None.

    Args:
        callback: A function(out, ctx, model) that will be iteratively called with model's output.
        model: Model to use for generating.
        inst_text: text for Thread instructions. Defaults to None.
        in_text: Text for Thread's initial MsgKind.IN. Defaults to None.
        trim_flags: Context trimming flags, when Thread is too long. Defaults to TRIM_DEFAULT.
        ctx: Optional input Context. Defaults to None.
        genconf: Model generation configuration. Defaults to None, defaults to model's.
    """
    
    if ctx is None:
        ctx = Context()
    else:
        ctx = ctx

    if inst_text is not None:
        ctx.inst = inst_text
    if in_text is not None:
        ctx.add_IN(in_text)
    
    if genconf is None:
        genconf = model.genconf

    if ctx.max_token_len is not None: # use from ctx
        max_token_len = ctx.max_token_len
    else: # assume max possible for model context and genconf
        max_token_len = model.ctx_len - genconf.max_tokens

    
    while True:

        if len(ctx) and ctx.last_kind == MsgKind.IN:
            # last is an IN message: we can trim and generate
        
            ctx.trim(trim_flags,
                     model,
                     max_token_len=max_token_len
                     )
       
            out = model.gen(ctx, genconf)
        else:
            out = None # first call
        
        res = callback(out, 
                       ctx, 
                       model,
                       genconf)

        if not res:
            break
            

    return ctx
            


def interact(model: Model,
             *,
             ctx: Optional[Context] = None,
             inst_text: Optional[str] = None,
             trim_flags: Trim = TRIM_DEFAULT,
             
             genconf: Optional[GenConf] = None,
             ) -> Context:
    """Interact with model as in a chat, using input().

    Includes a list of commands: type !? to see help.

    Args:
        model: Model to use for generating.
        ctx: Optional input Context. Defaults to None.
        inst_text: text for Thread instructions. Defaults to None.
        trim_flags: Context trimming flags, when Thread is too long. Defaults to TRIM_DEFAULT.
        genconf: Model generation configuration. Defaults to None, defaults to model's.    

    Returns:
        Context after all the interactions.
    """

    def callback(out: Union[GenOut,None], 
                 ctx: Context, 
                 model: Model,
                 genconf: GenConf) -> bool:

        if out is not None:
            if out.res != GenRes.OK_STOP:
                print(f"***Result={GenRes.as_text(out.res)}***")

            if out.text:
                text = out.text
            else:
                text = "***No text out***"
                
            ctx.add_OUT(text)
            print(text)
            print()

        
        def print_thread_info():
            if ctx.max_token_len is not None: # use from ctx
                max_token_len = ctx.max_token_len
            else: # assume max possible for model context and genconf
                max_token_len = model.ctx_len - genconf.max_tokens

            length = model.token_len(ctx, genconf)
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
                    ctx.add_IN(user)
                    continue
                    
                elif user.startswith("!"): # a command
                    params = user[1:].split("=")
                    cmd = params[0]
                    params = params[1:]
    
                    if cmd == "inst":
                        ctx.clear()
                        if params:
                            text = params[0].replace("\\n", "\n")
                            ctx.inst = text
                            
                    elif cmd == "add" or cmd == "a":
                        if params:
                            try:
                                path = params[0]
                                ctx.addx(path=path)
                                ct = ctx.last_text
                                print(ct[:500])
                            except FileNotFoundError:
                                print(f"Could not load '{path}'")
                        else:
                            print("Path needed")
                                                
                    elif cmd == 'c':
                        print_thread_info()
                        print(ctx)
                        
                    elif cmd == 'cl':
                        if not params:
                            params.append("ctx.json")
                        try:
                            ctx.load(params[0])
                            print(f"Loaded context from {params[0]}")
                        except FileNotFoundError:
                            print(f"Could not load '{params[0]}'")
                        
                    elif cmd == 'cs':
                        if not params:
                            params.append("ctx.json")
                        ctx.save(params[0])
                        print(f"Saved context to {params[0]}")
    
                    elif cmd == 'tl':
                        print_thread_info()
                        
                    elif cmd == 'i':
                        print(f"Model:\n{model.info()}")
                        print(f"GenConf:\n{genconf}\n")
                        
                        print_thread_info()

                    # elif cmd == 'p':
                    #     print(model.text_from_turns(ctx.turns))
                        
                    # elif cmd == 'to':
                    #     token_ids = model.tokens_from_turns(ctx.turns)
                    #     print(f"Prompt tokens={token_ids}")
                                                
    
                    else:
                        print(f"Unknown command '!{cmd}' - known commands:\n"
                              " !inst[=text] - clear messages and add inst (system) message\n"
                              " !add|!a=path - load file and add to last msg\n"
                              " !c - list context msgs\n"
                              " !cl=path - load context (default=ctx.json)\n"
                              " !cs=path - save context (default=ctx.json)\n"
                              " !tl - thread's token length\n"
                              " !i - model and genconf info\n"
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

        
        ctx.add_IN(user)
        
        return True # continue looping
            


    # start prompt loop
    ctx = loop(callback,
               model,
               
               ctx=ctx,
               inst_text=inst_text,
               in_text=None, # call callback for first prompt
               trim_flags=trim_flags)

    return ctx







def recursive_summarize(model: Model,
                        text: Optional[str] = None,
                        path: Optional[str] = None,
                        overlap_size: int = 20) -> str:
    """Recursively summarize a (large) text or text file.
     
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
    ctx = Context(pinned_inst_text=inst_text)

    # split initial text
    max_token_len = model.ctx_len - model.genconf.max_tokens - (model.tokenizer.token_len(inst_text + in_text) + 16) 
    logger.debug(f"Max ctx token len {max_token_len}")
    
    token_len_fn = model.tokenizer.token_len_lambda
    logger.debug(f"Initial text token_len {token_len_fn(text)}") # type: ignore[arg-type]
    
    spl = RecursiveTextSplitter(max_token_len, overlap_size, len_fn=token_len_fn)

    round = 0
    while True: # summarization rounds
        logger.debug(f"Round {round} {'='*60}")
        
        in_list = spl.split(text=text)
        in_len = sum([len(t) for t in in_list])

        logger.debug(f"Split in {len(in_list)} parts, total len {in_len} chars")
        
        out_list = []
        for i,t in enumerate(in_list):
    
            logger.debug(f"{round}>{i} {'='*30}")
            
            ctx.clear()
            ctx.add_IN(in_text)
            ctx.add_IN(t)
    
            out = model.gen(ctx)        
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



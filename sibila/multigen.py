"""Functions for comparing output across models.

- thread_multigen(), query_multigen() and multigen(): Compare outputs across models.
- cycle_gen_print(): For a list of models, sequentially grow a Thread with model responses to given IN messages.
"""

from typing import Any, Optional, Union, Callable

import json, csv, re
from io import StringIO

import logging
logger = logging.getLogger(__name__)


from .gen import (
    GenConf,
    GenOut
)

from .thread import (
    Thread,
    Msg
)


from .model import (
    Model
)

from .json_schema import JSchemaConf

from .models import Models



def _default_gencall_text(model: Model,
                          thread: Thread,
                          genconf: Optional[GenConf] = None) -> GenOut:
    out = model.gen(thread, genconf)
    return out



def make_dataclass_gencall(cls: Any, # dataclass definition
                           schemaconf: Optional[JSchemaConf] = None) -> Callable:

    def dataclass_gencall(model: Model,
                          thread: Thread,
                          genconf: Optional[GenConf] = None) -> GenOut:    
        out = model.gen_dataclass(cls, 
                                  thread,
                                  genconf,
                                  schemaconf)
        return out

    return dataclass_gencall


def make_pydantic_gencall(cls: Any, # Pydantic BaseModel class
                          schemaconf: Optional[JSchemaConf] = None) -> Callable:

    def pydantic_gencall(model: Model,
                         thread: Thread,
                         genconf: Optional[GenConf] = None) -> GenOut:    
        out = model.gen_pydantic(cls, 
                                 thread,
                                 genconf,
                                 schemaconf)
        return out

    return pydantic_gencall



def make_extract_gencall(target: Any,
                         schemaconf: Optional[JSchemaConf] = None) -> Callable:

    def extract_gencall(model: Model,
                        thread: Thread,
                        genconf: Optional[GenConf] = None) -> GenOut:    
        out = model.gen_extract(target, 
                                thread,
                                genconf,
                                schemaconf)
        return out

    return extract_gencall







def multigen(threads: list[Thread],
             *,
             models: Optional[list[Model]] = None, # existing models
                
             model_names: Optional[list[str]] = None,
             model_names_del_after: bool = True,
    
             gencall: Optional[Callable] = None,
             genconf: Optional[GenConf] = None
             ) -> list[list[GenOut]]:
    """Generate a list of Threads in multiple models, returning the GenOut for each [thread,model] combination.
    
    Actual generation for each model is implemented by the gencall arg Callable with this signature:
        def gencall(model: Model,
                    thread: Thread,
                    genconf: GenConf) -> GenOut
    
    Args:
        threads: List of threads to input into each model.
        models: A list of initialized models. Defaults to None.
        model_names: --Or-- A list of Models names. Defaults to None.
        model_names_del_after: Delete model_names models after using them: important or an out-of-memory error will eventually happen. Defaults to True.
        gencall: Callable function that does the actual generation. Defaults to None, which will use a text generation default function.
        genconf: Model generation configuration to use in models. Defaults to None, meaning default values.

    Raises:
        ValueError: Only one of models or model_names can be given.

    Returns:
        A list of lists in the format [thread,model] of shape (len(threads), len(models)). For example: out[0] holds threads[0] results on all models, out[1]: threads[1] on all models, ...
    """

    if not ((models is None) ^ ((model_names is None))):
        raise ValueError("Only one of models or model_names can be given")

    if gencall is None:
        gencall = _default_gencall_text
    
    mod_count = len(models) if models is not None else len(model_names) # type: ignore[arg-type]

    all_out = []
    
    for i in range(mod_count):
        if models is not None:
            model = models[i]
            logger.debug(f"Model: {model.desc}")
        else:
            name = model_names[i] # type: ignore[index]
            model = Models.create(name)
            logger.info(f"Model: {name} -> {model.desc}")

        mod_out = []
        for th in threads:
            out = gencall(model, th, genconf)
    
            mod_out.append(out)

        all_out.append(mod_out)

        if model_names_del_after and models is None:
            model.close()
            del model
        
    # all_out is currently shaped (M,T) -> transpose to (T,M), so that each row contains thread t for all models
    tout = []
    for t in range(len(threads)):
        tmout = [] # thread t for all models
        for m in range(mod_count):
            tmout.append(all_out[m][t])

        tout.append(tmout)
        
    return tout




def nice_print(type: str,
               val: Any,
               json_kwargs: dict
               ):
    
    if type == "dic":
        text = json.dumps(val, **json_kwargs)
        
    elif type == "value":
        indent_text = " " * 4
        text = re.sub(r", ([a-z0-9_]+)=", 
                      ",\n" + indent_text + "\\1=", 
                      repr(val),
                      flags=re.IGNORECASE)
        
    else:
        text = str(val)

    return text



def format_text(f: StringIO,
                table: list[list[GenOut]], # [ins, model_outs_for_in]
               
                title_list: list[str],
                model_names: list[str],
                   
                out_keys: list[str] = ["text","dic","value"],
        
                json_kwargs: dict = {"indent": 2,
                                     "sort_keys": False,
                                     "ensure_ascii": False
                                     }):

    def lprint(outs_for_in, model_names):
        
        for index,out in enumerate(outs_for_in): # foreach model out
            print("=" * 20, model_names[index], "->", out.res.name, file=f)

            wrote_anything = False
            first = True
            
            out_dict = out.as_dict()
            for k in out_keys:
                
                if k in out_dict and out_dict[k] is not None:
                    
                    if not first and wrote_anything:
                        print("-" * 20, file=f)
                    else:
                        first = False

                    text = nice_print(k, out_dict[k], json_kwargs)
                    print(text, file=f)

                    if out_dict[k]:
                        wrote_anything = True
                        
            if not wrote_anything:        
                text = nice_print("text", out.text, json_kwargs)
                print(text, file=f)
                    
            
    for il in range(len(title_list)): # for each in
        print("/" * 60, "\n", 
              title_list[il], "\n",
              "/" * 60,
              sep="",
              file=f
              )
        lprint(table[il], model_names)
        print(file=f)




def format_csv(f: StringIO,
               table: list[list[GenOut]], # [ins, model_outs_for_in]
               
               title_list: list[str],
               model_names: list[str],
                   
               out_keys: list[str] = ["text","dic","value"],
        
               json_kwargs: dict = {"indent": 2,
                                    "sort_keys": False,
                                    "ensure_ascii": False
                                    }):
    fieldnames = ["Threads"] + model_names
    
    writer = csv.writer(f, 
                        dialect=csv.excel, 
                        )
    
    writer.writerow(fieldnames)
            
    for il in range(len(title_list)): # for each in

        row = [title_list[il]]

        for index,out in enumerate(table[il]): # foreach model out

            cell = ""
            out_dict = out.as_dict()
            first = True
            for k in out_keys:
                
                if k in out_dict and out_dict[k] is not None:
                    
                    if not first and cell:
                        cell += "\n" + "-" * 20 + "\n"
                    else:
                        first = False

                    text = nice_print(k, out_dict[k], json_kwargs)
                    cell += text
                        
            if not cell:
                cell = nice_print("text", out.text, json_kwargs)

            row.append(cell)
            
        writer.writerow(row)






def thread_multigen(threads: list[Thread],
                    model_names: list[str],
                   
                    text: Union[str,list[str],None] = None,
                    csv: Union[str,list[str],None] = None,
                   
                    gencall: Optional[Callable] = None,                   
                    genconf: Optional[GenConf] = None,
    
                    out_keys: list[str] = ["text","dic", "value"],
                   
                    thread_titles: Optional[list[str]] = None                   
                    ) -> list[list[GenOut]]:
    """Generate a single thread on a list of models, returning/saving results in text/CSV.

    Actual generation for each model is implemented by an optional Callable with this signature:
        def gencall(model: Model,
                    thread: Thread,
                    genconf: GenConf) -> GenOut

    Args:
        threads: List of threads to input into each model.
        model_names: A list of Models names.
        text: An str list with "print"=print results, path=a path to output a text file with results. Defaults to None.
        csv: An str list with "print"=print CSV results, path=a path to output a CSV file with results. Defaults to None.
        gencall: Callable function that does the actual generation. Defaults to None, which will use a text generation default function.
        genconf: Model generation configuration to use in models. Defaults to None, meaning default values.
        out_keys: A list with GenOut members to output. Defaults to ["text","dic", "value"].
        thread_titles: A human-friendly title for each Thread. Defaults to None.

    Returns:
        A list of lists in the format [thread,model] of shape (len(threads), len(models)). For example: out[0] holds threads[0] results on all models, out[1]: threads[1] on all models, ...
    """

    assert isinstance(model_names, list), "model_names must be a list of strings"
    
    table = multigen(threads,
                     model_names=model_names, 
                     gencall=gencall,
                     genconf=genconf)
    
    # table[threads,models]

    if thread_titles is None:
        thread_titles = [str(th) for th in threads]

    def format(format_fn, cmds):
        if cmds is None or not cmds:
            return

        f = StringIO(newline='')

        format_fn(f,
                  table, 
                  title_list=thread_titles,
                  model_names=model_names,
                  out_keys=out_keys)
        fmtd = f.getvalue()
        
        if not isinstance(cmds, list):
            cmds = [cmds]
        for c in cmds:
            if c == 'print':
                print(fmtd)
            else: # path
                with open(c, "w", encoding="utf-8") as f:
                    f.write(fmtd)
                
    format(format_text, text)
    format(format_csv, csv)
        
    return table







def query_multigen(in_list: list[str],
                   inst_text: str,                                                
                   model_names: list[str],
   
                   text: Union[str,list[str],None] = None, # "print", path
                   csv: Union[str,list[str],None] = None, # "print", path
                
                   gencall: Optional[Callable] = None,                   
                   genconf: Optional[GenConf] = None,
    
                   out_keys: list[str] = ["text","dic", "value"],
                   in_titles: Optional[list[str]] = None
                   ) -> list[list[GenOut]]:
    """Generate an INST+IN thread on a list of models, returning/saving results in text/CSV.

    Actual generation for each model is implemented by an optional Callable with this signature:
        def gencall(model: Model,
                    thread: Thread,
                    genconf: GenConf) -> GenOut

    Args:
        in_list: List of IN messages to initialize Threads.
        inst_text: The common INST to use in all models.
        model_names: A list of Models names.
        text: An str list with "print"=print results, path=a path to output a text file with results. Defaults to None.
        csv: An str list with "print"=print CSV results, path=a path to output a CSV file with results. Defaults to None.
        gencall: Callable function that does the actual generation. Defaults to None, which will use a text generation default function.
        genconf: Model generation configuration to use in models. Defaults to None, meaning default values.
        out_keys: A list with GenOut members to output. Defaults to ["text","dic", "value"].
        in_titles: A human-friendly title for each Thread. Defaults to None.

    Returns:
        A list of lists in the format [thread,model] of shape (len(threads), len(models)).        
        For example: out[0] holds threads[0] results on all models, out[1]: threads[1] on all models, ...
    """    

    th_list = []
    for in_text in in_list:
        th = Thread.make_INST_IN(inst_text, in_text)
        th_list.append(th)

    if in_titles is None:
        in_titles = in_list

    out = thread_multigen(th_list,                     
                          model_names=model_names, 
                          text=text,
                          csv=csv,
                          gencall=gencall,
                          genconf=genconf,
                          out_keys=out_keys,
                          thread_titles=in_titles)

    return out







def cycle_gen_print(in_list: list[str],
                    inst_text: str,                                                
                    model_names: list[str],
    
                    gencall: Optional[Callable] = None,                   
                    genconf: Optional[GenConf] = None,
    
                    out_keys: list[str] = ["text","dic", "value"],

                    json_kwargs: dict = {"indent": 2,
                                         "sort_keys": False,
                                         "ensure_ascii": False}
                    ):
    """For a list of models, sequentially grow a Thread with model responses to given IN messages and print the results.

    Works by doing:
    
    1. Generate an INST+IN prompt for a list of models. (Same INST for all).
    2. Append the output of each model to its own Thread.
    3. Append the next IN prompt and generate again. Back to 2.

    Actual generation for each model is implemented by an optional Callable with this signature:
        def gencall(model: Model,
                    thread: Thread,
                    genconf: GenConf) -> GenOut

    Args:
        in_list: List of IN messages to initialize Threads.
        inst_text: The common INST to use in all models.
        model_names: A list of Models names.
        gencall: Callable function that does the actual generation. Defaults to None, which will use a text generation default function.
        genconf: Model generation configuration to use in models. Defaults to None, meaning default values.
        out_keys: A list with GenOut members to output. Defaults to ["text","dic", "value"].
        json_kwargs: JSON dumps() configuration. Defaults to {"indent": 2, "sort_keys": False, "ensure_ascii": False }.
    """

    assert isinstance(model_names, list), "model_names must be a list of strings"

    if gencall is None:
        gencall = _default_gencall_text
            
    
    n_model = len(model_names)
    n_ins = len(in_list)

    for m in range(n_model):
        
        name = model_names[m]
        model = Models.create(name)
        
        print('=' * 80)
        print(f"Model: {name} -> {model.desc}")
        
        th = Thread(inst=inst_text)
        
        for i in range(n_ins):
            in_text = in_list[i]
            print(f"IN: {in_text}")
            
            th += Msg.make_IN(in_text)

            out = gencall(model, th, genconf)

            out_dict = out.as_dict()

            print("OUT")
            
            for k in out_keys:
                
                if k in out_dict and out_dict[k] is not None:
                    
                    if k != out_keys[0]: # not first
                        print("-" * 20)

                    val = nice_print(k, out_dict[k], json_kwargs)
                    print(val)
                    
            th += Msg.make_OUT(out.text)

        model.close()
        del model

    


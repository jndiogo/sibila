"""A bag of assorted utilities."""

from .model import Tokenizer



def dict_merge(dest: dict, 
               src: dict):
    """
    Recursive merge of dictionary entries into dest dict
    """
    for key in src:
        if key in dest:
            if isinstance(dest[key], dict) and isinstance(src[key], dict):
                dict_merge(dest[key], src[key])
            else: # not both dictionaries: overwrite dest's entry with src's
                dest[key] = src[key]
        else: # copy new entry to dest
            dest[key] = src[key]




    


def clear_mem(var_names: list[str]):

    """
    clear_mem(["tokenizer", "model"])
    """
    
    for name in var_names:
        try:
            del globals()[name]
            print("Deleted", name)
        except IndexError:
            ...
            
    try:
        import gc
        gc.collect()
        
        # Empty VRAM cache
        import torch
        torch.cuda.empty_cache()

    except Exception:
        ...




import sys, os
class mute_stdout_stderr():
    '''
    Based on: https://github.com/abetlen/llama-cpp-python/issues/478
    '''
    def __enter__(self):
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr




# =========================================================================== Tokenization


def detok(tokenizer: Tokenizer, 
          text=None, ids=None, add_special=False):
    
    assert (text is not None) ^ (ids is not None), "Only one of text or ids"
    
    if text is not None:
        ids = tokenizer.encode(text, add_special=add_special)
        
    print("Tokens:", ids)
    
    s = tokenizer.decode(ids)

    print("Decode: █" + s + "█")
    
    for o in ids:
        ot = tokenizer.decode([o])
        print(" ", o, "= █" + ot + "█")

    if text is not None and add_special:
        print("Equal:", text == s)





def token_comp(hf_model, llamacpp_model,
               text):
    """
    Compare tokenization between HuggingFace and LlamaCpp
    """
    
    import transformers
    from llama_cpp import (
        Llama,
        LlamaTokenizer,
    )
    
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model)
    
    lc_model = Llama(model_path=llamacpp_model, verbose=False)
    lc_tokenizer = LlamaTokenizer(lc_model)


    hf = hf_tokenizer(text, add_special_tokens=False).input_ids    

    print(f"HF={hf_model} - LlamaCpp={llamacpp_model}")
    lc = lc_tokenizer.encode(text, add_bos=False)
    if hf == lc:
        print("LlamaCpp matches HF")
    else:
        print("LlamaCpp doesn't match HF:")
        for h,l in zip(hf,lc):
            print(h == l, h, l)




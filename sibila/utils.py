"""A bag of general utilities."""

import os

def dict_merge(dest: dict, 
               src: dict):
    """
    Recursive merge of dictionary entries into dest dict
    """
    for key in src:
        if key in dest:
            if isinstance(dest[key], dict) and isinstance(src[key], dict):
                dict_merge(dest[key], src[key])
            else: # not both are dicts: overwrite dest's entry with src's
                dest[key] = src[key]
        else: # copy new entry to dest
            dest[key] = src[key]


def is_subclass_of(cls, base_cls):
    """Safe issubclass what also works for instances"""
    return isinstance(cls,type) and issubclass(cls, base_cls)


def synth_desc(flags: int,
               name: str) -> str:
    """Create a description from a key or variable name.
    For example:
        class_label -> "Class label"
    flags:
        0: just copy
        1: replace _ with space and capitalize()
    """
    if flags & 1:
        name = name.replace("_", " ").capitalize()

    return name



def expand_path(path: str) -> str:
    if '~' in path:
        path = os.path.expanduser(path)

    path = os.path.abspath(path)
    path = os.path.normpath(path) # normalize absolute path
    return path

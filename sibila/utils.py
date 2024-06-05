"""A bag of general utilities."""

from typing import Optional

import os, base64
from urllib.request import urlopen, Request


DEFAULT_USER_AGENT = '''"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.3"'''



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
    if path.startswith("~"):
        path = os.path.expanduser(path)

    path = os.path.abspath(path)
    path = os.path.normpath(path) # normalize absolute path
    return path




def quote_text(text: str) -> str:
    if "'" not in text:
        quote = "'"
    elif '"' not in text:
        quote = '"'
    else:
        quote = "'"
        text = text.replace("'", "\\'")
    return quote + text + quote




def join_text(a: str,
              b: str,
              join_sep: str) -> str:        
    """Join two strings with a separator, avoid adding on initial empty string and separator repetition."""
    if b:
        if a:            
            if not a.endswith(join_sep):
                a += join_sep + b
            else:
                a += b
        else:
            a = b
    return a




def is_url(text: str) -> bool:
    return (text.startswith("https://") or
            text.startswith("http://") or
            text.startswith("ftp://") or
            text.startswith("data:"))


def get_mime_from_extension(ext: str) -> str:
    # Image types only, for now
     
    if ext.startswith("."):
        ext = ext[1:]

    ext = ext.lower()
    if ext == 'png':
        mime_type = "png"
    elif ext == 'webp':
        mime_type = "webp"
    elif ext == 'gif':
        mime_type = "gif"
    elif ext == 'svg':
        mime_type = "svg+xml"
    else: # fallback: jpeg
        mime_type = "jpeg"

    return "image/" + mime_type



def load_image_as_base64_data_url(path: str) -> str:

    with open(path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    _, extension = os.path.splitext(path)

    mime_type = get_mime_from_extension(extension)

    return f"data:{mime_type};base64,{base64_image}"


def download_url(url: str,
                 user_agent: Optional[str] = None) -> dict:
    
    if user_agent is None:
        user_agent = DEFAULT_USER_AGENT
    headers = {"User-Agent": user_agent}

    with urlopen(Request(url, headers=headers)) as f:
        if f.status != 200:
            raise ValueError(f"Error {f.status} downloading '{url}'")
        out_headers = f.headers
        data_bytes = f.read()

    return {
        "bytes": data_bytes,
        "content-type": out_headers.get("Content-Type")
    }



def download_as_base64_data_url(url: str,
                                user_agent: Optional[str] = None) -> str:
    
    download = download_url(url, user_agent)

    base64_image = base64.b64encode(download["bytes"]).decode('utf-8')

    mime_type = download.get("content-type")
    if not mime_type:
        _, extension = os.path.splitext(url)
        mime_type = get_mime_from_extension(extension)

    return f"data:{mime_type};base64,{base64_image}"



def get_bytes_from_url(url: str,
                       user_agent: Optional[str] = None) -> bytes:
    """Supports base64 'data:' and https/http urls."""
    
    if url.startswith("data:"):
        base64_image = url.split(",")[1]
        image_bytes = base64.b64decode(base64_image)
        return image_bytes
    
    else:
        download = download_url(url, user_agent)
        return download["bytes"]
    

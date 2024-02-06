from typing import Optional, Union, Callable

import os, json, re

from pprint import pformat

import logging
logger = logging.getLogger(__name__)



class FormatDir:
    """A singleton to store chat templates for the fine-tuned models used in Sibila.
    
    Detects chat templates from model name/filename or uses from metadata if possible.

    This directory can be setup from a JSON file or by calling add().

    Any new directory entries with the same name replace previous ones on each new call.
    
    Initializes from file base_formatdir.json in this module's directory.

    This env variable is checked during initialization to load from a file:

        SIBILA_FORMAT_CONF: path of a JSON configuration file to add().

    An example of a format directory JSON config file:

    ``` json
    {
        "chatml": {
            # template is a Jinja2 template for this model
            "template": "{% for message in messages %}..."
        },

        "openchat": {
            "match": "openchat.3", # a regexp to match model name or filename
            "template": "{{ bos_token }}..."
        },    

        "phi2": {
            "match": "phi-2",
            "template": "..."
        },

        "phi": "phi2",
        # this is an alias "phi" -> "phi2"
    }
    ```

    Jinja2 templates receive a standard ChatML messages list (created from a Thread) and must deal with the following:

    - In models that don't use a system message, template must take care of prepending it to first user message.
    
    - The add_generation_prompt template variable is always set as True.
    """
    
    dir: Union[dict,None] = None # model directory configuration

    
    BASE_CONF_FILENAME = "base_formatdir.json"
    
    ENV_FORMAT_CONF = "SIBILA_FORMAT_CONF"
    

    @classmethod
    def ensure(cls):
        """Make sure class is initialized.

        Env variables checked:
            SIBILA_FORMAT_CONF: path of a JSON configuration file to add().
        """

        if cls.dir is None:
            cls.dir = {}

            # read base formatdir.json in same folder as this file
            path = os.path.abspath(__file__)
            path = os.path.dirname(path)
            path = os.path.join(path, cls.BASE_CONF_FILENAME)
            if os.path.isfile(path):
                logger.debug(f"Reading base dir conf from '{path}'")
                update_dir_json(cls.dir, path)
                
            path = os.environ.get(cls.ENV_FORMAT_CONF)
            if path is not None:
                path_list = path.split(";")
                for path in path_list:
                    update_dir_json(cls.dir, path)
                    logger.info(f"Loading conf (via env variable) from: '{path}'")
    
            cls._sanity_check()


    
    
    @classmethod
    def add(cls,
            conf_path: Optional[str] = None,
            conf: Optional[dict] = None,
            ):
        """Add a JSON file or configuration dict to the format directory.

        Args:
            conf_path: Path to a JSON file with dirctory configuration. See class __doc__ for format. Defaults to None.
            conf: A dict with configuration as if loaded from JSON by json.loads(). Defaults to None.

        Raises:
            TypeError: Only one of conf_path or conf can be given.
        """

        def expand_path(path: str) -> str:
            if '~' in path:
                path = os.path.expanduser(path)
            return os.path.abspath(path)
        
        if not ((conf_path is not None) ^ (conf is not None)):
            raise TypeError("One of conf_path or conf must be given")

        cls.ensure()
        
        # conf directory loading
        if conf is not None:
            cls.dir.update(conf)
        else:
            update_dir_json(cls.dir, conf_path)

        cls._sanity_check()
        
        
    
    
    @classmethod
    def get(cls,
            name: str) -> Union[dict,None]:
        """Get a format entry by name, following aliases if required.

        Args:
            name: Format name.

        Returns:
            Format dict with chat template.
        """

        cls.ensure()

        na = name.lower()
        while na in cls.dir.keys():
            val = cls.dir[na]
            if isinstance(val, str): # str means link -> follow it
                na = val
            else:
                logger.debug(f"FormatDir get('{name}'): found '{na}' entry")
                return cls._prepare_entry(na, val)

        return None

    
    
    @classmethod
    def search(cls,
               model_id: str) -> Union[dict,None]:
        """Search for model name or filename in the registry.

        Args:
            model_id: Name of filename of model.

        Returns:
            Format dict with chat template or None if none found.
        """

        # Todo: cache compiled re patterns in "_re" entries

        cls.ensure()

        for name,val in cls.dir.items():
            if isinstance(val, str): # a link: ignore when searching
                continue
            if "match" not in val:
                continue
                
            patterns = val["match"]
            if isinstance(patterns, str):
                patterns = [patterns]
                
            for pat in patterns:
                if re.search(pat, model_id, flags=re.IGNORECASE):
                    logger.debug(f"FormatDir search('{model_id}'): found '{name}' entry")
                    return cls._prepare_entry(name, val)
                                
        return None

    
        
    @classmethod
    def info(cls) -> str:
        """Format directory listing."""
        out = ""
        out += f"Directory: {pformat(cls.dir)}"
        return out


    @classmethod
    def clear(cls):
        """Clear the model directory."""
        cls.dir = None



    
    @classmethod
    def _prepare_entry(cls,
                       name: str,
                       val: Union[dict,str]):
        val = val.copy()
        
        if "{{" not in val["template"]: # a link to another template entry
            linked_name = val["template"]
            if linked_name not in cls.dir:
                raise ValueError(f"Broken template link at '{name}': '{linked_name}' does not exist")
            val2 = cls.dir[linked_name]
            val["template"] = val2["template"]
            
        return val



    @classmethod
    def _sanity_check(cls):
        # sanity check complete directory
        for name,val in cls.dir.items():
            if isinstance(val, str): # a link -> does pointed-to exist?
                if val not in cls.dir.keys():
                    raise ValueError(f"Entry '{name}' points to non-existant entry '{val}'")
            else:
                if not isinstance(val, dict):
                    raise ValueError(f"Entry '{name}' must be a dict")
                    
                if "template" not in val:
                    raise ValueError(f"Entry '{name}' must have a 'template' value")
        


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = utils

def update_dir_json(dir: dict,
                    path: str):
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as f:
        new_dir = json.load(f)
        dir.update(new_dir)



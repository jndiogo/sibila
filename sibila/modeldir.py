from typing import Any, Optional, Union, Callable

import os, json

from pprint import pformat

import logging
logger = logging.getLogger(__name__)

from .model import (
    GenConf,
    Model
)

from .utils import dict_merge




class ModelDir:
    """Model directory that unifies (and simplify) model access and configuration.

    Useful to create models from resource names like "llamacpp:openchat" or "openai:gpt-4". 
    This makes it simple to change a model, store model settings, to compare model outputs, etc.
    
    User can add new entries from script or with JSON filenames, via the add() call.
    New directory entries with the same name are merged into existing ones for each added config.

    Uses file base_modeldir.json in this script's directory for the initial defaults, 
    which the user can augment by calling add() with own config files or directly adding model config with add_model().

    These env variables are checked and used during initialization:

        SIBILA_MODEL_DIR_CONF: path of a JSON configuration file to add().

        SIBILA_MODEL_SEARCH_PATH: ';'-delimited list of folders where to find models.


    An example of a model directory JSON config file:

    ``` json
    {
        # "llamacpp" is a provider, you can then create models with names 
        # like "provider:model_name", for ex: "llamacpp:openchat"
        "llamacpp": { 

            "default": {
                # Place here default args for all llamacpp: models. 
                # Each entry below can then override as needed.
            },
            
            "openchat": { # this is model definition
                "name": "openchat-3.5-1210.Q4_K_M.gguf",
                "format": "openchat" # FormatDir's format used by this model
                                     # (formats are chat templates)
            },

            "phi2": {
                "name": "phi-2.Q5_K_M.gguf", # model filename
                "format": "phi2"
            },

            "oc": "openchat" 
            # this is an alias: "oc" forwards to the "openchat" entry
        },

        # The "openai" provider. A model can be created with name: "openai:gpt-4"
        "openai": { 

            "default": {}, # default settings for all openai models
            
            "gpt-3.5": {
                "name": "gpt-3.5-turbo-1106" # OpenAI's  model name
            },

            "gpt-4": {
                "name": "gpt-4-1106-preview"
            },
        },

        # Entry "alias" is not a provider but a way to have simpler alias names.
        # For example you can use "alias:develop" or even simpler, just "develop".
        "alias": { 
            "develop": "llamacpp:openchat",
            "production": "openai:gpt-3.5"
        }
    
    }

    ```

    """
    
    dir: Any = None # model directory configuration => Union[dict[str,Any],None]
    """Model directory dict."""

    search_path: list[str] = [] 
    """Model search path: a list of folders with model locations."""

    genconf: Union[GenConf,None] = None
    """Default GenConfig for created models."""


    DEFAULT_SEARCH_PATH = "."
    
    PROVIDER_CONF = {
        "llamacpp": {
            "mandatory": ["name"],
            "flags": ["name_passthrough"]
        },
        "openai": {
            "mandatory": ["name"],
            "flags": ["name_passthrough"]
        }
    }
    ALL_PROVIDER_NAMES = list(PROVIDER_CONF.keys()) + ["alias"] # providers + "alias"
    
    BASE_CONF_FILENAME = "base_modeldir.json"

    ENV_DIR_CONF = "SIBILA_MODEL_DIR_CONF"
    ENV_SEARCH_PATH = "SIBILA_MODEL_SEARCH_PATH"


    
    @classmethod
    def ensure(cls):
        """Make sure class is initialized.

        Env variables checked:
            SIBILA_MODEL_DIR_CONF: path of a JSON configuration file to add().
            SIBILA_MODEL_SEARCH_PATH: ';'-delimited folder list where to find models.        
        """

        if cls.dir is not None: 
            return
            
        # dir
        cls.dir = {}
        if not cls.search_path:
            cls.add_search_path(cls.DEFAULT_SEARCH_PATH)

        # read base formatdir.json in same folder as this file
        path = os.path.abspath(__file__)
        path = os.path.dirname(path)
        path = os.path.join(path, cls.BASE_CONF_FILENAME)
        if os.path.isfile(path):
            logger.debug(f"Reading base dir conf from '{path}'")
            merge_dir_json(path, cls.dir, cls.search_path, False)

        # check env var
        path = os.environ.get(cls.ENV_DIR_CONF)
        if path is not None:
            path_list = path.split(";")
            for path in path_list:
                merge_dir_json(path, cls.dir, cls.search_path)
                logger.info(f"Loaded conf (via env variable) from: '{path}'")

        # check env search_path: 
        path = os.environ.get(cls.ENV_SEARCH_PATH)
        if path is not None:
            search_path = path.split(";")
            # absolute/home
            cls.add_search_path(search_path)

    
    
    @classmethod
    def add(cls,
            conf_path: Optional[str] = None,
            conf: Optional[dict] = None,
            ):
        """Add a JSON file or configuration dict to the model directory.
        When adding a JSON file, its folder is also added to the search path for model files.
        ~/ can be used in paths for current account's home directory.

        Args:
            conf_path: Path to a JSON file with directory configuration. See class __doc__ for format. Defaults to None.
            conf: A dict with configuration as if loaded from JSON by json.loads(). Defaults to None.

        Raises:
            TypeError: Only one of conf_path or conf can be given.
        """
        
        if not ((conf_path is not None) ^ (conf is not None)):
            raise TypeError("Only one of conf_path or conf can be given")

        cls.ensure()
        
        # conf directory loading
        if conf is not None:
            cls.dir.update(conf)
        else:
            merge_dir_json(conf_path, cls.dir, cls.search_path) # type: ignore[arg-type]

        cls._sanity_check()

    

    
    @classmethod
    def add_model(cls,
                  res_name: str,
                  conf_or_link: Union[dict,str]):
        
        """Add configuration or model alias at given res_name.

        Args:
            res_name: A name in the form "provider:model_name", for example "openai:gtp-4".
            conf_or_link: A configuration dict or an alias name (to an existing model).

        Raises:
            ValueError: If unknown provider.
        """
        
        cls.ensure()
        
        provider,_ = provider_name_from_urn(res_name)
        if provider not in cls.ALL_PROVIDER_NAMES:
            raise ValueError(f"Unknown provider '{provider}' in '{res_name}'")
        
        cls.dir[provider] = conf_or_link

        cls._sanity_check()
        
    

       

    @classmethod
    def add_search_path(cls,
                        path: Union[str,list[str]]):
        """Prepends new paths to model search path.

        During initialization env variable SIBILA_MODEL_SEARCH_PATH is searched for ';'-delimited paths.
        ~/ can be used in paths for current account's home directory.        

        Args:
            path: A path or list of paths to add to model search path.
        """

        cls.ensure()

        prepend_path(cls.search_path, path)

    

     
    @classmethod
    def set_genconf(cls,
                    genconf: GenConf):
        """Set the GenConf to use as default for model creation.

        Args:
            genconf: Model generation configuration.
        """
        cls.genconf = genconf

    

    
    @classmethod
    def create(cls,
               res_name: str,

               # common to all providers
               genconf: Optional[GenConf] = None,
               ctx_len: Optional[int] = None,

               # model-specific overriding:
               **over_args: Union[Any]) -> Model:
        """Create a model after an entry in the model directory.

        Args:
            res_name: Resource name in the format: provider:model_name, for example "llamacpp:openchat".
            genconf: Optional model generation configuration. Overrides set_genconf() value and any directory defaults. Defaults to None.
            ctx_len: Maximum context length to be used. Overrides directory defaults. Defaults to None.
            over_args: Model-specific creation args, which will override default args set in model directory.

        Returns:
            Model: the initialized model.
        """
               
        cls.ensure()        
            
        # resolve "alias:name" res names, or "name": "link_name" links
        provider,name = cls.resolve_urn(res_name)
        # arriving here, prov as a non-link dict entry
        logger.debug(f"Resolved '{res_name}' to '{provider}','{name}'")

        prov = cls.dir[provider]
        
        args = (prov.get("default")).copy() or {}
        prov_conf = cls.PROVIDER_CONF[provider]    

        if name in prov:
            model_args = prov[name]
    
            # default(if any) <- model_args <- over_args
            args = (prov.get("default")).copy() or {}
            args.update(model_args)        
            args.update(over_args)
    
        else:                
            if "name_passthrough" in prov_conf["flags"]:
                model_args = {
                    "name": name                
                }
            else:
                raise ValueError(f"Model '{name}' not found in provider '{provider}'")
            
            args.update(model_args)
            args.update(over_args)

        # override genconf, ctx_len
        if genconf is None:
            genconf = cls.genconf
        if genconf is not None:
            args["genconf"] = genconf

        if ctx_len is not None:
            args["ctx_len"] = ctx_len

        logger.debug(f"Creating model '{provider}:{name}' with resolved args: {args}")

        model: Model
        if provider == "llamacpp":

            # resolve filename -> path
            path = cls._locate_file(args["name"])
            if path is None:
                raise FileNotFoundError(f"File not found in '{res_name}' while looking for file '{args['name']}'. Make sure you initialized ModelDir with a path to this file's folder")

            logger.debug(f"Resolved '{args['name']}' to '{path}'")
            
            del args["name"]
            args["path"] = path
                        
            from .llamacpp import LlamaCppModel

            model = LlamaCppModel(**args)

        
        elif provider == "openai":

            from .openai import OpenAIModel
                    
            model = OpenAIModel(**args)
            
        """            
        elif provider == "hf":
            from .hf import HFModel
            
            model = HFModel(**args)
        """
           
        return model

    


    @classmethod
    def clear(cls):
        """Clear the model directory."""
        cls.dir = None
        cls.search_path = []





    @classmethod
    def resolve_urn(cls,
                    res_name: str) -> tuple[str,str]:
        """
        Checks format and if provider exists, follows string links until a dict or non-existent name key.
        res_name must be in format provider:model_name
        Returns tuple of provider_name, model_name. providre_name must exist, model_name may not
        """

        while True:
            provider, name = provider_name_from_urn(res_name)
                
            if provider not in cls.ALL_PROVIDER_NAMES:
                raise ValueError(f"Don't know how to handle provider '{provider}'. Can only handle the following providers: {cls.ALL_PROVIDER_NAMES}")

            prov = cls.dir[provider]

            if name in prov and isinstance(prov[name], str): # follow string link
                res_name = prov[name]
                if ":" not in res_name: # a local provider link
                    res_name = provider + ":" + res_name
                    
            elif provider == "alias" and name not in prov: # no alias with that name
                raise ValueError(f"Alias not found '{name}'")
                
            else: 
                break
            
        return provider, name
    

    @classmethod
    def _locate_file(cls,
                     path: str) -> Union[str,None]:
        
        if os.path.isabs(path): # absolute?
            if os.path.isfile(path):
                return path
            else:
                return None
        
        for dir in cls.search_path:
            full_path = os.path.join(dir, path)
            if os.path.isfile(full_path):
                return full_path
            
        return None

        
    @classmethod
    def _sanity_check(cls):
        
        for prov in cls.dir.keys():
            if prov == "alias":
                for name,link_name in cls.dir[prov].items():
                    if not isinstance(link_name, str):
                        raise ValueError(f"Alias entries must be strings at alias:{name}")
                        
            else: # real providers
                if prov not in cls.PROVIDER_CONF.keys():
                    raise ValueError(f"Don't know how to handle provider '{prov}'. Can only handle the following providers: {cls.PROVIDER_CONF.keys()}")
    
                # check if mandatory keys are in each model entry
                mandatory_keys = cls.PROVIDER_CONF[prov]["mandatory"]
                prov_models = cls.dir[prov]
                for model_name in prov_models.keys():
                    if model_name == "default" or isinstance(prov_models[model_name], str):
                        continue # skip "default" args entry and string links
                    
                    model_entry = prov_models[model_name]
                    if not all(mand in model_entry for mand in mandatory_keys):
                        raise ValueError(f"ModelDir entry '{prov}:{model_name}' doesn't have all mandatory keys for this provider ({mandatory_keys})") 
                               
                
        # ensure all providers have their own entry
        for p in cls.PROVIDER_CONF.keys():
            if p not in cls.dir.keys():
                cls.dir[p] = {}

        if "alias" not in cls.dir:
            cls.dir["alias"] = {}
            

    
    @classmethod
    def info(cls):
        cls.ensure()
        
        out = ""
        out += f"Directory: {pformat(cls.dir)}\n"
        out += f"Model search path: {cls.search_path}\n"
        out += f"Genconf: {cls.genconf}"
        return out
              


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = utils

def expand_path(path: str) -> str:
    if '~' in path:
        path = os.path.expanduser(path)

    path = os.path.abspath(path)
    path = os.path.normpath(path) # normalize absolute path
    return path


def merge_dir_json(path: str,
                   dir: dict,
                   search_path: list[str],
                   add_to_search_path: bool = True):
    
    path = os.path.abspath(path)
    
    with open(path, "r", encoding="utf-8") as f:
        in_dir = json.load(f)

    # add JSON file folder to search_path
    base_dir = os.path.dirname(path)
    if add_to_search_path:
        prepend_path(search_path, base_dir)

    # parse _config entry
    if "_config" in in_dir:
        config = in_dir["_config"]
        
        path_list = config.get("extend_search_path")
        if path_list is not None:
            
            if not isinstance(path_list, list):
                path_list = [path_list]

            # resolve with respect to JSON file folder
            for i in range(len(path_list)):
                path_list[i] = os.path.join(base_dir, path_list[i])
            
            logger.info(f"Adding {path_list} to search path because of config['extend_search_path'] entry")
            
            prepend_path(search_path, path_list)

            # del in_dir["_config"]["extend_search_path"]
        
        del in_dir["_config"]


    dict_merge(dir, in_dir)


    
def prepend_path(base_list: list[str],
                 path: Union[str,list[str]]):
    
    if isinstance(path, str):
        path_list = [path]
    else:
        path_list = path.copy()

    for p in path_list[::-1]:
        p = expand_path(p)
        if p in base_list:
            base_list.remove(p)            
        base_list.insert(0,p)


def provider_name_from_urn(res_name: str) -> tuple[str,str]:
    if ":" in res_name:
        provider_name = tuple(res_name.split(":"))
        if len(provider_name) > 2:
            raise ValueError("Model resource name must be in the format provider:model_name")    
    else:
        provider_name = "alias", res_name # type: ignore[assignment]
        
    return provider_name # type: ignore[return-value]


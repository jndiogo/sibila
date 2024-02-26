"""Models is a singleton class that centralizes model configuration and creation."""

from typing import Any, Optional, Union, Callable

import os, json, re
from copy import copy

from pprint import pformat

import logging
logger = logging.getLogger(__name__)

from .model import (
    GenConf,
    Model
)

from .utils import (
    dict_merge,
    expand_path
)




class Models:
    """Model and template format directory that unifies (and simplifies) model access and configuration.

    This env variable is checked and used during initialization:
        SIBILA_MODELS: ';'-delimited list of folders where to find: models.json, formats.json and the model files.


    = Model Directory ================================

    Useful to create models from resource names like "llamacpp:openchat" or "openai:gpt-4". 
    This makes it simple to change a model, store model settings, to compare model outputs, etc.
    
    User can add new entries from script or with JSON filenames, via the add() call.
    New directory entries with the same name are merged into existing ones for each added config.

    Uses file base_models.json in this script's directory for the initial defaults, 
    which the user can augment by calling setup() with own config files or directly adding model config with add_model().

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
                "format": "openchat" # chat template format used by this model
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

    = Format Directory ================================

    Detects chat templates from model name/filename or uses from metadata if possible.

    This directory can be setup from a JSON file or by calling add().

    Any new directory entries with the same name replace previous ones on each new call.
    
    Initializes from file base_formats.json in this module's directory.

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


    # ======================================================== Model directory
    model_dir: Any = None # model directory configuration => Union[dict[str,Any],None]
    """Model directory dict."""

    search_path: list[str] = [] 
    """Model search path: list of folders with models."""

    genconf: Union[GenConf,None] = None
    """Default GenConfig for created models."""

    ENV_VAR_NAME = "SIBILA_MODELS"

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
    
    MODELS_CONF_FILENAME = "models.json"
    MODELS_BASE_CONF_FILENAME = "base_" + MODELS_CONF_FILENAME

    

    # ======================================================== Format directory    
    format_dir: Any = None # model directory configuration => Union[dict[str,Any],None]
    """Format directory dict."""

    FORMATS_CONF_FILENAME = "formats.json"
    FORMATS_BASE_CONF_FILENAME = "base_" + FORMATS_CONF_FILENAME











    @classmethod
    def setup(cls,
              path: Optional[Union[str,list[str]]] = None,
              clear: bool = False):
        """Initialize models and formats directory from given model files folder and/or contained configuration files.
        Path can start with "~/" current account's home directory.

        Args:
            path: Path to a folder or to "models.json" or "formats.json" configuration files. Defaults to None which tries to initialize from defaults and env variable.
            clear: Set to clear existing directories before loading from path arg.
        """
        
        if clear:
            cls.clear()

        cls._ensure()

        if path is not None:
            if isinstance(path, str):
                path_list = [path]
            else:
                path_list = path

            cls._read_any(path_list)
            
        cls._sanity_check_models()
        cls._sanity_check_formats()
        
    


    @classmethod
    def clear(cls):
        """Clear directories. Member genconf is not cleared."""
        cls.model_dir = None
        cls.search_path = []
        cls.format_dir = None





    
    @classmethod
    def info(cls,
             verbose: bool = False) -> str:
        """Return information about current setup.

        Args:
            verbose: If False, formats directory values are abbreviated. Defaults to False.

        Returns:
            Textual information about the current setup.
        """
        
        cls._ensure()

        out = ""
        
        out += f"Model search path: {cls.search_path}\n"
        out += f"Models directory:\n{pformat(cls.model_dir, sort_dicts=False)}\n"
        out += f"Model Genconf:\n{cls.genconf}\n"

        if not verbose:
            fordir = {}
            for key in cls.format_dir:
                fordir[key] = copy(cls.format_dir[key])
                if isinstance(fordir[key], dict) and "template" in fordir[key]:
                    fordir[key]["template"] = fordir[key]["template"][:14] + "..."
        else:
            fordir = cls.format_dir

        out += f"Formats directory:\n{pformat(fordir)}"

        return out
    

    




    # ================================================================== Models
    
    @classmethod
    def create(cls,
               res_name: str,

               # common to all providers
               genconf: Optional[GenConf] = None,
               ctx_len: Optional[int] = None,

               # model-specific overriding:
               **over_args: Union[Any]) -> Model:
        """Create a model.

        Args:
            res_name: Resource name in the format: provider:model_name, for example "llamacpp:openchat".
            genconf: Optional model generation configuration. Used instead of set_genconf() value and any directory defaults. Defaults to None.
            ctx_len: Maximum context length to be used. Overrides directory defaults. Defaults to None.
            over_args: Model-specific creation args, which will override default args set in model directory.

        Returns:
            Model: the initialized model.
        """
               
        cls._ensure()        
            
        # resolve "alias:name" res names, or "name": "link_name" links
        provider,name = cls.resolve_model_urn(res_name)
        # arriving here, prov as a non-link dict entry
        logger.debug(f"Resolved model '{res_name}' to '{provider}','{name}'")

        prov = cls.model_dir[provider]
        
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
                raise FileNotFoundError(f"File not found in '{res_name}' while looking for file '{args['name']}'. Make sure you initialized Models with a path to this file's folder")

            logger.debug(f"Resolved llamacpp model '{args['name']}' to '{path}'")
            
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
    def add_model(cls,
                  res_name: str,
                  conf_or_link: Union[dict,str]):
        
        """Add model configuration or name alias for given res_name.

        Args:
            res_name: A name in the form "provider:model_name", for example "openai:gtp-4".
            conf_or_link: A configuration dict or an alias name (to an existing model).

        Raises:
            ValueError: If unknown provider.
        """
        
        cls._ensure()
        
        provider,_ = provider_name_from_urn(res_name)
        if provider not in cls.ALL_PROVIDER_NAMES:
            raise ValueError(f"Unknown provider '{provider}' in '{res_name}'")
        
        cls.model_dir[provider] = conf_or_link

        cls._sanity_check_models()
       



    @classmethod
    def add_search_path(cls,
                        path: Union[str,list[str]]):
        """Prepends new paths to model files search path.

        Args:
            path: A path or list of paths to add to model search path.
        """

        cls._ensure()

        prepend_path(cls.search_path, path)

        logger.debug(f"Adding '{path}' to search_path")

         

    @classmethod
    def set_genconf(cls,
                    genconf: GenConf):
        """Set the GenConf to use as default for model creation.

        Args:
            genconf: Model generation configuration.
        """
        cls.genconf = genconf



    @classmethod
    def resolve_model_urn(cls,
                          res_name: str) -> tuple[str,str]:
        """
        Checks format and if provider exists, follows string links until a dict or non-existent name key.
        res_name must be in format provider:model_name
        Returns tuple of provider_name, model_name. provider_name must exist, model_name may not
        """

        while True:
            provider, name = provider_name_from_urn(res_name)
                
            if provider not in cls.ALL_PROVIDER_NAMES:
                raise ValueError(f"Don't know how to handle provider '{provider}'. Can only handle the following providers: {cls.ALL_PROVIDER_NAMES}")

            prov = cls.model_dir[provider]

            if name in prov and isinstance(prov[name], str): # follow string link
                res_name = prov[name]
                if ":" not in res_name: # a local provider link
                    res_name = provider + ":" + res_name
                    
            elif provider == "alias" and name not in prov: # no alias with that name
                raise ValueError(f"Alias not found for '{name}'. Did you mean 'llamacpp:{name}' or 'openai:{name}'?")
                
            else: 
                break
            
        return provider, name
    

    # =========================================================================== Formats

    @classmethod
    def get_format(cls,
                   name: str) -> Union[dict,None]:
        """Get a format entry by name, following aliases if required.

        Args:
            name: Format name.

        Returns:
            Format dict with chat template.
        """

        cls._ensure()

        na = name.lower()
        while na in cls.format_dir.keys():
            val = cls.format_dir[na]
            if isinstance(val, str): # str means link -> follow it
                na = val
            else:
                logger.debug(f"Format get('{name}'): found '{na}' entry")
                return cls._prepare_format_entry(na, val)

        return None


    
    
    @classmethod
    def search_format(cls,
                      model_id: str) -> Union[dict,None]:
        """Search for model name or filename in the registry.

        Args:
            model_id: Name of filename of model.

        Returns:
            Format dict with chat template or None if none found.
        """

        # Todo: cache compiled re patterns in "_re" entries

        cls._ensure()

        for name,val in cls.format_dir.items():
            if isinstance(val, str): # a link: ignore when searching
                continue
            if "match" not in val:
                continue
                
            patterns = val["match"]
            if isinstance(patterns, str):
                patterns = [patterns]
                
            for pat in patterns:
                if re.search(pat, model_id, flags=re.IGNORECASE):
                    logger.debug(f"Format search for '{model_id}' found '{name}' entry")
                    return cls._prepare_format_entry(name, val)
                                
        return None



    @classmethod
    def is_format_supported(cls,
                            model_id: str) -> bool:
        """Checks if there's template support for a model with this name.

        Args:
            model_id: Model filename or general name.

        Returns:
            True if Models knows the format.
        """
        return cls.search_format(model_id) is not None




    @classmethod
    def _prepare_format_entry(cls,
                              name: str,
                              val: Union[dict,str]):
        val = copy(val)
        
        if "{{" not in val["template"]: # type: ignore[index] # a link to another template entry
            linked_name = val["template"] # type: ignore[index]
            if linked_name not in cls.format_dir:
                raise ValueError(f"Broken template link at '{name}': '{linked_name}' does not exist")
            val2 = cls.format_dir[linked_name]
            val["template"] = val2["template"] # type: ignore[index]
            
        return val




    @classmethod
    def add_format(cls,
                   conf: dict,
                   ):
        """Add a JSON file or configuration dict to the format directory.

        Args:
            conf: A dict with configuration as if loaded from JSON by json.loads(). Defaults to None.
        """

        cls._ensure()
        
        cls.format_dir.update(conf)

        cls._sanity_check_formats()
        

    




    # ================================================================== Lower level

    @classmethod
    def _ensure(cls,
                load_from_env: bool = True):
        """Make sure class is initialized.

        Env variable checked:
            SIBILA_MODELS: ';'-delimited folder list where to find models.        
        """

        if cls.model_dir is not None: 
            return
            
        # model and format dirs
        cls.model_dir = {}
        cls.format_dir = {}

        # always add "." to search path
        cls.add_search_path(cls.DEFAULT_SEARCH_PATH)

        path: Union[str, None]
        # read default base_model.json in same folder as this file
        path = os.path.abspath(__file__)
        path = os.path.dirname(path)
        path = os.path.join(path, cls.MODELS_BASE_CONF_FILENAME)
        if os.path.isfile(path):
            cls._read_models(path)

        # read base_formats.json in same folder
        path = os.path.abspath(__file__)
        path = os.path.dirname(path)
        path = os.path.join(path, cls.FORMATS_BASE_CONF_FILENAME)
        if os.path.isfile(path):
            cls._read_formats(path)
            

        # check env var
        if load_from_env:
            path = os.environ.get(cls.ENV_VAR_NAME)
            if path is not None:
                path_list = path.split(";")
                cls._read_any(path_list)




    @classmethod
    def _read_models(cls,
                     models_path: str):
 
        logger.info(f"Loading models conf from: '{models_path}'")
        merge_dir_json(cls.model_dir,
                       models_path)

        dir_path = os.path.dirname(models_path)
        cls.add_search_path(dir_path)


    @classmethod
    def _read_formats(cls,
                      formats_path: str):
        
        logger.info(f"Loading formats conf from '{formats_path}'")
        update_dir_json(cls.format_dir, formats_path)
        


    @classmethod
    def _read_folder(cls,
                     path: str):

        if not os.path.isdir(path):
            raise OSError(f"Directory not found: '{path}'")
        
        cls.add_search_path(path)

        # read models
        models_path = os.path.join(path, cls.MODELS_CONF_FILENAME)
        if os.path.isfile(models_path):
            cls._read_models(models_path)

        # read formats
        formats_path = os.path.join(path, cls.FORMATS_CONF_FILENAME)
        if os.path.isfile(formats_path):
            cls._read_formats(formats_path)
           

    @classmethod
    def _read_any(cls,
                  path: Union[str,list[str]]):
        
        if isinstance(path, str):
            path_list = [path]
        else:
            path_list = path
        
        for path in path_list:
            path = expand_path(path)

            if os.path.isdir(path):
                cls._read_folder(path)

            elif path.find(cls.MODELS_CONF_FILENAME) >= 0:
                cls._read_models(path)

            elif path.find(cls.FORMATS_CONF_FILENAME) >= 0:
                cls._read_formats(path)



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
    def _sanity_check_models(cls):
        
        for prov in cls.model_dir.keys():
            if prov == "alias":
                for name,link_name in cls.model_dir[prov].items():
                    if not isinstance(link_name, str):
                        raise ValueError(f"Alias entries must be strings at alias:{name}")
                        
            else: # real providers
                if prov not in cls.PROVIDER_CONF.keys():
                    raise ValueError(f"Don't know how to handle provider '{prov}'. Can only handle the following providers: {cls.PROVIDER_CONF.keys()}")
    
                # check if mandatory keys are in each model entry
                mandatory_keys = cls.PROVIDER_CONF[prov]["mandatory"]
                prov_models = cls.model_dir[prov]
                for model_name in prov_models.keys():
                    if model_name == "default" or isinstance(prov_models[model_name], str):
                        continue # skip "default" args entry and string links
                    
                    model_entry = prov_models[model_name]
                    if not all(mand in model_entry for mand in mandatory_keys):
                        raise ValueError(f"Models entry '{prov}:{model_name}' doesn't have all mandatory keys for this provider ({mandatory_keys})") 
                               
                
        # ensure all providers have their own entry
        for p in cls.PROVIDER_CONF.keys():
            if p not in cls.model_dir.keys():
                cls.model_dir[p] = {}

        if "alias" not in cls.model_dir:
            cls.model_dir["alias"] = {}
            


    @classmethod
    def _sanity_check_formats(cls):

        # sanity check complete directory
        for name,val in cls.format_dir.items():
            if isinstance(val, str): # a link -> does pointed-to exist?
                if val not in cls.format_dir.keys():
                    raise ValueError(f"Entry '{name}' points to non-existant entry '{val}'")
            else:
                if not isinstance(val, dict):
                    raise ValueError(f"Entry '{name}' must be a dict")
                    
                if "template" not in val:
                    raise ValueError(f"Entry '{name}' must have a 'template' value")
                      





# ======================================================================== Utils


def prepend_path(base_list: list[str],
                 path: Union[str,list[str]]):
    
    if isinstance(path, str):
        path_list = [path]
    else:
        path_list = path.copy()

    for p in path_list[::-1]:
        p = expand_path(p)
        while p in base_list:
            base_list.remove(p)            
        base_list.insert(0,p)




def merge_dir_json(dir: dict,
                   path: str):
    
    path = os.path.abspath(path)
    
    with open(path, "r", encoding="utf-8") as f:
        in_dir = json.load(f)

    dict_merge(dir, in_dir)



    



def provider_name_from_urn(res_name: str) -> tuple[str,str]:
    if ":" in res_name:
        provider_name = tuple(res_name.split(":"))
        if len(provider_name) > 2:
            raise ValueError("Model resource name must be in the format provider:model_name")    
    else:
        provider_name = "alias", res_name # type: ignore[assignment]
        
    return provider_name # type: ignore[return-value]





def update_dir_json(dir: dict,
                    path: str):
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as f:
        new_dir = json.load(f)
        dir.update(new_dir)

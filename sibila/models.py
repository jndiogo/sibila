"""Models is a singleton class that centralizes model configuration and creation."""

from typing import Any, Optional, Union, Callable

import os, json, re
from copy import copy, deepcopy

from importlib.resources import files, as_file

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

import sibila.res


class Models:
    """Model and template format directory that unifies (and simplifies) model access and configuration.

    This env variable is checked and used during initialization:
        SIBILA_MODELS: ';'-delimited list of folders where to find: models.json, formats.json and model files.


    = Models Directory =

    Useful to create models from resource names like "llamacpp:openchat" or "openai:gpt-4". 
    This makes it simple to change a model, store model settings, to compare model outputs, etc.
    
    User can add new entries from script or with JSON filenames, via the add() call.
    New directory entries with the same name are merged into existing ones for each added config.

    Uses file "sibila/res/base_models.json" for the initial defaults, which the user can augment 
    by calling setup() with own config files or directly adding model config with set_model().

    An example of a model directory JSON config file:

    ``` json
    {
        # "llamacpp" is a provider, you can then create models with names 
        # like "provider:model_name", for ex: "llamacpp:openchat"
        "llamacpp": { 

            "_default": { # place here default args for all llamacpp: models.
                "genconf": {"temperature": 0.0}
                # each model entry below can then override as needed
            },
            
            "openchat": { # a model definition
                "name": "openchat-3.5-1210.Q4_K_M.gguf",
                "format": "openchat" # chat template format used by this model
            },

            "phi2": {
                "name": "phi-2.Q4_K_M.gguf", # model filename
                "format": "phi2",
                "genconf": {"temperature": 2.0} # a hot-headed model
            },

            "oc": "openchat" 
            # this is a link: "oc" forwards to the "openchat" entry
        },

        # The "openai" provider. A model can be created with name: "openai:gpt-4"
        "openai": { 

            "_default": {}, # default settings for all OpenAI models
            
            "gpt-3.5": {
                "name": "gpt-3.5-turbo-1106" # OpenAI's model name
            },

            "gpt-4": {
                "name": "gpt-4-1106-preview"
            },
        },

        # "alias" entry is not a provider but a way to have simpler alias names.
        # For example you can use "alias:develop" or even simpler, just "develop" to create the model:
        "alias": { 
            "develop": "llamacpp:openchat",
            "production": "openai:gpt-3.5"
        }
    }
    ```

    Rules for entry inheritance/overriding
    
    Entries in the '_default' key of each provider will serve as defaults for models of that provider.
    Model entries in base_models_dir (automatically loaded from 'res/base_models.json') are overridden 
    by any entries of the same name loaded from a local 'models.json' file with Models.setup(). Here,
    overridden means local keys of the same name replace base keys (as a dict.update()).
    However '_default' entries only apply separately to either base_models_dir or 'local models.json', 
    as in a lexical scope.


    = Format Directory =

    Detects chat templates from model name/filename or uses from metadata if possible.

    This directory can be setup from a JSON file or by calling set_format().

    Any new entries with the same name replace previous ones on each new call.
    
    Initializes from file "sibila/res/base_formats.json".


    Example of a "formats.json" file:

    ``` json
    {
        "chatml": {
            # template is a Jinja template for this model
            "template": "{% for message in messages %}..."
        },

        "openchat": {
            "match": "openchat", # a regexp to match model name or filename
            "template": "{{ bos_token }}..."
        },    

        "phi": {
            "match": "phi",
            "template": "..."
        },

        "phi2": "phi",
        # this is a link: "phi2" -> "phi"
    }
    ```

    Jinja2 templates receive a standard ChatML messages list (created from a Thread) and must deal with the following:

    - In models that don't use a system message, template must take care of prepending it to first user message.
    
    - The add_generation_prompt template variable is always set as True.

    """


    # ======================================================== Model directory
    models_dir: Any = None # model directory configuration => Union[dict[str,Any],None]
    """Local models directory dict: loaded in setup()."""

    base_models_dir: Any = None
    """Base models directory dict: loaded at init from res/base_models.json."""

    models_search_path: list[str] = [] 
    """Model search path: list of folders with models."""

    genconf: Union[GenConf,None] = None
    """Default GenConf for created models."""

    ENV_VAR_NAME = "SIBILA_MODELS"

    PROVIDER_CONF = {
        "anthropic": {
            "mandatory": [],
            "flags": ["name_passthrough"]
        },
        "fireworks": {
            "mandatory": [],
            "flags": ["name_passthrough"]
        },
        "groq": {
            "mandatory": [],
            "flags": ["name_passthrough"]
        },
        "llamacpp": {
            "mandatory": ["name"],
            "flags": ["name_passthrough", "local"]
        },
        "mistral": {
            "mandatory": [],
            "flags": ["name_passthrough"]
        },
        "openai": {
            "mandatory": [],
            "flags": ["name_passthrough"]
        },
        "together": {
            "mandatory": [],
            "flags": ["name_passthrough"]
        },
    }
    ALL_PROVIDER_NAMES = list(PROVIDER_CONF.keys()) + ["alias"] # providers + "alias"

    @classmethod
    def EMPTY_MODELS_DIR(_) -> dict:
        return {
            "anthropic": {},
            "fireworks": {},
            "groq": {},
            "llamacpp": {},
            "mistral": {},
            "openai": {},
            "together": {},
            "alias": {},
        }

    MODELS_CONF_FILENAME = "models.json"
    MODELS_BASE_CONF_FILENAME = "base_" + MODELS_CONF_FILENAME

    DEFAULT_ENTRY_NAME = "_default"


    # ======================================================== Format directory    
    formats_dir: Any = None # loaded models directory configuration => Union[dict[str,Any],None]
    """Loaded/local formats directory dict: loaded in setup()."""

    base_formats_dir: Any = None # base models directory configuration => Union[dict[str,Any],None]
    """Base formats directory dict: loaded at init from res/base_formats.json."""

    @classmethod
    def EMPTY_FORMATS_DIR(_) -> dict:
        return {}

    FORMATS_CONF_FILENAME = "formats.json"
    FORMATS_BASE_CONF_FILENAME = "base_" + FORMATS_CONF_FILENAME











    @classmethod
    def setup(cls,
              path: Optional[Union[str,list[str]]] = None,
              clear: bool = False,
              add_cwd: bool = True,
              load_from_env: bool = True):
        """Initialize models and formats directory from given model files folder and/or contained configuration files.
        Path can start with "~/" current account's home directory.

        Args:
            path: Path to a folder or to "models.json" or "formats.json" configuration files. Defaults to None which tries to initialize from defaults and env variable.
            clear: Set to clear existing directories before loading from path arg.
            add_cwd: Add current working directory to search path.
            load_from_env: Load from SIBILA_MODELS env variable?
        """
        
        if clear:
            cls.clear()

        cls._ensure(add_cwd, 
                    load_from_env)

        if path is not None:
            if isinstance(path, str):
                path_list = [path]
            else:
                path_list = path

            cls._read_any(path_list)
        
    


    @classmethod
    def clear(cls):
        """Clear directories. Members base_models_dir and base_formats_dir and genconf are not cleared."""
        cls.models_dir = None
        cls.models_search_path = []
        cls.formats_dir = None





    
    @classmethod
    def info(cls,
             include_base: bool = True,
             verbose: bool = False) -> str:
        """Return information about current setup.

        Args:
            verbose: If False, formats directory values are abbreviated. Defaults to False.

        Returns:
            Textual information about the current setup.
        """
        
        cls._ensure()

        out = ""
                
        out += f"Models search path: {cls.models_search_path}\n"

        models_dir = cls.fused_models_dir() if include_base else cls.models_dir
        out += f"Models directory:\n{pformat(models_dir, sort_dicts=False)}\n"

        out += f"Model Genconf:\n{cls.genconf}\n"

        formats_dir = cls.fused_formats_dir() if include_base else cls.formats_dir

        if not verbose:
            fordir = {}
            for key in formats_dir:
                fordir[key] = deepcopy(formats_dir[key])
                if isinstance(fordir[key], dict) and "template" in fordir[key]:
                    fordir[key]["template"] = fordir[key]["template"][:14] + "..."
        else:
            fordir = formats_dir

        out += f"Formats directory:\n{pformat(fordir)}"

        return out
    

    




    # ================================================================== Models
    
    @classmethod
    def create(cls,
               res_name: str,

               # common to all providers
               genconf: Optional[GenConf] = None,
               ctx_len: Optional[int] = None,

               *,
               # debug/testing
               resolved_create_args: Optional[dict] = None,

               # model-specific overriding:
               **over_args: Union[Any]) -> Model:
        """Create a model.

        Args:
            res_name: Resource name in the format: provider:model_name, for example "llamacpp:openchat".
            genconf: Optional model generation configuration. Overrides set_genconf() value and any directory defaults. Defaults to None.
            ctx_len: Maximum context length to be used. Overrides directory defaults. Defaults to None.
            resolved_create_args: Pass an empty dict to be filled by this method with the resolved args used in model creation. Defaults to None.
            over_args: Model-specific creation args, which will override default args set in model directory.

        Returns:
            Model: the initialized model.
        """
               
        try:
            provider, _, args = cls.resolve_model_entry(res_name, **over_args)
        except ValueError as e:
            raise NameError(str({e}))
        
        # override genconf, ctx_len
        if genconf is None:
            genconf = cls.genconf

        if genconf is not None:
            args["genconf"] = genconf

        elif "genconf" in args and isinstance(args["genconf"], dict):
            # transform dict into a GenConf instance:
            args["genconf"] = GenConf.from_dict(args["genconf"])

        if ctx_len is not None:
            args["ctx_len"] = ctx_len

        if resolved_create_args is not None:
            resolved_create_args.update(args)


        logger.debug(f"Resolved '{res_name}' to provider '{provider}' with args: {args}")


        model: Model
        if provider == "anthropic":

            from .anthropic import AnthropicModel
            model = AnthropicModel(**args)
            
        elif provider == "fireworks":

            from .schema_format_openai import FireworksModel
            model = FireworksModel(**args)
            
        elif provider == "groq":

            from .schema_format_openai import GroqModel
            model = GroqModel(**args)
            
        elif provider == "llamacpp":
            from .llamacpp import LlamaCppModel, extract_sub_paths

            # resolve filename -> path. Path filenames can be in the form model1*model2
            sub_paths = args["name"].split('*')

            # only resolve first path. If not found, let LlamaCpp raise the error below
            sub_paths[0] = cls._locate_file(sub_paths[0]) or sub_paths[0]
            
            # rejoin located paths with '*' (if multiple)
            path = '*'.join(sub_paths)
            logger.debug(f"Resolved llamacpp model '{args['name']}' to '{path}'")

            # rename "name" -> "path" which LlamaCppModel is expecting
            del args["name"]
            args["path"] = path
                        
            model = LlamaCppModel(**args)
        
        elif provider == "mistral":

            from .mistral import MistralModel
            model = MistralModel(**args)
            
        elif provider == "openai":

            from .openai import OpenAIModel
            model = OpenAIModel(**args)
            
        elif provider == "together":

            from .schema_format_openai import TogetherModel
            model = TogetherModel(**args)

        else:
            raise ValueError(f"Unknown provider '{provider}' for '{res_name}'")
        
    
           
        return model




    @classmethod
    def add_models_search_path(cls,
                               path: Union[str,list[str]]):
        """Prepends new paths to model files search path.

        Args:
            path: A path or list of paths to add to model search path.
        """

        cls._ensure()

        prepend_path(cls.models_search_path, path)

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
    def list_models(cls,
                    name_query: str,
                    providers: list[str],
                    include_base: bool,
                    resolved_values: bool) -> dict:
        """List format entries matching query.

        Args:
            name_query: Case-insensitive substring to match model names. Empty string for all.
            providers: Filter by these exact provider names. Empty list for all.
            include_base: Also list fused values from base_models_dir.
            resolved_values: Return resolved entries or raw ones.

        Returns:
            A dict where keys are model res_names and values are respective entries.
        """

        cls._ensure()

        models_dir = cls.fused_models_dir() if include_base else cls.models_dir

        out = {}

        name_query = name_query.lower()

        for prov_name in models_dir:

            if providers and prov_name not in providers:
                continue

            prov_dic = models_dir[prov_name]

            for name in prov_dic:

                if name == cls.DEFAULT_ENTRY_NAME:
                    continue

                if name_query and name_query not in name.lower():
                    continue

                entry_res_name = prov_name + ":" + name

                if resolved_values:
                    # okay to use get_model_entry() because it resolves to fused
                    res = cls.get_model_entry(entry_res_name) # type: ignore[assignment]
                    if res is None:
                        continue
                    else:
                        val = res[1]
                else:
                    val = prov_dic[name]

                out[entry_res_name] = val

        return out





    @classmethod
    def get_model_entry(cls,
                        res_name: str) -> Union[tuple[str,dict],None]:
        """Get a resolved model entry. Resolved means following any links.

        Args:
            res_name: Resource name in the format: provider:model_name, for example "llamacpp:openchat".

        Returns:
            Resolved entry (res_name,dict) or None if not found.
        """
               
        cls._ensure()        
            
        models_dir = cls.fused_models_dir()

        # resolve "alias:name" res names, or "name": "link_name" links
        provider,name = resolve_model(models_dir, res_name, cls.ALL_PROVIDER_NAMES)
        # arriving here, prov as a non-link dict entry
        logger.debug(f"Resolved model '{res_name}' to '{provider}','{name}'")

        prov = models_dir[provider]
        
        if name in prov:
            return provider + ":" + name, prov[name]
        else:
            return None

    @classmethod
    def has_model_entry(cls,
                        res_name: str) -> bool:
        return cls.get_model_entry(res_name) is not None



    @classmethod
    def resolve_model_entry(cls,
                            res_name: str,

                            # model-specific overriding:
                            **over_args: Union[Any]) -> tuple[str,str,dict]:
        """Resolve a name (provider:name) into the known dict for that model.

        Args:
            res_name: Resource name in the format: provider:model_name, for example "llamacpp:openchat".
            genconf: Optional model generation configuration. Overrides set_genconf() value and any directory defaults. Defaults to None.
            ctx_len: Maximum context length to be used. Overrides directory defaults. Defaults to None.
            over_args: Model-specific creation args, which will override default args set in model directory.

        Returns:
            Tuple of resolved provider, name, dict of creation/config values.
        """
               
        cls._ensure()        
            
        models_dir = cls.fused_models_dir()

        # resolve "alias:name" res names, or "name": "link_name" links
        provider,name = resolve_model(models_dir, res_name, cls.ALL_PROVIDER_NAMES)
        # arriving here, prov as a non-link dict entry

        prov = models_dir[provider]
        
        if name in prov:

            # which _default to use? local or base?
            if name in cls.models_dir[provider]: # in local: get fused _default
                default_args = prov.get(cls.DEFAULT_ENTRY_NAME, {})
            else: # not in local: get base_models _default, without values from fused
                base_prov = cls.base_models_dir[provider]
                default_args = base_prov.get(cls.DEFAULT_ENTRY_NAME, {})

            args = deepcopy(default_args)
            if "name" not in args:
                args["name"] = name
    
            # _default <- model_args <- over_args
            args.update(prov[name])        
            args.update(over_args)
    
        else:
            prov_conf: dict = cls.PROVIDER_CONF[provider] # type: ignore[assignment]
            if "name_passthrough" in prov_conf["flags"]:
                model_args = {
                    "name": name                
                }
            else:
                raise ValueError(f"Model '{name}' not found in provider '{provider}'")

            # fused _default <- model_args(only name) <- over_args
            args = deepcopy(prov.get(cls.DEFAULT_ENTRY_NAME, {}))
            args.update(model_args)        
            args.update(over_args)

        return provider,name, args
    






    @classmethod
    def set_model(cls,
                  res_name: str,
                  model_name: str,
                  format_name: Optional[str] = None,
                  genconf: Optional[GenConf] = None):
        """Add model configuration for given res_name.

        Args:
            res_name: A name in the form "provider:model_name", for example "openai:gtp-4".
            model_name: Model name or filename identifier.
            format_name: Format name used by model. Defaults to None.
            genconf: Base GenConf to use when creating model. Defaults to None.

        Raises:
            ValueError: If unknown provider.
        """

        cls._ensure()
        
        provider,name = provider_name_from_urn(res_name, False)
        if provider not in cls.ALL_PROVIDER_NAMES:
            raise ValueError(f"Unknown provider '{provider}' in '{res_name}'")
        
        entry: dict = {
            "name": model_name
        }

        if format_name:
            if not cls.has_format_entry(format_name):
                raise ValueError(f"Could not find format '{format_name}'")
            entry["format"] = format_name

        if genconf:
            entry["genconf"] = genconf.as_dict()

        cls.models_dir[provider][name] = entry

       


    @classmethod
    def update_model(cls,
                     res_name: str,
                     model_name: Optional[str] = None,
                     format_name: Optional[str] = None,
                     genconf: Union[GenConf,str,None] = None):
        
        """update model fields

        Args:
            res_name: A name in the form "provider:model_name", for example "openai:gtp-4".
            model_name: Model name or filename identifier. Defaults to None.
            format_name: Format name used by model. Use "" to delete. Defaults to None.
            genconf: Base GenConf to use when creating model. Defaults to None.

        Raises:
            ValueError: If unknown provider.
        """

        cls._ensure()
        
        provider,name = provider_name_from_urn(res_name, False)
        if provider not in cls.ALL_PROVIDER_NAMES:
            raise ValueError(f"Unknown provider '{provider}' in '{res_name}'")
        
        entry = cls.models_dir[provider][name]

        if model_name:
            entry["name"] = model_name

        if format_name is not None:
            if format_name != "":
                if not cls.has_format_entry(format_name):
                    raise ValueError(f"Could not find format '{format_name}'")
                entry["format"] = format_name
            else:
                del entry["format"]

        if genconf is not None:
            if genconf != "":
                entry["genconf"] = genconf
            else:
                del entry["genconf"]




    @classmethod
    def set_model_link(cls,
                       res_name: str,
                       link_name: str):
        """Create a model link into another model.

        Args:
            res_name: A name in the form "provider:model_name", for example "openai:gtp-4".
            link_name: Name of model this entry links to.

        Raises:
            ValueError: If unknown provider.
        """
        
        cls._ensure()
        
        provider,name = provider_name_from_urn(res_name, True)
        if provider not in cls.ALL_PROVIDER_NAMES:
            raise ValueError(f"Unknown provider '{provider}' in '{res_name}'")
        
        # first: ensure link_name is a res_name
        if ':' not in link_name:
            link_name = provider + ":" + link_name

        if not cls.has_model_entry(link_name):
            raise ValueError(f"Could not find linked model '{link_name}'")

        # second: check link name is without provider if same
        link_split = link_name.split(":")
        if len(link_split) == 2:
            if link_split[0] == provider: # remove same "provider:"
                link_name = link_split[1]

        cls.models_dir[provider][name] = link_name




    @classmethod
    def delete_model(cls,
                     res_name: str):
        """Delete a model entry.

        Args:
            res_name: Model entry in the form "provider:name".
        """

        cls._ensure()

        provider, name = provider_name_from_urn(res_name,
                                                allow_alias_provider=False)

        if provider not in cls.ALL_PROVIDER_NAMES:
            raise ValueError(f"Unknown provider '{provider}', must be one of: {cls.ALL_PROVIDER_NAMES}")

        prov = cls.models_dir[provider]        
        if name not in prov:
            raise ValueError(f"Model '{res_name}' not found")

        # verify if any entry links to name:
        def check_link_to(link_to_name: str, 
                          provider: str) -> Union[str, None]:
            
            for name,entry in cls.models_dir[provider].items():
                if isinstance(entry,str) and entry == link_to_name:
                    return name
            return None
        
        offender = check_link_to(name, provider)
        if offender is not None:
            raise ValueError(f"Cannot delete '{res_name}', as entry '{provider}:{offender}' links to it")

        offender = check_link_to(name, "alias")
        if offender is not None:
            raise ValueError(f"Cannot delete '{res_name}', as entry 'alias:{offender}' links to it")

        del prov[name]




   

    @classmethod
    def save_models(cls,
                    path: Optional[str] = None,
                    include_base: bool = False):

        cls._ensure()

        if path is None:
            if len(cls.models_search_path) != 1:
                raise ValueError("No path arg provided and multiple path in cls.search_path. Don't know where to save.")
            
            path = os.path.join(cls.models_search_path[0], "models.json")

        with open(path, "w", encoding="utf-8") as f:
            models_dir = cls.fused_models_dir() if include_base else cls.models_dir
            
            # clear providers with no models:
            for provider in cls.ALL_PROVIDER_NAMES:
                if provider in models_dir and not models_dir[provider]:
                    del models_dir[provider]
            
            json.dump(models_dir, f, indent=4)

        return path




    @classmethod
    def resolve_provider_defaults(cls,
                                  provider: str,
                                  key_list: list,
                                  origin: int) -> dict:
        """Resolve _default values for provider. Model classes like LlamaCppModel use this to get defaults for ctx_len, for example

        Args:
            provider: Provider name.
            key_list: List of keys for which we want default values.
            origin: Which _default to use - 0=base_models_dir, 1=models_dir, 2=fused

        Returns:
            Dictionary of values for each key in key_list, or None if such key is not found.
        """
               
        cls._ensure()        

        if origin == 0:
            models_dir = cls.base_models_dir
        elif origin == 1:
            models_dir = cls.models_dir
        else:
            models_dir = cls.fused_models_dir()

        prov = models_dir[provider]
        defaults = prov.get(cls.DEFAULT_ENTRY_NAME, {})

        out = {}
        for key in key_list:
            out[key] = defaults.get(key)

        return out
    

    @classmethod
    def fused_models_dir(cls) -> dict:
        dir = deepcopy(cls.base_models_dir)
        dict_merge(dir, cls.models_dir)
        return dir







    # =========================================================================== Formats

    @classmethod
    def list_formats(cls,
                     name_query: str,
                     include_base: bool,
                     resolved_values: bool) -> dict:
        """List format entries matching query.

        Args:
            name_query: Case-insensitive substring to match format names. Empty string for all.
            include_base: Also list base_formats_dir.
            resolved_values: Return resolved entries or raw ones.

        Returns:
            A dict where keys are format names and values are respective entries.
        """

        cls._ensure()

        out = {}

        name_query = name_query.lower()

        formats_dir = cls.fused_formats_dir() if include_base else cls.formats_dir

        for name in formats_dir.keys():

            if name_query and name_query not in name.lower():
                continue

            val = formats_dir[name]

            if resolved_values:
                res = cls.get_format_entry(name)
                if res is None:
                    continue
                else:
                    val = res[1]

            out[name] = val

        return out




    @classmethod
    def get_format_entry(cls,
                         name: str) -> Union[tuple[str,dict],None]:
        """Get a resolved format entry by name, following links if required.

        Args:
            name: Format name.

        Returns:
            Tuple of (resolved_name, format_entry).
        """

        cls._ensure()

        return get_format_entry(cls.fused_formats_dir(), name)

    @classmethod
    def has_format_entry(cls,
                         name: str) -> bool:
        return cls.get_format_entry(name) is not None




    @classmethod
    def get_format_template(cls,
                            name: str) -> Union[str,None]:
        """Get a format template by name, following links if required.

        Args:
            name: Format name.

        Returns:
            Resolved format template str.
        """

        res = cls.get_format_entry(name)
        return None if res is None else res[1]["template"]




    @classmethod
    def match_format_entry(cls,
                           name: str) -> Union[tuple[str,dict],None]:
        """Search the formats registry, based on model name or filename.

        Args:
            name: Name or filename of model.

        Returns:
            Tuple (name, format_entry) where name is a resolved name. Or None if none found.
        """

        cls._ensure()

        return search_format(cls.fused_formats_dir(), name)


    @classmethod
    def match_format_template(cls,
                              name: str) -> Union[str,None]:
        """Search the formats registry, based on model name or filename.

        Args:
            name: Name or filename of model.

        Returns:
            Format template or None if none found.
        """

        res = cls.match_format_entry(name)

        return None if res is None else res[1]["template"]




    @classmethod
    def folder_match_format_template(cls,
                                     model_path: str) -> Union[str,None]:
        """Locally search for format in a models.json file located in the same folder as the model.
        Doesn't add read entries to class models directory.

        Args:
            model_path: Model path.

        Returns:
            Format template.
        """

        cls._ensure()

        models_dir:dict = {}
        formats_dir:dict = {}

        folder_path = os.path.dirname(model_path)

        models_path = os.path.join(folder_path, cls.MODELS_CONF_FILENAME)
        if os.path.isfile(models_path):
            logger.debug(f"Loading *local* models conf from '{models_path}'")
            try:
                merge_dir_json(models_dir, models_path)
                
            except Exception:
                logger.warning(f"Could not load 'models.json' at '{models_path}', while looking for model format. "
                               "Please verify JSON syntax is correct.")
                models_dir = {}

        formats_path = os.path.join(folder_path, cls.FORMATS_CONF_FILENAME)
        if os.path.isfile(formats_path):
            logger.debug(f"Loading *local* formats conf from '{formats_path}'")
            try:
                update_dir_json(formats_dir, formats_path)

            except Exception:
                logger.warning(f"Could not load 'formats.json' at '{formats_path}', while looking for model format. "
                               "Please verify JSON syntax is correct.")
                formats_dir = {}
                
        if not models_dir and not formats_dir: # nothing to do
            return None


        # fuse: folder_dirs <- models_folder_dirs <- base_dirs
        dir = cls.fused_models_dir()
        dict_merge(dir, models_dir)
        models_dir = dir
        dir = cls.fused_formats_dir()
        dir.update(formats_dir)
        formats_dir = dir


        filename = os.path.basename(model_path)

        # 1: check models_dir
        for provider, conf_entry in cls.PROVIDER_CONF.items():

            if "local" not in conf_entry["flags"]: # type: ignore[index] # filter local providers
                continue

            if provider in models_dir:
                prov = models_dir[provider]

                for mod_name,entry in prov.items():

                    if "name" in entry and entry["name"] == filename:
                        logger.debug(f"Located model '{mod_name}': {entry}")

                        if "format" in entry:
                            format: str = entry["format"] # type: ignore[assignment]
                            if "{{" in format: # Jinja template
                                return format
                            
                            else: # format name available in formats_dir?
                                template = get_format_entry(formats_dir, format)
                                if template is not None:
                                    logger.debug(f"Found format '{format}'")
                                    return template[1]["template"]

        # 2: check in formats_dir
        res = search_format(formats_dir, filename)
        if res is not None:
            logger.debug(f"Found format '{res[0]}'")
            return res[1]["template"]
        
        return None










    @classmethod
    def set_format(cls,
                   name: str,
                   template: str,
                   match: Optional[str] = None):
        """Add a format entry to the format directory.

        Args:
            name: Format entry name.
            template: The Chat template format in Jinja2 format
            match: Regex that matches names/filenames that use this format. Default is None.
        """

        cls._ensure()

        if "{{" not in template: # a link_name for the template
            if not cls.has_format_entry(template):
                raise ValueError(f"Could not find linked template entry '{template}'.")

        entry = {
            "template": template
        }
        if match is not None:
            entry["match"] = match

        cls.formats_dir[name] = entry        
        



    @classmethod
    def set_format_link(cls,
                        name: str,
                        link_name: str):
        """Add a format link entry to the format directory.

        Args:
            name: Format entry name.
            link_name: Name of format that this entry links to.
        """

        cls._ensure()

        if not cls.has_format_entry(link_name):
            raise ValueError(f"Could not find linked entry '{link_name}'.")

        cls.formats_dir[name] = link_name
        



    @classmethod
    def delete_format(cls,
                      name: str):
        """Delete a format entry.

        Args:
            name: Format entry name.
        """

        cls._ensure()

        if name not in cls.formats_dir:
            raise ValueError(f"Format name '{name}' not found.")
        
        for check_name,entry in cls.formats_dir.items():
            if isinstance(entry,str) and entry == name:
                raise ValueError(f"Cannot delete '{name}', as entry '{check_name}' links to it")

        del cls.formats_dir[name]






    @classmethod
    def save_formats(cls,
                     path: Optional[str] = None,
                     include_base: bool = False):
        
        cls._ensure()

        if path is None:
            if len(cls.models_search_path) != 1:
                raise ValueError("No path arg provided and multiple path in cls.search_path. Don't know where to save.")
            
            path = os.path.join(cls.models_search_path[0], "formats.json")

        with open(path, "w", encoding="utf-8") as f:
            formats_dir = cls.fused_formats_dir() if include_base else cls.formats_dir
            json.dump(formats_dir, f, indent=4)

        return path


    @classmethod
    def fused_formats_dir(cls) -> dict:
        dir = deepcopy(cls.base_formats_dir)
        dir.update(cls.formats_dir)
        return dir








    # ================================================================== Lower level

    @classmethod
    def _ensure(cls,
                add_cwd: bool = True,
                load_from_env: bool = True):
        """Make sure class is initialized.

        Env variable checked:
            SIBILA_MODELS: ';'-delimited folder list where to find models.        
        """

        if cls.models_dir is not None: 
            return
            
        # model and format dirs
        cls.models_dir = cls.EMPTY_MODELS_DIR()
        cls.formats_dir = cls.EMPTY_FORMATS_DIR()

        if add_cwd:
            # add CWD to search path, so that relative paths work
            try: # try but don't add a deleted dir
                cwd_path = os.getcwd()
                if os.path.isdir(cwd_path):
                    cls.add_models_search_path(cwd_path)
            except FileNotFoundError:
                ...

        path: Union[str, None]

        cls.base_models_dir = cls.EMPTY_MODELS_DIR()
        cls.base_formats_dir = cls.EMPTY_FORMATS_DIR()

        # read base_models.json from res folder
        source = files(sibila.res).joinpath(cls.MODELS_BASE_CONF_FILENAME)
        with as_file(source) as src_path:
            path = str(src_path)
            if os.path.isfile(path):
                read_models(cls.base_models_dir, path)
                sanity_check_models(cls.base_models_dir,
                                    cls.PROVIDER_CONF,
                                    cls.DEFAULT_ENTRY_NAME)

        # read base_formats.json from res folder
        source = files(sibila.res).joinpath(cls.FORMATS_BASE_CONF_FILENAME)
        with as_file(source) as src_path:
            path = str(src_path)
            if os.path.isfile(path):
                read_formats(cls.base_formats_dir, path)
                sanity_check_formats(cls.base_formats_dir)

 
        # check env var
        if load_from_env:
            path = os.environ.get(cls.ENV_VAR_NAME)
            if path is not None:
                path_list = path.split(";")
                cls._read_any(path_list)






    @classmethod
    def _read_any(cls,
                  path: Union[str,list[str]]):
        
        if isinstance(path, str):
            path_list = [path]
        else:
            path_list = path

        models_loaded = False
        formats_loaded = False

        for path in path_list:
            path = expand_path(path)

            if os.path.isdir(path):

                cls.add_models_search_path(path)

                # read models
                models_path = os.path.join(path, cls.MODELS_CONF_FILENAME)
                if os.path.isfile(models_path):
                    read_models(cls.models_dir, models_path)
                    models_loaded = True

                # read formats
                formats_path = os.path.join(path, cls.FORMATS_CONF_FILENAME)
                if os.path.isfile(formats_path):
                    read_formats(cls.formats_dir, formats_path)
                    formats_loaded = True

            else:
                dir_path = os.path.dirname(path)
                cls.add_models_search_path(dir_path)

                if path.endswith(cls.MODELS_CONF_FILENAME):
                    read_models(cls.models_dir, path)
                    models_loaded = True

                elif path.endswith(cls.FORMATS_CONF_FILENAME):
                    read_formats(cls.formats_dir, path)
                    formats_loaded = True


        if models_loaded:
            sanity_check_models(cls.fused_models_dir(),
                                cls.PROVIDER_CONF,
                                cls.DEFAULT_ENTRY_NAME)
                    
        if formats_loaded:
            sanity_check_formats(cls.fused_formats_dir())




    @classmethod
    def _locate_file(cls,
                     path: str) -> Union[str,None]:
        
        if os.path.isabs(path): # absolute?
            if os.path.isfile(path):
                return path
            else:
                return None
        
        for dir in cls.models_search_path:
            full_path = os.path.join(dir, path)
            if os.path.isfile(full_path):
                return full_path
            
        return None

        






# ======================================================================== Utils


# ====================== models utils
            
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



def read_models(models_dir: dict,
                models_path: str):

    logger.debug(f"Loading models conf from: '{models_path}'")
    merge_dir_json(models_dir, models_path)




def merge_dir_json(dir: dict,
                   path: str):
    
    path = os.path.abspath(path)
    
    with open(path, "r", encoding="utf-8") as f:
        in_dir = json.load(f)

    dict_merge(dir, in_dir)




def provider_name_from_urn(res_name: str,
                           allow_alias_provider: bool) -> tuple[str,str]:
    if ":" in res_name:
        provider_name = tuple(res_name.split(":"))
        if len(provider_name) > 2:
            raise ValueError(f"Model resource name must be in the format provider:model_name (for '{res_name}')")
    else:
        if allow_alias_provider:
            provider_name = "alias", res_name # type: ignore[assignment]
        else:
            raise ValueError(f"Alias not allowed (for '{res_name}')")
        
    return provider_name # type: ignore[return-value]



def resolve_model(models_dir: dict,
                  res_name: str,
                  valid_providers: list) -> tuple[str,str]:
    """
    Checks if provider exists, follows string links until a dict or non-existent name key.
    Arg res_name must be in format provider:model_name
    Returns tuple of provider_name, model_name. provider_name must exist, model_name may not
    """

    while True:
        provider, name = provider_name_from_urn(res_name, True)
            
        if provider not in valid_providers:
            raise ValueError(f"Don't know how to handle provider '{provider}'. Can only handle the following providers: {valid_providers}")

        prov = models_dir[provider]

        if name in prov and isinstance(prov[name], str): # follow string link
            res_name = prov[name]
            if ":" not in res_name: # a local provider link
                res_name = provider + ":" + res_name
                
        elif provider == "alias" and name not in prov: # no alias with that name
            raise ValueError(f"Alias not found for '{name}'. Did you mean 'llamacpp:{name}' or 'openai:{name}'?")
            
        else: 
            break
        
    return provider, name



def sanity_check_models(models_dir: dict,
                        provider_conf: dict,
                        default_entry_name: str):
    
    for prov in models_dir.keys():
        if prov == "alias":
            for name,link_name in models_dir[prov].items():
                if not isinstance(link_name, str):
                    raise ValueError(f"Alias entries must be strings at alias:{name}")
                    
        else: # real providers
            if prov not in provider_conf.keys():
                raise ValueError(f"Don't know how to handle provider '{prov}'. Can only handle the following providers: {provider_conf.keys()}")

            # check if mandatory keys are in each model entry
            mandatory_keys = provider_conf[prov]["mandatory"]
            prov_models = models_dir[prov]
            for model_name in prov_models.keys():
                if model_name == default_entry_name or isinstance(prov_models[model_name], str):
                    continue # skip "_default" args entry and string links
                
                model_entry = prov_models[model_name]
                if not all(mand in model_entry for mand in mandatory_keys):
                    raise ValueError(f"Models entry '{prov}:{model_name}' doesn't have all mandatory keys for this provider ({mandatory_keys})") 
                            
            
    # ensure all providers have their own entry
    for p in provider_conf.keys():
        if p not in models_dir.keys():
            models_dir[p] = {}

    if "alias" not in models_dir:
        models_dir["alias"] = {}
        







# ======================== formats utils


def read_formats(formats_dir: dict,
                 formats_path: str):
    logger.debug(f"Loading formats conf from '{formats_path}'")
    update_dir_json(formats_dir, formats_path)



def update_dir_json(dir: dict,
                    path: str):
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as f:
        new_dir = json.load(f)
        dir.update(new_dir)



def resolve_format_entry(formats_dir: dict,
                         name: str,
                         val: Union[dict,str]):
    val = deepcopy(val)
    
    if "{{" not in val["template"]: # type: ignore[index] # a link to another template entry
        linked_name = val["template"] # type: ignore[index]
        if linked_name not in formats_dir:
            raise ValueError(f"Broken template link at '{name}': '{linked_name}' does not exist")
        val2 = formats_dir[linked_name]
        val["template"] = val2["template"] # type: ignore[index]
        
    return val


def get_format_entry(formats_dir: dict,
                     name: str) -> Union[tuple[str,dict],None]:
    """Get a resolved format entry by name, following links if required.

    Args:
        name: Format name.

    Returns:
        Tuple of (resolved_name, format_entry).
    """

    na = name.lower()
    while na in formats_dir.keys():
        val = formats_dir[na]
        if isinstance(val, str): # str means link -> follow it
            na = val
        else:
            logger.debug(f"Format get('{name}'): found '{na}' entry")
            return na, resolve_format_entry(formats_dir,
                                            na,
                                            val)
    return None



def search_format(formats_dir: dict,
                  model_id: str) -> Union[tuple[str,dict],None]:

    # TODO: cache compiled re patterns in "_re" entries

    for name,val in formats_dir.items():
        if isinstance(val, str): # a link: ignore when searching
            continue
        if "match" not in val:
            continue
            
        patterns = val["match"]
        if isinstance(patterns, str):
            patterns = [patterns]
            
        for pat in patterns:
            if re.search(pat, model_id, flags=re.IGNORECASE):
                logger.info(f"Format search for '{model_id}' found '{name}' entry")
                return name, resolve_format_entry(formats_dir,
                                                  name,
                                                  val)
    return None


def sanity_check_formats(formats_dir: dict):

    # sanity check complete directory
    for name,val in formats_dir.items():
        if isinstance(val, str): # a link -> does pointed-to exist?
            if val not in formats_dir.keys():
                raise ValueError(f"Entry '{name}' points to non-existent entry '{val}'")
        else:
            if not isinstance(val, dict):
                raise ValueError(f"Entry '{name}' must be a dict")
                
            if "template" not in val:
                raise ValueError(f"Entry '{name}' must have a 'template' value")
                    






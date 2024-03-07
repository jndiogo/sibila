"""Models is a singleton class that centralizes model configuration and creation."""

from typing import Any, Optional, Union, Callable

import os, json, re
from copy import copy

from importlib_resources import files, as_file

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


    = Model Directory =

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
    """Model directory dict."""

    models_search_path: list[str] = [] 
    """Model search path: list of folders with models."""

    genconf: Union[GenConf,None] = None
    """Default GenConf for created models."""

    ENV_VAR_NAME = "SIBILA_MODELS"

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

    EMPTY_MODELS_DIR: dict = {
        "llamacpp": {},
        "openai": {},
        "alias": {}
    }

    MODELS_CONF_FILENAME = "models.json"
    MODELS_BASE_CONF_FILENAME = "base_" + MODELS_CONF_FILENAME

    DEFAULT_ENTRY_NAME = "_default"


    # ======================================================== Format directory    
    formats_dir: Any = None # model directory configuration => Union[dict[str,Any],None]
    """Format directory dict."""

    EMPTY_FORMATS_DIR: dict = {}

    FORMATS_CONF_FILENAME = "formats.json"
    FORMATS_BASE_CONF_FILENAME = "base_" + FORMATS_CONF_FILENAME











    @classmethod
    def setup(cls,
              path: Optional[Union[str,list[str]]] = None,
              clear: bool = False,
              add_cwd: bool = True,
              load_base: bool = True,
              load_from_env: bool = True):
        """Initialize models and formats directory from given model files folder and/or contained configuration files.
        Path can start with "~/" current account's home directory.

        Args:
            path: Path to a folder or to "models.json" or "formats.json" configuration files. Defaults to None which tries to initialize from defaults and env variable.
            clear: Set to clear existing directories before loading from path arg.
            add_cwd: Add current working directory to search path.
            load_base: Whether to load "base_models.json" and "base_formats.json" from "sibila/res" folder.
            load_from_env: Load from SIBILA_MODELS env variable?
        """
        
        if clear:
            cls.clear()

        cls._ensure(add_cwd, 
                    load_base,
                    load_from_env)

        if path is not None:
            if isinstance(path, str):
                path_list = [path]
            else:
                path_list = path

            cls._read_any(path_list)
        
    


    @classmethod
    def clear(cls):
        """Clear directories. Member genconf is not cleared."""
        cls.models_dir = None
        cls.models_search_path = []
        cls.formats_dir = None





    
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
        
        out += f"Models search path: {cls.models_search_path}\n"
        out += f"Models directory:\n{pformat(cls.models_dir, sort_dicts=False)}\n"
        out += f"Model Genconf:\n{cls.genconf}\n"

        if not verbose:
            fordir = {}
            for key in cls.formats_dir:
                fordir[key] = copy(cls.formats_dir[key])
                if isinstance(fordir[key], dict) and "template" in fordir[key]:
                    fordir[key]["template"] = fordir[key]["template"][:14] + "..."
        else:
            fordir = cls.formats_dir

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
            genconf: Optional model generation configuration. Overrides set_genconf() value and any directory defaults. Defaults to None.
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

        prov = cls.models_dir[provider]
        
        if name in prov:
            model_args = prov[name]
    
            # _default(if any) <- model_args <- over_args
            args = (prov.get(cls.DEFAULT_ENTRY_NAME)).copy() or {}
            args.update(model_args)        
            args.update(over_args)
    
        else:                
            prov_conf = cls.PROVIDER_CONF[provider]    

            if "name_passthrough" in prov_conf["flags"]:
                model_args = {
                    "name": name                
                }
            else:
                raise ValueError(f"Model '{name}' not found in provider '{provider}'")
            
            args = {}
            args.update(model_args)
            args.update(over_args)


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

        logger.debug(f"Creating model '{provider}:{name}' with resolved args: {args}")


        model: Model
        if provider == "llamacpp":

            # resolve filename -> path
            path = cls._locate_file(args["name"])
            if path is None:
                raise FileNotFoundError(f"File not found in '{res_name}' while looking for file '{args['name']}'. Make sure you called Models.setup() with a path to the file's folder")

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
                    resolved_values: bool) -> dict:
        """List format entries matching query.

        Args:
            name_query: Case-insensitive substring to match model names. Empty string for all.
            providers: Filter by these exact provider names. Empty list for all.
            resolved_values: Return resolved entries or raw ones.

        Returns:
            A dict where keys are model res_names and values are respective entries.
        """

        cls._ensure()

        out = {}

        name_query = name_query.lower()

        for prov_name in cls.models_dir:

            if providers and prov_name not in providers:
                continue

            prov_dic = cls.models_dir[prov_name]

            for name in prov_dic:

                if name == cls.DEFAULT_ENTRY_NAME:
                    continue

                if name_query and name_query not in name.lower():
                    continue

                entry_res_name = prov_name + ":" + name

                if resolved_values:
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
            
        # resolve "alias:name" res names, or "name": "link_name" links
        provider,name = cls.resolve_model_urn(res_name)
        # arriving here, prov as a non-link dict entry
        logger.debug(f"Resolved model '{res_name}' to '{provider}','{name}'")

        prov = cls.models_dir[provider]
        
        if name in prov:
            return provider + ":" + name, prov[name]
        else:
            return None

    @classmethod
    def has_model_entry(cls,
                        res_name: str) -> bool:
        return cls.get_model_entry(res_name) is not None




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

        provider,name = cls.resolve_model_urn(res_name)

        prov = cls.models_dir[provider]

        del prov[name]






    @classmethod
    def resolve_model_urn(cls,
                          res_name: str) -> tuple[str,str]:
        """
        Checks format and if provider exists, follows string links until a dict or non-existent name key.
        res_name must be in format provider:model_name
        Returns tuple of provider_name, model_name. provider_name must exist, model_name may not
        """

        while True:
            provider, name = provider_name_from_urn(res_name, True)
                
            if provider not in cls.ALL_PROVIDER_NAMES:
                raise ValueError(f"Don't know how to handle provider '{provider}'. Can only handle the following providers: {cls.ALL_PROVIDER_NAMES}")

            prov = cls.models_dir[provider]

            if name in prov and isinstance(prov[name], str): # follow string link
                res_name = prov[name]
                if ":" not in res_name: # a local provider link
                    res_name = provider + ":" + res_name
                    
            elif provider == "alias" and name not in prov: # no alias with that name
                raise ValueError(f"Alias not found for '{name}'. Did you mean 'llamacpp:{name}' or 'openai:{name}'?")
                
            else: 
                break
            
        return provider, name
    

    @classmethod
    def save_models(cls,
                    path: Optional[str] = None):
        if path is None:
            if len(cls.models_search_path) != 1:
                raise ValueError("No path arg provided and multiple path in cls.search_path. Don't know where to save.")
            
            path = os.path.join(cls.models_search_path[0], "models.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(cls.models_dir, f, indent=4)

        return path







    # =========================================================================== Formats

    @classmethod
    def list_formats(cls,
                     name_query: str,
                     resolved_values: bool) -> dict:
        """List format entries matching query.

        Args:
            name_query: Case-insensitive substring to match format names. Empty string for all.
            resolved_values: Return resolved entries or raw ones.

        Returns:
            A dict where keys are format names and values are respective entries.
        """

        cls._ensure()

        out = {}

        name_query = name_query.lower()

        for name in cls.formats_dir.keys():

            if name_query and name_query not in name.lower():
                continue

            val = cls.formats_dir[name]

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

        na = name.lower()
        while na in cls.formats_dir.keys():
            val = cls.formats_dir[na]
            if isinstance(val, str): # str means link -> follow it
                na = val
            else:
                logger.debug(f"Format get('{name}'): found '{na}' entry")
                return na, resolve_format_entry(cls.formats_dir,
                                                na,
                                                val)
        return None
    
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

        return search_format(cls.formats_dir, name)


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






    @staticmethod
    def folder_match_format_entry(model_path: str) -> Union[tuple[str,dict],None]:
        """Locally search for format in a formats.json file located in the same folder as the model.
        Doesn't add format to class formats directory.

        Args:
            model_path: Model path.

        Returns:
            Tuple (name, format_entry) where name is a resolved name. Or None if none found.
        """

        folder_path = os.path.dirname(model_path)
        formats_path = os.path.join(folder_path, Models.FORMATS_CONF_FILENAME)
        if not os.path.isfile(formats_path):
            return None

        logger.info(f"Loading local formats conf from '{formats_path}'")

        formats_dir:dict = {}
        try:
            update_dir_json(formats_dir, formats_path)
        except Exception:
            raise ValueError(f"Could not load 'formats.json' at '{formats_path}', while looking for model format. "
                             "Please verify that he JSON syntax is correct.")

        sanity_check_formats(formats_dir)

        model_id = os.path.basename(model_path)
        return search_format(formats_dir, model_id)


    @staticmethod
    def folder_match_format_template(model_path: str) -> Union[str,None]:
        """Locally search for format in a formats.json file located in the same folder as the model.
        Doesn't add format to class formats directory.

        Args:
            model_path: Model path.

        Returns:
            Format template.
        """

        res = Models.folder_match_format_entry(model_path)
        return None if res is None else res[1]["template"]




    @classmethod
    def is_format_supported(cls,
                            model_id: str) -> bool:
        """Checks if there's template support for a model with this name.

        Args:
            model_id: Model filename or general name.

        Returns:
            True if Models knows the format.
        """

        return cls.match_format_entry(model_id) is not None







    @classmethod
    def set_format(cls,
                   name: str,
                   match: str,
                   template: str):
        """Add a format entry to the format directory.

        Args:
            name: Format entry name.
            match: Regex that matches names/filenames that use this format.
            template: The Chat template format in Jinja2 format
        """

        cls._ensure()

        if "{{" not in template: # a link_name for the template
            if not cls.has_format_entry(template):
                raise ValueError(f"Could not find linked template entry '{template}'.")

        entry = {
            "match": match,
            "template": template
        }
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

        if not cls.has_format_entry(name):
            raise ValueError(f"Format name '{name}' not found.")
        
        del cls.formats_dir[name]





    @classmethod
    def merge_from(cls,
                   path: str,
                   preserve_current: bool = True):
        path = expand_path(path)

        if preserve_current:
            with open(path, "r", encoding="utf-8") as f:
                new_dir = json.load(f)
            new_dir.update(cls.formats_dir)
            cls.formats_dir = new_dir

        else: # normal update: new with the same name will override current
            update_dir_json(cls.formats_dir, path)

        sanity_check_formats(cls.formats_dir)


    @classmethod
    def save_formats(cls,
                     path: Optional[str] = None):
        if path is None:
            if len(cls.models_search_path) != 1:
                raise ValueError("No path arg provided and multiple path in cls.search_path. Don't know where to save.")
            
            path = os.path.join(cls.models_search_path[0], "formats.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(cls.formats_dir, f, indent=4)

        return path






    # ================================================================== Lower level

    @classmethod
    def _ensure(cls,
                add_cwd: bool = True,
                load_base: bool = True,
                load_from_env: bool = True):
        """Make sure class is initialized.

        Env variable checked:
            SIBILA_MODELS: ';'-delimited folder list where to find models.        
        """

        if cls.models_dir is not None: 
            return
            
        # model and format dirs
        cls.models_dir = cls.EMPTY_MODELS_DIR
        cls.formats_dir = cls.EMPTY_FORMATS_DIR

        if add_cwd:
            # add "." to search path, so that paths relative paths work
            cls.add_models_search_path(".")


        path: Union[str, None]

        if load_base:
            # read base_models.json from res folder
            source = files(sibila.res).joinpath(cls.MODELS_BASE_CONF_FILENAME)
            with as_file(source) as src_path:
                path = str(src_path)
                if os.path.isfile(path):
                    cls._read_models(path,
                                     add_folder_to_search_path=False)

            # read base_formats.json from res folder
            source = files(sibila.res).joinpath(cls.FORMATS_BASE_CONF_FILENAME)
            with as_file(source) as src_path:
                path = str(src_path)
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
                     models_path: str,
                     add_folder_to_search_path: bool = True):
 
        logger.info(f"Loading models conf from: '{models_path}'")
        merge_dir_json(cls.models_dir,
                       models_path)

        cls._sanity_check_models()

        if add_folder_to_search_path:
            dir_path = os.path.dirname(models_path)
            cls.add_models_search_path(dir_path)


    @classmethod
    def _read_formats(cls,
                      formats_path: str):
        
        logger.info(f"Loading formats conf from '{formats_path}'")
        update_dir_json(cls.formats_dir, formats_path)

        sanity_check_formats(cls.formats_dir)
        


    @classmethod
    def _read_folder(cls,
                     path: str):

        if not os.path.isdir(path):
            raise OSError(f"Directory not found: '{path}'")
        
        cls.add_models_search_path(path)

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
        
        for dir in cls.models_search_path:
            full_path = os.path.join(dir, path)
            if os.path.isfile(full_path):
                return full_path
            
        return None

        


    @classmethod
    def _sanity_check_models(cls):
        
        for prov in cls.models_dir.keys():
            if prov == "alias":
                for name,link_name in cls.models_dir[prov].items():
                    if not isinstance(link_name, str):
                        raise ValueError(f"Alias entries must be strings at alias:{name}")
                        
            else: # real providers
                if prov not in cls.PROVIDER_CONF.keys():
                    raise ValueError(f"Don't know how to handle provider '{prov}'. Can only handle the following providers: {cls.PROVIDER_CONF.keys()}")
    
                # check if mandatory keys are in each model entry
                mandatory_keys = cls.PROVIDER_CONF[prov]["mandatory"]
                prov_models = cls.models_dir[prov]
                for model_name in prov_models.keys():
                    if model_name == cls.DEFAULT_ENTRY_NAME or isinstance(prov_models[model_name], str):
                        continue # skip "_default" args entry and string links
                    
                    model_entry = prov_models[model_name]
                    if not all(mand in model_entry for mand in mandatory_keys):
                        raise ValueError(f"Models entry '{prov}:{model_name}' doesn't have all mandatory keys for this provider ({mandatory_keys})") 
                               
                
        # ensure all providers have their own entry
        for p in cls.PROVIDER_CONF.keys():
            if p not in cls.models_dir.keys():
                cls.models_dir[p] = {}

        if "alias" not in cls.models_dir:
            cls.models_dir["alias"] = {}
            





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





def update_dir_json(dir: dict,
                    path: str):
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as f:
        new_dir = json.load(f)
        dir.update(new_dir)



def resolve_format_entry(formats_dir: dict,
                         name: str,
                         val: Union[dict,str]):
    val = copy(val)
    
    if "{{" not in val["template"]: # type: ignore[index] # a link to another template entry
        linked_name = val["template"] # type: ignore[index]
        if linked_name not in formats_dir:
            raise ValueError(f"Broken template link at '{name}': '{linked_name}' does not exist")
        val2 = formats_dir[linked_name]
        val["template"] = val2["template"] # type: ignore[index]
        
    return val




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
                logger.debug(f"Format search for '{model_id}' found '{name}' entry")
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
                    






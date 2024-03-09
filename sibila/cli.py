"""

Model search and download functionality based on llama-cpp-python's:
https://github.com/abetlen/llama-cpp-python/blob/main/docker/open_llama/hug_model.py


Downloads from Hugging Face models hub website:
 https://huggingface.co

 

# example Hugging Face URLS
https://huggingface.co/api/models?author=TheBloke&search=openchat-3.5&tag=
https://huggingface.co/api/models/TheBloke/openchat-3.5-0106-GGUF
https://huggingface.co/{model_id}/resolve/main/{filename}

"""

from typing import Any, Optional, Union, Callable

import requests, json, os, struct, argparse, signal, sys, tempfile
from pprint import pprint

import logging
logger = logging.getLogger(__name__)

from tqdm import tqdm

from sibila import (
    Models,
    GenConf
)
from sibila.models import search_format
from sibila.utils import expand_path



HF_MODEL_SEARCH_URL = "https://huggingface.co/api/models"
HF_MODEL_INFO_URL = "https://huggingface.co/api/models/{model_id}"
HF_MODEL_DOWNLOAD_URL = "https://huggingface.co/{model_id}/resolve/main/{filename}"
HF_ABOUT = "For information about this and other models, please visit https://huggingface.co"


BASE_FORMATS_URL = "https://raw.githubusercontent.com/jndiogo/sibila/main/sibila/res/base_formats.json"

DOWNLOAD_TIMEOUT = 30
INDENT = " " * 2



def die(message: str,
        exit_code: int = 1):
    print(message,
          file=sys.stderr)
    exit(exit_code)    



def resolve_models_dir(models_dir: Union[str,None],
                       can_default_to_current: bool) -> str:

    if models_dir is None: # check env variable SIBILA_MODELS

        path = os.environ.get(Models.ENV_VAR_NAME)
        if path:
            path_list = path.split(";")
            models_dir = path_list[0]

        elif (os.path.isfile(Models.MODELS_CONF_FILENAME) or
              os.path.isfile(Models.FORMATS_CONF_FILENAME)): # models.json or formats.json in current folder
            models_dir = "./"

        elif can_default_to_current: # safe to default to current folder
            models_dir = "./"

        else:
            die("Could not find a 'models' folder and can only default to current folder if 'models.json' or 'formats.json' is present.\n"
                "Please provide the location of a 'models' folder with option -m.")


    models_dir = expand_path(models_dir) # type: ignore[arg-type]

    if not os.path.isdir(models_dir):
        die(f"Unable to locate a 'models' directory. Please provide one with -model_dir arg or set env variable {Models.ENV_VAR_NAME}")

    return models_dir




def request_json(url: str, 
                 params: Optional[dict] = None):
    
    logger.info(f"Making request to {url}...")

    response = requests.get(url, params=params)

    if response.status_code == 200:
        return json.loads(response.text)
    else:
        logger.error(f"Request failed with status code {response.status_code}")
        return None
    


def download_file(url: str,
                  destination: str,
                  progress_bar: bool = True,
                  remove_partial: bool = True,
                  remove_on_ctrl_c: bool = True):
    """_summary_

    Args:
        url: _description_
        destination: _description_
        remove_on_error: _description_. Defaults to True.
        remove_on_interruption: _description_. Defaults to True.
    """

    response = requests.get(url,
                            stream=True,
                            timeout=DOWNLOAD_TIMEOUT)

    if response.status_code == 200:
        size_res = response.headers.get("Content-length")
        if size_res:
            size = int(size_res) # type: ignore[assignment]
        else:
            size = -1

        if remove_on_ctrl_c:
            def signal_handler(sig, frame):
                os.remove(destination)
                die(f"Removed partially downloaded file '{destination}'")

            original_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, signal_handler)

        try:
            with open(destination, 'wb') as f:

                format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                with tqdm(total=size,
                          bar_format=format, 
                          disable=not progress_bar,
                          leave=False) as pbar:

                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if chunk:  # filter out keep-alive new chunks
                            pbar.update(len(chunk))
                            f.write(chunk)

        except Exception as e:
            print(f"Download error:\n{e}")
            if remove_partial:
                os.remove(destination)
                die(f"Removed partially downloaded file '{destination}'")

        if remove_on_ctrl_c:
            signal.signal(signal.SIGINT, original_handler)

    else:
        die(f"Unable to download: status code {response.status_code}")





def search_hub_models(query: str,
                      filter_model_id_gguf: Optional[str] = None,
                      filter_filename: Optional[str] = None,
                      filter_filename_gguf: bool = True,
                      author: Optional[str] = None,
                      tags: Optional[str] = None) -> dict:
    """Search for models in Hugging Face model hub.

    Args:
        query: Text query to search for.
        max_models: Maximum count to return. Use -1 to return {} if more than 1 result is found. Defaults to 10e10.
        author: Optional author filter. Defaults to None.
        tags: Optional tags filter. Defaults to None.

    Returns:
        Dictionary where keys are model names, each entry contains "filenames": [] and "tags": []. Returns {} if not found.
    """
    if filter_filename is not None:
        filter_filename = filter_filename.lower()

    # query parameters
    params = {
        "search": query
    }
    if author is not None:
        params["author"] = author
    if tags is not None:
        params["tags"] = tags

    models = request_json(HF_MODEL_SEARCH_URL, params=params)
    logger.debug(f"HF_MODEL_SEARCH_URL params={params}:\n{models}")

    if models is None:
        return {}
    
    model_dic = {}

    # iterate over the models
    for model in models:
        model_id = model['id']

        if filter_model_id_gguf:
            if "gguf" not in model_id.lower():
                continue

        model_info = request_json(HF_MODEL_INFO_URL.format(model_id=model_id))

        logger.debug(f"HF_MODEL_INFO_URL model_id={model_id}:\n{model_info}")

        # print(model_info['siblings'])
        if model_info is None:
            continue

        entry = {"filenames": [],
                 "tags": model_info.get('tags', [])
                 }

        for sibling in model_info.get('siblings', []):
            filename = sibling.get('rfilename')
            if filename:
                if filter_filename_gguf and not filename.lower().endswith(".gguf"):
                    continue

                if filter_filename and filter_filename not in filename.lower():
                    continue

                entry["filenames"].append(filename)

        if len(entry["filenames"]):
            model_dic[model_id] = entry


    # sort by model name
    model_dic = dict(sorted(model_dic.items()))

    return model_dic




def list_hub_models_and_filenames(models_dic: dict) -> str:
    out = ""
    for model_id,dic in models_dic.items():
        out += f"{model_id}\n"
        for filename in dic["filenames"]:
            out += f"{INDENT}{filename}\n"
        out += "\n"
    return out



def check_res_name(name: str):
    if ":" not in name:
        die(f"Model res_name is not including 'provider:'. Did you mean 'llamacpp:{name}' or 'openai:{name}'?")



# ========================================================================= models

def models(args):

    can_default_to_current = bool(args.test_name)

    models_dir = resolve_models_dir(args.models_dir,
                                    can_default_to_current=can_default_to_current)
    
    print(f"Using models directory '{models_dir}'")

    Models.setup(models_dir,
                 clear=True,
                 add_cwd=False,
                 load_from_env=False)


    if args.set_resname_name_format: # sibila models -s res_name model_name_or_filename [format]

        if len(args.set_resname_name_format) < 2 or len(args.set_resname_name_format) > 3:
            die("Option -s requires 2 or 3 arguments: -s res_name name_or_filename [format]")

        res_name = args.set_resname_name_format[0]
        check_res_name(res_name)
        model_name = args.set_resname_name_format[1]

        if res_name.startswith("llamacpp:"):
            model_name = os.path.basename(model_name)

        format = args.set_resname_name_format[2] if len(args.set_resname_name_format) == 3 else None

        try:
            Models.set_model(res_name=res_name,
                             model_name=model_name,
                             format_name=format)
        
            path = Models.save_models()

            format_disp = "" if format is None else f", format='{format}'"
            print(f"Set model '{res_name}' with name='{model_name}'{format_disp} at '{path}'.")

        except Exception as e:
            die(f"Error: {e}")

    elif args.set_resname_link: # sibila models -sl res_name linked_name

        check_res_name(args.set_resname_link[0])

        try:
            Models.set_model_link(*args.set_resname_link)
            
            path = Models.save_models()
            print(f"Set model '{args.set_resname_link[0]}' linking to '{args.set_resname_link[1]}' at '{path}'.")

        except Exception as e:
            die(f"Error: {e}")


    elif args.format_resname_name: # sibila models -f res_name format

        res_name = args.format_resname_name[0]
        check_res_name(res_name)
        
        format_name = args.format_resname_name[1]
        try:
            Models.update_model(res_name,
                                model_name=None,
                                format_name=format_name,
                                genconf=None)

            path = Models.save_models()
            print(f"Updated model '{args.format_resname_name[0]}' with format '{args.format_resname_name[1]}' at '{path}'.")

        except Exception as e:
            die(f"Error: {e}")


    elif args.delete_name: # sibila models -d res_name 

        check_res_name(args.delete_name)

        try:
            Models.delete_model(args.delete_name)

            path = Models.save_models()
            print(f"Deleted model entry '{args.delete_name}' at '{path}'.")

        except Exception as e:
            die(f"Error: {e}")

    elif args.test_name: # sibila models -t res_name 

        check_res_name(args.test_name)

        try:
            print(f"Testing model '{args.test_name}'...")
            model = Models.create(args.test_name) # noqa: F841
            del model
            
            print(f"Model '{args.test_name}' was properly created and should run fine.")

        except Exception as e:
            die(f"Error: {e}")


    elif args.list_query or ( # horrible but necessary hack follows:
        "-l" in sys.argv or 
        "--list" in sys.argv):

        list_query = args.list_query if args.list_query is not None else ""

        if ":" in list_query:
            prov,name = list_query.split(":")
        else:
            prov = ""
            name = list_query

        dic = Models.list_models(name,
                                 providers=[prov] if prov else [],
                                 include_base=args.base,
                                 resolved_values=False)
        
        if dic:
            if args.base:
                print("Listing local and base models:")
            else:
                print("Listing only local models (without base models):")

            last_prov = None
            for res_name,val in dic.items():

                prov,name = res_name.split(":")
                if prov != last_prov:
                    print(f"\n{prov}:")
                    last_prov = prov

                print(INDENT, end="")
                print(f"{name}: ", end="")

                if isinstance(val, dict):
                    print(val)
                else: # link
                    print(f"-> {val}")

            print()
        else:
            print(f"No models found for query '{args.list_query}'")
        
    else:
        assert False, f"Impossible option in {args}"




# ========================================================================= formats

def formats(args):
    models_dir = resolve_models_dir(args.models_dir,
                                    can_default_to_current=False)
    
    print(f"Using models directory '{models_dir}'")

    Models.setup(models_dir,
                 clear=True,
                 add_cwd=False,
                 load_from_env=False)


    if args.query_name: # sibila formats -q query

        res = Models.match_format_entry(args.query_name)

        if res is None:
            print(f"No format found for name/filename '{args.query_name}'")
        else:
            name, dic = res
            print(f"Name/filename '{args.query_name}' matches format entry '{name}'.\nFormat entry '{name}':")
            print(INDENT + f"match: '{dic['match']}'")
            template = dic['template'].replace("\n", "\\n")
            print(INDENT + f"template: █{template}█")


    elif args.set_name_template_match: # sibila formats -s entry_name template [match_regex]

        if len(args.set_name_template_match) < 2 or len(args.set_name_template_match) > 3:
            die("Option -s requires 2 or 3 arguments: -s name template [match_regex]")

        try:
            name = args.set_name_template_match[0]
            template = args.set_name_template_match[1]
            match = args.set_name_template_match[2] if len(args.set_name_template_match) == 3 else None

            if '{{' not in template: # filename or link
                if os.path.isfile(template):
                    with open(template, "r", encoding="utf-8") as f:
                        template = f.read()
                elif Models.get_format_entry(template) is None:
                    raise ValueError("Can't understand template arg. It can be a Jinja2 text template, its filename or the name to an existing format entry (whose template will be used).")
                
            Models.set_format(name,
                              template,
                              match)
            
            path = Models.save_formats()
            template_disp = template if len(template) <= 24 else template[:24] + "..."
            print(f"Set format '{name}', template='{template_disp}'", end='')
            if match:
                print(f", match='{match}'")
            else:
                print()

        except Exception as e:
            die(f"Error: {e}")

    elif args.set_name_link: # sibila formats -sl entry_name linked_name

        try:
            Models.set_format_link(*args.set_name_link)
            
            path = Models.save_formats()
            print(f"Set format '{args.set_name_link[0]}' linking to '{args.set_name_link[1]}' at '{path}'.")

        except Exception as e:
            die(f"Error: {e}")


    elif args.delete_name: # sibila formats -d name 

        try:
            Models.delete_format(args.delete_name)

            path = Models.save_formats()
            print(f"Deleted format entry '{args.delete_name}' at '{path}'.")

        except Exception as e:
            die(f"Error: {e}")


    elif args.list_query or ( # horrible but necessary hack follows:
        "-l" in sys.argv or 
        "--list" in sys.argv): # sibila formats -l [query]

        dic = Models.list_formats(args.list_query if args.list_query is not None else "", 
                                  include_base=args.base,
                                  resolved_values=False)
        if dic:
            if args.base:
                print("Listing local and base formats:")
            else:
                print("Listing only local formats (without base formats):")

            for name,val in dic.items():

                print(f"\n{name}: ", end="")

                if isinstance(val, dict):
                    print(val)
                else: # link
                    print(f"-> {val}")

            print()

        else:
            print(f"No formats found for query '{args.list_query}'")


    else:
        assert False, f"Impossible option in {args}"




# ========================================================================= hub

def hub(args):
    has_set = args.set_resname_format is not None

    if has_set and args.model_download is None:
        die("Option error: -s/--set can only be used with option -d/--download.")


    if args.model_download: # sibila hub -d model_id -f filename -a exact_author -s set name

        models_dir = resolve_models_dir(args.models_dir,
                                        # args.set requires an explicit model_dir, to avoid mistakes
                                        can_default_to_current=not has_set)
        
        print("Searching...")
            
        query = args.model_download
        models_dic = search_hub_models(query, 
                                       author=args.author,
                                       filter_filename=args.filename,
                                       filter_model_id_gguf=True,
                                       filter_filename_gguf=True)
        
        logger.debug(f"search_models():\n{models_dic}")

        if len(models_dic) == 0:
            die(f"Could not find any model matching  query '{query}'.\n" +
                "Run 'sibila hf -l query' to find model names and filenames.")
            
        elif len(models_dic) > 1:
            mlist = list_hub_models_and_filenames(models_dic)

            die(f"Found multiple models matching query '{query}':\n\n{mlist}" +
                "Restrict terms in your query or use -a author and -f filename.\n" + 
                HF_ABOUT)

        model_id = next(iter(models_dic))
        filenames = models_dic[model_id]["filenames"]

        if len(filenames) > 1:
            flist = INDENT + ("\n" + INDENT).join(filenames)

            die(f"Multiple filenames found:\n\n{flist}\n\n" +
                "Filter one of the above with -f 'filename_filter'. For example: -f Q4_K_M\n" + 
                HF_ABOUT)

        filename = filenames[0]

        logger.debug(f"HF_MODEL_DOWNLOAD_URL model_id={model_id} filename={filename}")

        url = HF_MODEL_DOWNLOAD_URL.format(model_id=model_id,
                                           filename=filename)
    
        path = os.path.join(models_dir, filename)

        print(f"Downloading model '{model_id}' file '{filename}' to '{path}'")

        download_file(url, path)

        print(f"\nDownload complete.\n{HF_ABOUT}")

        if args.set_resname_format:

            if len(args.set_resname_format) > 2:
                die("Option -s requires 1 or arguments: -s res_name [format]")
            
            Models.setup(models_dir,
                         clear=True,
                         add_cwd=False,
                         load_from_env=False)

            name = args.set_resname_format[0]
            format_name = args.set_resname_format[1] if len(args.set_resname_format) >= 2 else None

            if not name.startswith("llamacpp:"):
                name = "llamacpp:" + name

            genconf = None
            existing = Models.get_model_entry(name)
            if existing:
                entry = existing[1]
                if format_name is None and "format" in entry:
                    format_name = entry["format"]

                if "genconf" in entry:
                    genconf = GenConf.from_dict(entry["genconf"])


            Models.set_model(res_name=name, 
                             model_name=filename,
                             format_name=format_name,
                             genconf=genconf)


            models_dir_path = os.path.join(models_dir, 'models.json')
            Models.save_models(models_dir_path)

            print(f"\nAlso set model entry with name '{name}' at '{models_dir_path}'.\n")


    elif args.model_list: # sibila hub -l query -f filename -a exact_author 

        print("Searching...")

        models_dic = search_hub_models(args.model_list, 
                                       author=args.author,
                                       filter_model_id_gguf=True,
                                       filter_filename=args.filename,
                                       filter_filename_gguf=True)
        
        if len(models_dic) == 0:
            print("No models found.")
        else:
            text = list_hub_models_and_filenames(models_dic)
            print()
            print(text, end='')
            print(HF_ABOUT)

    else:
        raise ValueError("Unknown option")









def main():


    # create argument parser
    parser = argparse.ArgumentParser(
        prog="sibila",
        description="Sibila CLI tool for managing models and formats.",
        epilog="For help, check https://jndiogo.github.io/sibila/")


    from .__init__ import __version__ as version  # type: ignore[import-not-found]
    parser.add_argument('--version', action='version', version='%(prog)s ' + version)

    subparser = parser.add_subparsers(title="actions",
                                      description="Use 'models' and 'formats' to manage, 'hub' to search and download models.",
                                      help="Run 'sibila {command} --help' for specific help.",
                                      required=True)



    def add_common(parser: Any):
        parser.add_argument("-m", "--models_dir",                             
                            type=str,
                            help="Path to 'models' folder. If not provided, looks for SIBILA_MODELS env variable or defaults to current folder in some (safer) commands.")
        parser.add_argument('-v', '--verbose', 
                            action='count',
                            default=0,
                            help='Verbose output level, Repeat argument for more verboseness')



    # ============================================================================= models
    parser_models = subparser.add_parser('models',
                                         description="Manage a models.json config file.")

    exclusive_models = parser_models.add_mutually_exclusive_group(required=True)
    
    exclusive_models.add_argument('-l', '--list',
                                  metavar="QUERY",
                                  type=str,
                                  nargs="?",
                                  default="",
                                  dest="list_query",
                                  help="List models filtering by model name substring or res_name in the form 'provider:query'. Examples: 'gpt', 'llamacpp:', 'llamacpp:open'.")

    parser_models.add_argument('-b', '--base', 
                               action="store_true",
                               default=False,
                               help="Also list base models. Use only with option -l/--list.")

    exclusive_models.add_argument('-t', '--test', 
                                  metavar="NAME",
                                  type=str,
                                  dest="test_name",
                                  help="Test if model entry with this res_name can be created. Checks if format template is present.")

    exclusive_models.add_argument('-s', '--set',
                                  metavar="NAME",
                                  type=str,
                                  nargs='+',
                                  dest="set_resname_name_format",
                                  help="Set model entry with 2 or 3 args: res_name name_or_filename [format].")
    
    exclusive_models.add_argument('-sl', '--setlink',
                                  metavar="NAME",
                                  type=str,
                                  nargs=2,
                                  dest="set_resname_link",
                                  help="Set model link with 2 args: res_name linked_name.")

    exclusive_models.add_argument('-f', '--format', 
                                  metavar="NAME",
                                  type=str,
                                  nargs=2,
                                  dest="format_resname_name",
                                  help="Update format in existing model, 2 args: res_name format.")

    exclusive_models.add_argument('-d', '--delete', 
                                  metavar="NAME",
                                  type=str,
                                  dest="delete_name",
                                  help="Delete model res_name entry.")

    add_common(parser_models)
    parser_models.set_defaults(func=models)




    # ============================================================================= formats
    parser_formats = subparser.add_parser('formats',
                                          description="Manage a formats.json config file.")

    exclusive_formats = parser_formats.add_mutually_exclusive_group(required=True)
    
    exclusive_formats.add_argument('-l', '--list',
                                   metavar="QUERY",
                                   type=str,
                                   nargs="?",
                                   default="",
                                   dest="list_query",
                                   help="List formats filtering by name substring.")

    parser_formats.add_argument('-b', '--base', 
                                action="store_true",
                                default=False,
                                help="Also list base formats. Use only with option -l/--list.")

    exclusive_formats.add_argument('-q', '--query',
                                   metavar="QUERY",
                                   type=str,
                                   dest="query_name",
                                   help="Search for a format from this model name or filename in 'formats.json'.")

    exclusive_formats.add_argument('-s', '--set',
                                   metavar="NAME",
                                   type=str,
                                   nargs='+',
                                   dest="set_name_template_match",
                                   help="Set format entry with 2 or 3 args: name template match_regex. Arg template can be a Jinja2 template string or filename, or the name of an existing template format entry.")
    
    exclusive_formats.add_argument('-sl', '--setlink',
                                   metavar="NAME",
                                   type=str,
                                   nargs=2,
                                   dest="set_name_link",
                                   help="Set format link with 2 args: name linked_name.")


    exclusive_formats.add_argument('-d', '--delete', 
                                   metavar="NAME",
                                   type=str,
                                   dest="delete_name",
                                   help="Delete format entry.")

   
    add_common(parser_formats)
    parser_formats.set_defaults(func=formats)




    # ============================================================================= hub
    parser_hf = subparser.add_parser('hub',
                                     description="Search for and download models from Hugging Face hub.")

    exclusive_hf = parser_hf.add_mutually_exclusive_group(required=True)

    exclusive_hf.add_argument('-d', '--download', 
                              metavar="MODEL_ID",
                              type=str,
                              dest="model_download",
                              help="Model id or name. Query is case-insensitive, allows multiple words separated with space. For example: TheBloke/openchat-3.5-1210-GGUF")

    exclusive_hf.add_argument('-l', '--list', 
                              metavar="QUERY",
                              type=str,
                              dest="model_list",
                              help="List model names matching query. Query is case-insensitive, allows multiple words separated with space. For example: openchat-3.5. ")


    parser_hf.add_argument("-f", "--filename", 
                           type=str, 
                           help='Case-insensitive filename substring match, for example: Q4_K_M')

    parser_hf.add_argument('-a', '--author', 
                           metavar="EXACT_AUTHOR",
                           type=str, 
                           help="Only from this author. Exact, case-sensitive author name. For example: TheBloke")

    parser_hf.add_argument("-s", "--set", 
                           metavar="NAME",
                           type=str,
                           nargs='+',
                           dest="set_resname_format",
                           help="Set model entry with args: res_name [format]. Use only with option -d/--download.")

    add_common(parser_hf)
   
    parser_hf.set_defaults(func=hub)

    


    args = parser.parse_args()

    if vars(args).get("verbose"):
        logging.basicConfig(level=logging.INFO if args.verbose == 1 else logging.DEBUG,
                            force=True)
        print(vars(args))
    
    args.func(args)
    exit(0)




if __name__ == '__main__':
    main()

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]
- feat: Add seed setting to GenConf. Commented-out because of lack of support in OpenAI models and some llama.cpp hardware accelerations.

## [0.4.0]
- feat: New providers: Mistral AI, Together.ai and Fireworks AI allowing access to all their chat-based models.
- feat: Model classes now support async calls with the '_async' prefix, for example extract_async(). This requires model API support: only remote models will benefit. Local models (via llama.cpp) can still be called with _async methods but do not have async IO that can run concurrently.
- feat: Add 'special' field to GenConf, allowing provider or model specific generation arguments.
- feat: All models now also accept model path/name starting with their provider names as in Models.create().
- feat: Change Model.json() to stop requiring a JSON Schema as first argument.
- fix: More robust JSON extraction for misbehaved remote models.
- fix: LlamaCppModel no longer outputting debug info when created in Jupyter notebook environment with verbose=False.
- fix: Default "gpt-4" model in 'sibila/res/base:models.json' now points to gpt-4-1106-preview, the first GPT-4 model that accepts json-object output.
- docs: Add API references for new classes and _async() methods.
- docs: Add new async example.
- test: Add new tests for new providers/model classes.

## [0.3.6]
- feat: Migrate hardcoded OpenAI model entries from OpenAIModel to 'res/base_models.json'.
- feat: OpenAI now accepts unknown models using defaults from 'openai:_default' key in  'res/base_models.json'.
- feat: Support OpenAI models with a limit on max_tokens output values, like "gpt-4-turbo-preview" (input ctx_len of 128k but only up to 4k output tokens).
- feat: Auto-discover maximum ctx_len in LlamaCppModel loaded files, when 0 is passed.
- feat: Add negative int factor mode to GenConf.max_tokens setting, allowing for a percentage of model's context length.
- fix: Add coherent error exceptions when loading local and remote models.
- fix: Correct interact() error when GenConf.max_tokens=0.
- fix: Correct several chat template formats.
- test: Add many new tests for gpt-3.5/4 and llama.cpp models.
- docs: Update tips section.

## [0.3.5]
- feat: Split Models factory config in two levels: base definitions in sibila/res and Models.setup() loaded definitions from user folders. These levels never mix, but a fusion of the two is used for models/formats resolution. Only in this manner can "models" folder definitions be kept clean.
- fix: Option sibila formats -u is removed as result of the two-level Models factory.
- fix: Correct delete of link entries in models.json and formats.json, which was resolving to targets (and deleting them).
- fix: Raise ValueError when trying to generate from an empty prompt in LLamaCppModel.
- fix: Update Models to check linked entries when deleting.
- fix: Update template format discovery to work in more edge cases.
- test: Add test cases for sibila CLI and LlamaCppModel.

## [0.3.4]
- feat: Improve template format discovery by looking in same folder for models/formats.json.
- fix: Update legacy importlib_resources reference.
- docs: Improve text.

## [0.3.3]
- fix: Move base_models.json and base_formats.json to sibila/res folder.
- fix: Add base_models.json and base_formats.json to project build.
- fix: Correct .gitignore skipping valid files.
- docs: Update installation help and mentions to base_models/formats.json.

## [0.3.2]
- feat: Added sibila CLI for models and formats management.
- feat: Added methods in Models class for CLI functionality.
- fix: Blacklisting character control set in JSON strings grammar.
- docs: Improved docs and added section about sibila CLI.
- docs: Added CLI example.

## [0.3.1]
- feat: Improved documentation.
- feat: Model.known_models() returns a list of fixed known models or None if unlimited.
- feat: LlamaCppModel now also looks for the chat template format in a 'formats.json' file in the same folder as the model file.
- feat: Added GenConf.from_dict() and renamed asdict() to as_dict().
- fix: Creating a model entry in "models.json" with a genconf key was not being passed on model creation.

## [0.3.0]
- feat: Added Models singleton class that centralizes ModelDir and FormatDir.
- feat: New extract() and classify() methods for type-independent extraction and classification.
- feat: Renamed confusing gen() and gen_() method names to simpler alternatives type() and gen_type().
- feat: Replaced dictype definitions with dataclasses, a better to extract dictionaries.
- feat: Added version() and provider_version() to Model and children classes.
- fix: Using 2 * "\n" to separate message text from automatically added json_format_instructors ("Output JSON", etc.), to provide more meaningful separation.
- fix: Added requirement for package typing_extensions because of Self type and Python 3.9+ compatibility.

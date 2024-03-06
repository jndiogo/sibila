# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]


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

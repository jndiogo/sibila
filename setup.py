from setuptools import setup
import sys

if sys.version_info < (3,9):
    raise ValueError("Sibila requires python 3.9 or above.")


setup(name='Sibila',
    version='0.2.0',
    author='Jorge Diogo',
    packages=['sibila'],
    description='Use local or online large language models for structured output',
    license='MIT',
    install_requires=[
        'llama-cpp-python',
        'jinja2',
        'jsonschema',
        'openai',
        'tiktoken',
        # can be optionally installed:
        # 'pydantic', 
        # 'python-dotenv',
    ],
)

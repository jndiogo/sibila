from setuptools import setup
import sys

setup(name='Sibila',
      version='0.2.1',
      author='Jorge Diogo',
      packages=['sibila'],
      description='Structured queries from local or online LLM models',
      license='MIT',
      url="https://github.com/jndiogo/sibila",
      python_requires=">=3.9",
      install_requires=[
          'llama-cpp-python',
          'jinja2',
          'jsonschema',
          'openai',
          'tiktoken',
          'pydantic',
          # can be optionally installed:
          # 'python-dotenv',
      ],
)

# Test requirements

To run the tests with all providers, please do the following:

1. Copy the model 'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf' file into the res/ folder.

2. API keys for the supported providers must be set in env vars or stored in a .env file in the current or a parent folder. The following API keys are needed to run all tests:

    ANTHROPIC_API_KEY = "..."

    FIREWORKS_API_KEY = "..."

    GROQ_API_KEY = "..."

    MISTRAL_API_KEY = "..."

    OPENAI_API_KEY = "..."

    TOGETHER_API_KEY = "..."

3. For testing script 'test_common_llamacpp.py', there must exist a 'models' folder located at '../../models', with all the models listed in that script.


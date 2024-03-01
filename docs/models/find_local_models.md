---
title: Finding new models
---


## Chat or instruct types only

Sibila can use models that were fine-tuned for chat or instruct purposes. These models work in user - assistant turns or messages and use a chat template to properly compose those messages to the format that the model was fine-tuned to.

For example, the Llama2 model was released in two editions: a simple Llama2 text completion model and a Llama2-instruct model that was fine tuned for user-assistant turns. For Sibila you should always select chat or instruct versions of a model.

But which model to choose? You can look at model benchmark scores in popular listing sites:

- [https://llm.extractum.io/list/](https://llm.extractum.io/list/)
- [https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)
- [https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)




## Find a quantized version of the model

Since Large Language Models are quite big, they are usually quantized so that each parameter occupies a little more than 4 bits or half a byte. 

Without quantization, a 7 billion parameters model would require 14Gb of memory (with each parameter taking 16 bits) to load and a bit more during inference.

But with quantization techniques, a 7 billion parameters model can have a file size of only 4.4Gb (using about 50% more in memory - 6.8Gb), which makes it accessible to be ran in common GPUs or even in common RAM memory (albeit slower).

Quantized models are stored in a file format popularized by llama.cpp, the GGUF format (which means GPT-Generated Unified Format). We're using llama.cpp to run local models, so we'll be needing GGUF files.

A good place to find quantized models is in HuggingFace's model hub, particularly in the well-know TheBloke's (Tom Jobbins) area:

[https://huggingface.co/TheBloke](https://huggingface.co/TheBloke)


TheBloke is very prolific in producing quality quantized versions of models, usually shortly after they are released.

And a good model that we'll be using for the examples is a 4 bit quantization of the OpenChat-3.5 model, which itself is a fine-tuning of Mistral-7b:

[https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF](https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF)




## Download the file into the "models" folder

From HuggingFace, you can download the GGUF file (in this and any other quantized models in HuggingFace) by scrolling down to the "Provided files" section and clicking one of the links. Usually the files ending in "Q4_K_M" are very reasonable 4-bit quantizations.

In this case you'd download the file "openchat-3.5-1210.Q4_K_M.gguf" - save it into the "models" folder inside Sibila.

Because these models use a chat template format, we need to make sure it's the right one - see the [Setup chat template format](setup_format.md) section on how to handle this.

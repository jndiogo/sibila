---
title: Vision models
---

Vision models allow you to provide an image alongside your text query. Elements in this image can be referenced and its data extracted with normal methods like Model.extract() or Model.classify().

!!! example

    The photo variable below references this image, but a local image file path could also be provided:

    ![Image](https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Hohenloher_Freilandmuseum_-_Baugruppe_Hohenloher_Dorf_-_Bauerngarten_-_Ansicht_von_Osten_im_Juni.jpg/640px-Hohenloher_Freilandmuseum_-_Baugruppe_Hohenloher_Dorf_-_Bauerngarten_-_Ansicht_von_Osten_im_Juni.jpg)

    ``` python
    from sibila import Models

    model = Models.create("openai:gpt-4o")

    photo = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Hohenloher_Freilandmuseum_-_Baugruppe_Hohenloher_Dorf_-_Bauerngarten_-_Ansicht_von_Osten_im_Juni.jpg/640px-Hohenloher_Freilandmuseum_-_Baugruppe_Hohenloher_Dorf_-_Bauerngarten_-_Ansicht_von_Osten_im_Juni.jpg"

    model.extract(list[str],
                  ("Extract up to five of the most important elements in this photo.",
                   photo))
    ```

    !!! result
        ```
        ['House with red roof and beige walls',
         'Large tree with green leaves',
         'Garden with various plants and flowers',
         'Clear blue sky',
         'Wooden fence']
        ```

To pass an image location we can pass a tuple of (text,image_location) as in the example above. This tuple is a shortcut to create an Msg with the text prompt and the image location. See [Threads and messages](../thread.md) for more information.


## Remote models

At the time of writing (June 2024), the following remote vision models can be used in Sibila:

| Provider  | Models |
|-----------|--------|
| OpenAI    | gpt-4o |
| Anthropic | all models |

Of these, the OpenAI model is currently the most capable one (with regards to images).


## Local models

Local models are supported via Llama.cpp and its Llava engine. This means that two models have to be loaded: the text model and a projector model. The two models are passed by separating the GGUF filenames with a "*". For example:

``` python
# note the * separating the two GGUF files:
name = "moondream2-text-model-f16.gguf*moondream2-mmproj-f16.gguf"

model = LlamaCppModel(name,
                      ctx_len=2048)

# or via Models.create()
model = Models.create("llamacpp:" + name, 
                      ctx_len=2048)
```

In the example above, the context length argument ctx_len is being set, because image inputs do consume tokens, so a larger context is a good idea.


A list of small (up to 8B params) open source models available on June 2024:

| Model and HuggingFace link | GGUF filenames |
|----------------------------|----------------|
| [Llava-v1.5](https://huggingface.co/mys/ggml_llava-v1.5-7b) | llava-v1.5-ggml-model-q4_k.gguf*llava-v1.5-mmproj-model-f16.gguf |
| [Llava-v1.6 mistral](https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf) | llava-v1.6-mistral-7b.Q4_K_M.gguf*llava-v1.6-mistral-mmproj-model-f16.gguf |
| [Llava-v1.6 vicuna 7B](https://huggingface.co/cjpais/llava-v1.6-vicuna-7b-gguf) | llava-v1.6-vicuna-7b.Q4_K_M.gguf*llava-v1.6-vicuna-mmproj-model-f16.gguf |
| [Moondream2](https://huggingface.co/vikhyatk/moondream2) | moondream2-text-model-f16.gguf*moondream2-mmproj-f16.gguf |
| [Llava-phi-3](https://huggingface.co/xtuner/llava-phi-3-mini-gguf) | llava-phi-3-mini-int4.gguf*llava-phi-3-mini-mmproj-f16.gguf |
| [Llava-llama-3](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-gguf) | llava-llama-3-8b-v1_1-int4.gguf*llava-llama-3-8b-v1_1-mmproj-f16.gguf |
| [Llama3-vision](https://huggingface.co/qresearch/llama-3-vision-alpha-hf) | Meta-Llama-3-8B-Instruct-Q4_K_M.gguf*llama-3-vision-alpha-mmproj-f16.gguf |

Some of the filenames might have been renamed from the original downloaded names to avoid name collisions.

At the current time, these small models are mostly only capable of description tasks. Some larger 34B variants are also available.




---
title: Using a local model
---


Sibila uses llama.cpp to run local models, which are ordinary files in the GGUF format. You can download local models from places like the [HuggingFace model hub](https://huggingface.co/models).

Most current 7B quantized models are very capable for common data extraction tasks (and getting better all the time). We'll see how to find and setup local models for use with Sibila. If you only plan to use OpenAI remote models, you can skip this section.





<a id="examples"></a>

## OpenChat GGUF


By default, most of the examples included with Sibila use OpenChat, a very good 7B parameters quantized model, that you can download from:

[https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF/blob/main/openchat-3.5-1210.Q4_K_M.gguf](https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF/blob/main/openchat-3.5-1210.Q4_K_M.gguf)

In the linked page, click "download" and save this file into a "models" folder. If you downloaded the Sibila GitHub repository it already includes a "models" folder which you can use. Otherwise, just create a "models" folder, where you'll store your local model files.

Once the file "openchat-3.5-1210.Q4_K_M.gguf" is placed in the "models" folder, you should be able to run the examples.


## LlamaCppModel class

Local llama.cpp models can be used with the [LlamaCppModel](../api-reference/model.md#sibila.LlamaCppModel) class. Let's generate some text:

!!! example
    ``` python
    from sibila import LlamaCppModel

    model = LlamaCppModel("../../models/openchat-3.5-1210.Q4_K_M.gguf")

    model("I think that I shall never see.")
    ```

    !!! success "Result"
        ```
        'A poem as lovely as a tree.'
        ```

It worked: the model answered with the continuation of the famous poem.

You'll notice that the first time you create the model object and run a query, it will take longer, because the model must load all its parameters into layers in memory. The next queries will work much faster.





## A note about out of memory errors

An important thing to know if you'll be using local models is about "Out of memory" errors.

A 7B model like OpenChat-3.5, when quantized to 4 bits will occupy about 6.8 Gb of memory, in either GPU's VRAM or common RAM. If you try to run a second model at the same time, you might get an out of memory error and/or llama.cpp may crash: it all depends on the memory available in your computer.

This is less of a problem when running scripts from the command line, but in environments like Jupyter where you can have multiple open notebooks, you may get "out of memory" errors or python kernel errors like:

!!! error
    ```
    Kernel Restarting
    The kernel for sibila/examples/name.ipynb appears to have died.
    It will restart automatically.
    ```

If you get an error like this in JupyterLab, open the Kernel menu and select "Shut Down All Kernels...". This will get rid of any out-of-memory stuck models.

A good practice is to delete any local model after you no longer need it or right before loading a new one. A simple "del model" works fine, or you can add these two lines before creating a model:

``` python
try: del model
except: ...

model = LlamaCppModel(...)
```

This way, any existing model in the current notebook is deleted before creating a new one.

However this won't work across multiple notebooks. In those cases, open JupyterLab's Kernel menu and select "Shut Down All Kernels...". This will get rid of any models currently in memory.


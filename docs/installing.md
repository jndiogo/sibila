---
title: Installing
---

## Installation

Sibila requires Python 3.9+ and uses the llama-cpp-python package for local models and OpenAI's API to access remote models like GPT-4.

Install Sibila from PyPI by running:

```
pip install sibila
```

If you only plan to use remote models (OpenAI), there's nothing else you need to do. See [First Run](first_run.md) to get it going.



??? info "Installation in edit mode"
    Alternatively you can install Sibila in edit mode by downloading the GitHub repository and running the following in the base folder of the repository:

    ```
    pip install -e .
    ```




## Enabling llama.cpp hardware acceleration

If you intend to run local models, they can run without hardware acceleration but that will be slower.

Sibila uses llama-cpp-python, a python wrapper for llama.cpp and it's a good idea to make sure it was installed with the best optimization your computer can offer. 

Run one of the following lines in the command prompt, depending on which hardware you have:


### For CUDA - NVIDIA GPUs

=== "Linux and Mac"
    ```
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" \
    pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
    ```

=== "Windows"
    ```
    $env:CMAKE_ARGS = "-DLLAMA_CUBLAS=on"
    pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
    ```


### For Metal - Apple silicon macs

=== "Mac"
    ```
    CMAKE_ARGS="-DLLAMA_METAL=on" \
    pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
    ```




### For ROCm AMD GPUS

=== "Linux and Mac"
    ```
    CMAKE_ARGS="-DLLAMA_HIPBLAS=on" \
    pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
    ```

=== "Windows"
    ```
    $env:CMAKE_ARGS = "-DLLAMA_HIPBLAS=on"
    pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
    ```



### For Vulkan supporting GPUs

=== "Linux and Mac"
    ```
    CMAKE_ARGS="-DLLAMA_VULKAN=on" \
    pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
    ```

=== "Windows"
    ```
    $env:CMAKE_ARGS = "-DLLAMA_VULKAN=on"
    pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
    ```



### CPU acceleration (if none of the above)

=== "Linux and Mac"
    ```
    CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" \
    pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
    ```

=== "Windows"
    ```
    $env:CMAKE_ARGS = "-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
    pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
    ```




If you get an error, please see [llama-cpp-python's Installation configuration](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#installation-configuration).


You can always install llama-cpp-python without acceleration by running:

```
pip install llama-cpp-python
```

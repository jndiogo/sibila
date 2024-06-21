---
title: Installing
---

## Installation

Sibila requires Python 3.9+ and uses the llama-cpp-python package for local models and OpenAI/Mistral/other libraries to access remote models.

Install Sibila from PyPI by running:

```
pip install -U sibila
```

If you only plan to use remote models (OpenAI), there's nothing else you need to do. See [First Run](first_run.md) to get it going.



??? info "Installation in edit mode"
    Alternatively you can install Sibila in edit mode by downloading the GitHub repository and running the following in the base folder of the repository:

    ```
    pip install -e .
    ```




## Enabling llama.cpp hardware acceleration for local models

Local models will run faster with hardware acceleration enabled. Sibila uses llama-cpp-python, a python wrapper for llama.cpp and it's a good idea to make sure it was installed with the best optimization your computer can offer. 

See the following sections: depending on which hardware you have, you can run the listed command which will reinstall llama-cpp-python with the selected optimization. If any error occurs you can always install the non-accelerated version, as listed at the end. 


### For CUDA - NVIDIA GPUs

For CUDA acceleration in NVIDIA GPUs, you'll need to have the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed. If looking for a specific CUDA version, [see here](https://developer.nvidia.com/cuda-toolkit-archive).


=== "Linux"
    ```
    CMAKE_ARGS="-DLLAMA_CUDA=on" \
    pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
    ```
    The CUDA toolkit can also be installed from your Linux distro's package manager (e.g. apt install nvidia-cuda-toolkit).

=== "Windows"
    ```
    $env:CMAKE_ARGS = "-DLLAMA_CUDA=on"
    pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
    ```    
    [Installing llama-cpp-python with NVIDIA GPU Acceleration on Windows: A Short Guide](https://medium.com/@piyushbatra1999/installing-llama-cpp-python-with-nvidia-gpu-acceleration-on-windows-a-short-guide-0dfac475002d)


More info: [Installing llama-cpp-python with GPU Support](https://michaelriedl.com/2023/09/10/llama2-install-gpu.html).





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




If you get an error running the above commands, please see [llama-cpp-python's Installation configuration](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#installation-configuration).


### Non-accelerated

In any case, you can always install llama-cpp-python without acceleration by running:

```
pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

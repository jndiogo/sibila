# Tips and Tricks

Some general tips from experience with constrained model output in Sibila.

## Temperature

Sibila aims at exact results, so generation temperature defaults to 0. You should get the same results from the same model at all times.

For "creative" outputs, you can set the temperature to a non-zero value. This is done in GenConf, which can be passed in many places, for example during actual generation/extraction:


!!! example
    ``` python
    from sibila import (Models, GenConf)

    Models.setup("../models")

    model = Models.create("llamacpp:openchat") # default GenConf could be passed here

    for i in range(10):
        print(model.extract(int,
                    "Think of a random number between 1 and 100",
                    genconf=GenConf(temperature=2.)))
    ```

    !!! success "Result"
        ```
        72
        78
        75
        68
        39
        47
        53
        82
        72
        63
        ```






## Deterministic outputs

With temperature=0 and given a certain seed in GenConf, we should always get the same output for a fixed input prompt to a certain model. 

From what we've observed, in practice, when extracting structured data, you'll find variation inside free-form str fields, where the model is not being constrained. Other types like numbers will rarely see variating outputs.



### OpenAI models

In the OpenAI API link below, about "Reproducible outputs" you can read:

    "To receive (mostly) deterministic outputs across API calls, you can..."

    "There is a small chance that responses differ even when request parameters and system_fingerprint match, due to the inherent non-determinism of our models."

As far as logic goes, "mostly deterministic" and "inherent non-determinism" means not deterministic, so it seems you you can't have it in these models.

https://platform.openai.com/docs/guides/text-generation/reproducible-outputs

https://cookbook.openai.com/examples/reproducible_outputs_with_the_seed_parameter



### Local llama.cpp models

Some hardware accelerators like NVIDIA CUDA GPUS sacrifice determinism for better inference speed.

You can find more information in these two links:

https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

https://github.com/ggerganov/llama.cpp/issues/1340


This happens inside CUDA hardware and is not related with the seed number you set in GenConf - it also happens if you always provide the same seed number. 

Interestingly, there is a pattern: in CUDA, if you set a fixed GenConf seed and generate multiple times after creating the model, the first output will be different and all the others will be equal. Sounds like some sort of warm-up, and can be accounted for by generating an initial dummy output (from the same inputs), after creating the model.

We've never observed non-determinist outputs for llama.cpp fully running in the CPU, without hardware acceleration and this is probably true of other platforms. Given the same seed number and inputs you'll always get the same result when running in the CPU.

It's something that should not have a great impact, but that's important to be aware of.





## Split entities into separate classes

Suppose you want to extract a list of person names from a group. You could use the following class:

``` python
class Group(BaseModel):
    persons: list[str] = Field(description="List of persons")
    group_info: str

out = model.extract(Group, in_text)
```

But it tends to work better to separate the Person entity into its own class and leave the list in Group:

``` python
class Person(BaseModel):
    name: str

class Group(BaseModel):
    persons: list[Person]
    group_info: str

out = model.extract(Group, in_text)
```

The same applies to the equivalent dataclass definitions.

Adding descriptions seems to always help, specially for non-trivial extraction. Without descriptions, the model can only look into variable names for clues on what's wanted, so it's important to tell it what we want by adding field descriptions.

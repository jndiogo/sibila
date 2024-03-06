In this example we'll see how to use the sibila Command-Line Interface (CLI) to download a GGUF model from the [Hugging Face model hub](https://huggingface.co).

We'll then register it in the Models factory, so that it can be easily used with Models.create(). The Models factory is based in a folder where model GGUF format files are stored and two configuration files: "models.json" and "formats.json".

After Doing the above, we'll be able to use this model in Python with two lines:

``` python
Models.setup("../../models")

model = Models.create("llamacpp:rocket")
```

Let's run sibila CLI to get help:

```
> sibila --help

usage: sibila [-h] [--version] {models,formats,hub} ...

Sibila cli tool for managing models and formats.

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit

actions:
  hf, models, formats

  {models,formats,hub}  Run 'sibila {command} --help' for specific help.
```

Sibila CLI has three modes:

- models: to edit a 'models.json' file, create model entries set format, etc.
- formats: to edit a 'formats.json' file, add new formats, etc.
- hub: search and download models from Hugging Face model hub.

Specific help for each mode is available by doing: sibila mode --help


Let's download the Rocket 3B model, a small but capable model, fine-tuned for chat/instruct prompts:

[https://huggingface.co/TheBloke/rocket-3B-GGUF](https://huggingface.co/TheBloke/rocket-3B-GGUF)

We'll use a "sibila hub -d" command to download to "../../models" folder. We'll get the 4-bit quantization (Q4_K_M):

```
> sibila hub -d 'TheBloke/rocket-3B-GGUF' -f Q4_K_M -m '../../models'

Searching...
Downloading model 'TheBloke/rocket-3B-GGUF' file 'rocket-3b.Q4_K_M.gguf' to '../../models/rocket-3b.Q4_K_M.gguf'
                                                                                                             
Download complete.
For information about this and other models, please visit https://huggingface.co
```

After this command, the "rocket-3b.Q4_K_M.gguf" file has now been downloaded to the "../../models" folder.

We'll now register it with the Models factory, which is located in the folder to where we downloaded the model.

This can be done by editing the "models.json" file directly or even simpler, with a "sibila models -s" command:

```
> sibila models -s llamacpp:rocket rocket-3b.Q4_K_M.gguf -m '../../models'

Using models directory '../../models'
Set model 'llamacpp:rocket' with name='rocket-3b.Q4_K_M.gguf' at '/home/jorge/ai/sibila/models/models.json'.
```

An entry has now been created in "models.json" for this model.

However, we did not set the chat template format - but let's first test if the downloaded GGUF file already includes it in its metadata.

This is done with "sibila models -t":

```
> sibila models -t llamacpp:rocket -m '../../models'

Using models directory '../../models'
Testing model 'llamacpp:rocket'...
Error: Could not find a suitable chat template format for this model. Without a format, fine-tuned models cannot function properly. See the docs on how you can fix this: either setup the format in Models factory, or provide the chat template in the 'format' arg.
```

Error. Looks like we need to set the chat template format!

Checking the [model's page](https://huggingface.co/TheBloke/rocket-3B-GGUF), we find that it uses the ChatML prompt/chat template, which is great because it's one of the base formats included with Sibila.

So let's set the template format in the "llamacpp:rocket" entry we've just created:

```
> sibila models -f llamacpp:rocket chatml -m '../../models'

Using models directory '/home/jorge/ai/sibila/models'
Updated model 'llamacpp:rocket' with format 'chatml' at '/home/jorge/ai/sibila/models/models.json'.
```

Let's now test again:

```
> sibila models -t llamacpp:rocket -m '../../models'

Using models directory '../../models'
Testing model 'llamacpp:rocket'...
Model 'llamacpp:rocket' was properly created and should run fine.
```

Great - the model passed the test and should be ready for use.

Let's try using it from Python:


```python
from sibila import Models

Models.setup("../../models") # the folder with models and configs

model = Models.create("llamacpp:rocket") # model name in provider:name format

model("Hello there!")
```




    "Hello! I'm an AI language model here to assist you with your inquiries or generate content for you. I am programmed to be polite and respectful, so please let me know how I can help you today."



Seems to be working - and politely too!

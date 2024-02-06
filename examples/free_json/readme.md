# Free JSON Output

In this example we'll see how to generate valid JSON that doesn't conform to a type schema, that is we'll not force the model to generate any named keys nor types.


``` py


# To use a local model, change the filename in the next line and uncomment it.
# name = "llamacpp:openchat-3.5-1210.Q4_K_M.gguf"
# Also uncomment the next line to add models/ folder and its config
# ModelDir.add("../models/modeldir.json")

# Or instead, if you want to use GPT-4, uncomment the next line and set env variable OPENAI_API_KEY with your token:
# name = "openai:gpt-4"

model = ModelDir.create(name)

```
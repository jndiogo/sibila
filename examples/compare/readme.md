In this example we'll use an utility function from the multigen module that generates a table of answers to a list of questions by multiple models.

This function generates a 2-D table of input x model: each row is the output from different models to one question or input. This can be very helpful to compare how two or more models react to the same input. Such table can be printed or saved as a CSV file.

Instead of directly creating models, we'll start by defining their names: for a local model and a remote model which we'll compare. The multigen function will create the models itself.


```python
from sibila import ModelDir

# to use a local model, assuming it's in ../../models/:
# add models folder config which also adds to ModelDir path
ModelDir.add("../../models/modeldir.json")
# set the model's filename - change to your own model
local_name = "llamacpp:openchat-3.5-1210.Q4_K_M.gguf"

# to use an OpenAI model:
remote_name = "openai:gpt-3.5"
```

Now let's define a list of reviews that we'll ask the two models to do sentiment analysis upon.

These are generic product reviews, that you could find in an online store.


```python
reviews = [
"The user manual was confusing, but once I figured it out, the product more or less worked.",
"This widget changed my life! It's sleek, efficient, and worth every penny.",
"I'm disappointed with the product quality. It broke after just a week of use.",
"The customer service team was incredibly helpful in resolving my issue with the device.",
"I'm blown away by the functionality of this gadget. It exceeded my expectations.",
"The packaging was damaged upon arrival, but the product itself works great.",
"I've been using this tool for months, and it's still as good as new. Highly recommended!",
"I regret purchasing this item. It doesn't perform as advertised.",
"I've never had so much trouble with a product before. It's been a headache from day one.",
"I bought this as a gift for my friend, and they absolutely love it!",
"The price seemed steep at first, but after using it, I understand why. Quality product.",
"This gizmo is a game-changer for my daily routine. Couldn't be happier with my purchase!"
]

inst_text = "You are a helpful assistant that analyses text sentiment."
```

Since we just want to obtain a sentiment classification, we'll use a quick dictype definition of a "sentiment" field with three values: positive, negative or neutral.

Let's try the first review on a local model:


```python

sentiment_type = {
    "sentiment": {"type": ["positive", "neutral", "negative"]}
}

in_text = "Each line is a product review. Extract the sentiment associated with each review:\n\n" + reviews[0]

print(reviews[0])

local_model = ModelDir.create(local_name)
out = local_model.query_dictype(sentiment_type,
                                inst_text,
                                in_text)
# to clear memory
del local_model

print(out)
```

    The user manual was confusing, but once I figured it out, the product more or less worked.
    {'sentiment': 'neutral'}


Definitely neutral is a good answer for this one. 

Let's now try the remote model:


```python
print(reviews[0])

remote_model = ModelDir.create(remote_name)

out = remote_model.query_dictype(sentiment_type,
                                 inst_text,
                                 in_text)
del remote_model

print(out)
```

    The user manual was confusing, but once I figured it out, the product more or less worked.
    {'sentiment': 'neutral'}


And the remote model (GPT-3.5) seems to agree.

By using the query_multigen() function that we'll import from sibila.multigen, we'll be able to compare what multiple models generate in response to each input.

In our case the inputs will be the list of reviews. This function accepts a few other interesting arguments:
- text: type of text output, which can be the word "print" or a text filename to which it will save.
- csv: type of CSV output, which can also be "print" or a text filename to save into.
- out_keys: what we want listed: the generated raw text, a JSON dict or a Pydantic object. For our case "dict" is the better one.
- gencall: we need to pass a function that will call the model for each input. We use a predefined function and provide it with the sentiment_type definition.

Let's run it with our two models:


```python
from sibila.multigen import (
    query_multigen,
    make_dictype_gencall
)

out = query_multigen(reviews,
                     inst_text,
                     model_names = [local_name, remote_name],
                     text="print",
                     csv="sentiment.csv",
                     out_keys = ["dict"],
                     gencall = make_dictype_gencall(sentiment_type)
                     )
```

    ////////////////////////////////////////////////////////////
    The user manual was confusing, but once I figured it out, the product more or less worked.
    ////////////////////////////////////////////////////////////
    ==================== llamacpp:openchat-3.5-1210.Q4_K_M.gguf -> OK_STOP
    {"sentiment":"neutral"}
    ==================== openai:gpt-3.5 -> OK_STOP
    {"sentiment":"neutral"}
    
    ////////////////////////////////////////////////////////////
    This widget changed my life! It's sleek, efficient, and worth every penny.
    ////////////////////////////////////////////////////////////
    ==================== llamacpp:openchat-3.5-1210.Q4_K_M.gguf -> OK_STOP
    {"sentiment":"positive"}
    ==================== openai:gpt-3.5 -> OK_STOP
    {"sentiment":"positive"}
    
    ////////////////////////////////////////////////////////////
    I'm disappointed with the product quality. It broke after just a week of use.
    ////////////////////////////////////////////////////////////
    ==================== llamacpp:openchat-3.5-1210.Q4_K_M.gguf -> OK_STOP
    {"sentiment":"negative"}
    ==================== openai:gpt-3.5 -> OK_STOP
    {"sentiment":"negative"}
    
    ////////////////////////////////////////////////////////////
    The customer service team was incredibly helpful in resolving my issue with the device.
    ////////////////////////////////////////////////////////////
    ==================== llamacpp:openchat-3.5-1210.Q4_K_M.gguf -> OK_STOP
    {"sentiment":"positive"}
    ==================== openai:gpt-3.5 -> OK_STOP
    {"sentiment":"positive"}
    
    ////////////////////////////////////////////////////////////
    I'm blown away by the functionality of this gadget. It exceeded my expectations.
    ////////////////////////////////////////////////////////////
    ==================== llamacpp:openchat-3.5-1210.Q4_K_M.gguf -> OK_STOP
    {"sentiment":"positive"}
    ==================== openai:gpt-3.5 -> OK_STOP
    {"sentiment":"positive"}
    
    ////////////////////////////////////////////////////////////
    The packaging was damaged upon arrival, but the product itself works great.
    ////////////////////////////////////////////////////////////
    ==================== llamacpp:openchat-3.5-1210.Q4_K_M.gguf -> OK_STOP
    {"sentiment":"neutral"}
    ==================== openai:gpt-3.5 -> OK_STOP
    {"sentiment":"positive"}
    
    ////////////////////////////////////////////////////////////
    I've been using this tool for months, and it's still as good as new. Highly recommended!
    ////////////////////////////////////////////////////////////
    ==================== llamacpp:openchat-3.5-1210.Q4_K_M.gguf -> OK_STOP
    {"sentiment":"positive"}
    ==================== openai:gpt-3.5 -> OK_STOP
    {"sentiment":"positive"}
    
    ////////////////////////////////////////////////////////////
    I regret purchasing this item. It doesn't perform as advertised.
    ////////////////////////////////////////////////////////////
    ==================== llamacpp:openchat-3.5-1210.Q4_K_M.gguf -> OK_STOP
    {"sentiment":"negative"}
    ==================== openai:gpt-3.5 -> OK_STOP
    {"sentiment":"negative"}
    
    ////////////////////////////////////////////////////////////
    I've never had so much trouble with a product before. It's been a headache from day one.
    ////////////////////////////////////////////////////////////
    ==================== llamacpp:openchat-3.5-1210.Q4_K_M.gguf -> OK_STOP
    {"sentiment":"negative"}
    ==================== openai:gpt-3.5 -> OK_STOP
    {"sentiment":"negative"}
    
    ////////////////////////////////////////////////////////////
    I bought this as a gift for my friend, and they absolutely love it!
    ////////////////////////////////////////////////////////////
    ==================== llamacpp:openchat-3.5-1210.Q4_K_M.gguf -> OK_STOP
    {"sentiment":"positive"}
    ==================== openai:gpt-3.5 -> OK_STOP
    {"sentiment":"positive"}
    
    ////////////////////////////////////////////////////////////
    The price seemed steep at first, but after using it, I understand why. Quality product.
    ////////////////////////////////////////////////////////////
    ==================== llamacpp:openchat-3.5-1210.Q4_K_M.gguf -> OK_STOP
    {"sentiment":"positive"}
    ==================== openai:gpt-3.5 -> OK_STOP
    {"sentiment":"positive"}
    
    ////////////////////////////////////////////////////////////
    This gizmo is a game-changer for my daily routine. Couldn't be happier with my purchase!
    ////////////////////////////////////////////////////////////
    ==================== llamacpp:openchat-3.5-1210.Q4_K_M.gguf -> OK_STOP
    {"sentiment":"positive"}
    ==================== openai:gpt-3.5 -> OK_STOP
    {"sentiment":"positive"}
    
    


The output format is - see comments nearby -----> arrows:

```
//////////////////////////////////////////////////////////// -----> This is the modle input, a review:
This gizmo is a game-changer for my daily routine. Couldn't be happier with my purchase!
////////////////////////////////////////////////////////////
==================== llamacpp:openchat-3.5-1210.Q4_K_M.gguf -> OK_STOP  <----- Model name and query result for our local model
{"sentiment":"positive"}  <----- What the local model output
==================== openai:gpt-3.5 -> OK_STOP  <----- Model name and result for remote model
{"sentiment":"positive"}  <----- Remote model output
```

We also requested the creation of a CSV file which is named [sentiment.csv](sentiment.csv).

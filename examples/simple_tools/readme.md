In this example we'll look at a simple way to have the model choose among tools or give a straight answer, using structured data extraction. This can be advantageous to keep tool usage independent of model provider.

We'll use a Llama-3 8B model. Please make sure you have its file in the folder "../../models". You can use any GGUF format model, don't forget to set its filename in the name variable below, after the text "llamacpp:". Or you could likewise use any other remote model from any provider.

Jupyter notebook and Python script versions are available in the example's folder.

Let's create the model:


```python
from sibila import Models

# load env variables like OPENAI_API_KEY from a .env file (if available)
try: from dotenv import load_dotenv; load_dotenv()
except: ...

# delete any live model
try: model.close(); del model
except: pass

Models.setup("../../models")
name = "llamacpp:Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
# name = "openai:gpt-4o"

model = Models.create(name)
```

We'll use an enumerator to choose which tool to use and a generic field for the arguments of the tool call.

A special value of NO_TOOL signals that the model is giving a straight answer and no tool is being used.

It's important to explain what each tool does in the instructions text, so the model "knows" what to do.


```python
from typing import Literal

from pydantic import (
    BaseModel,
    Field
)

# which tool to use?
AnswerType = Literal["NO_TOOL", "WEB_SEARCH", "CALCULATOR", "NEW_NOTE"]

class AnswerOrTool(BaseModel):
    answer_type: AnswerType
    argument: str

inst = """\
If user requests live information, answer_type should be WEB_SEARCH and the argument field should be the query.
If the user requests a calculation, don't do the math, instead answer_type should be CALCULATOR and the argument field should be the math expression.
If the user asks to create a new note, answer_type should be NEW_NOTE and the argument field should be note's subject.
Otherwise, answer_type should be "NO_TOOL" and the answer should be given in the argument field.
"""
```

Let's now try a few queries to see how the model behaves:


```python
queries = [
    "Can you write a simple poem?",
    "What's the current NVIDIA stock market value?",
    "How much is 78*891?",
    "Create a new note to call Manuel to invite him to come over and visit me",
]

for q in queries:
    res = model.extract(AnswerOrTool, q, inst=inst)
    print(q)
    print(res)
    print()
```

    Can you write a simple poem?
    answer_type='NO_TOOL' argument='Here is a simple poem, with words so sweet,\nA gentle breeze that whispers at your feet.\nThe sun shines bright, the birds sing their song,\nAnd all around, life is strong.'
    
    What's the current NVIDIA stock market value?
    answer_type='WEB_SEARCH' argument='NVIDIA stock market value'
    
    How much is 78*891?
    answer_type='CALCULATOR' argument='78*891'
    
    Create a new note to call Manuel to invite him to come over and visit me
    answer_type='NEW_NOTE' argument='Call Manuel to invite him to come over and visit me'
    


Some of these tools should have their output feed back into the model for further queries. In this case we'd be better off using a message Thread, where the next IN message would contain the tool output and a query for the results we're looking for.

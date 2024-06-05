In this example we'll look at extracting structured information from a photo of a receipt.

Sibila supports local models with image input support, but we'll use OpenAI's GPT-4o which works quite well. Make sure to set your OPENAI_API_KEY env variable.

You can still use a local model by uncommenting the commented lines below. See the docs for some suggestions about local vision models.

Jupyter notebook and Python script versions are available in the example's folder.

Let's create the model:


```python
# load env variables like OPENAI_API_KEY from a .env file (if available)
try: from dotenv import load_dotenv; load_dotenv()
except: ...

from sibila import Models

# delete any previous model
try: del model
except: ...

# to use a local model, assuming it's in ../../models:
# setup models folder:
# Models.setup("../../models")
# model = Models.create("llamacpp:llava-llama-3-8b-v1_1-int4.gguf*llava-llama-3-8b-v1_1-mmproj-f16.gguf")

# to use an OpenAI model:
model = Models.create("openai:gpt-4o")
```

Let's use this photo of an [Italian receipt](https://commons.wikimedia.org/wiki/File:Receipts_in_Italy_13.jpg):

![Receipt](https://upload.wikimedia.org/wikipedia/commons/6/6a/Receipts_in_Italy_13.jpg)

To see if the model can handle it, let's try a free text query for the total. We'll pass a tuple of (text_prompt, image_url) -


```python
model(("How much is the total?", 
       "https://upload.wikimedia.org/wikipedia/commons/6/6a/Receipts_in_Italy_13.jpg"))
```




    'The total amount on the receipt is €5.88.'



Good. Can the model extract the receipt item lines?


```python
model(("List the lines of paid items in the receipt?", 
       "https://upload.wikimedia.org/wikipedia/commons/6/6a/Receipts_in_Italy_13.jpg"))
```




    'The lines of paid items in the receipt are:\n\n1. BIS BORSE TERM. S - €3.90\n2. GHIACCIO 2X400 G - €0.99\n3. GHIACCIO 2X400 G - €0.99'



It did extract them well. 

Let's wrap this in a Pydantic object to get structured data from the model. We'll add a field for the data listed in the receipt:


```python
from pydantic import BaseModel, Field
from datetime import datetime

class ReceiptLine(BaseModel):
    """Receipt line data"""
    description: str
    cost: float

class Receipt(BaseModel):
    """Receipt information"""
    total: float = Field(description="Total value")
    lines: list[ReceiptLine] = Field(description="List of lines of paid items")
    date: datetime = Field(description="Listed date")

info = model.extract(Receipt,
                     ("Extract receipt information.", 
                      "https://upload.wikimedia.org/wikipedia/commons/6/6a/Receipts_in_Italy_13.jpg"))
info
```




    Receipt(total=5.88, lines=[ReceiptLine(description='BIS BORSE TERM.S', cost=3.9), ReceiptLine(description='GHIACCIO 2X400 G', cost=0.99), ReceiptLine(description='GHIACCIO 2X400 G', cost=0.99)], date=datetime.datetime(2014, 8, 27, 19, 51, tzinfo=TzInfo(UTC)))




```python
for line in info.lines:
    print(line)
print("total:", info.total)
```

    description='BIS BORSE TERM.S' cost=3.9
    description='GHIACCIO 2X400 G' cost=0.99
    description='GHIACCIO 2X400 G' cost=0.99
    total: 5.88


All the information is correct and structured in an object that we can use as needed.

From here we could expand the Pydantic object with more fields to extract other information present in the receipt like merchant name, VAT number, etc.

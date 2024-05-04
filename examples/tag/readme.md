In this example we'll summarize and classify customer queries with tags. We'll use dataclasses to specify the structure of the information we want extracted (we could also use Pydantic BaseModel classes).

To use a local model, make sure you have its file in the folder "../../models". You can use any GGUF format model - [see here how to download the OpenChat model used below](https://jndiogo.github.io/sibila/models/local_model/#examples). If you use a different one, don't forget to set its filename in the name variable below, after the text "llamacpp:".

To use an OpenAI model, make sure you defined the env variable OPENAI_API_KEY with a valid token and uncomment the line after "# to use an OpenAI model:".

Available as a Jupyter notebook or a Python script in the example's folder.

Let's start by creating the model:


```python
from sibila import Models

# delete any previous model
try: del model
except: ...

# to use a local model, assuming it's in ../../models:
# setup models folder:
Models.setup("../../models")
# set the model's filename - change to your own model
model = Models.create("llamacpp:openchat-3.5-1210.Q4_K_M.gguf")

# to use an OpenAI model:
# model = Models.create("openai:gpt-4")
```

These will be our queries, ten typical customer support questions:


```python
queries = """\
1. Do you offer a trial period for your software before purchasing?
2. I'm experiencing a glitch with your app, it keeps freezing after the latest update.
3. What are the different pricing plans available for your subscription service?
4. Can you provide instructions on how to reset my account password?
5. I'm unsure about the compatibility of your product with my device, can you advise?
6. How can I track my recent order and estimate its delivery date?
7. Is there a customer loyalty program or rewards system for frequent buyers?
8. I'm interested in your online courses, but do you offer refunds if I'm not satisfied?
9. Could you clarify the coverage and limitations of your product warranty?
10. What are your customer support hours and how can I reach your team in case of emergencies?
"""
```

We'll start by summarizing each query. 

Let's try just using field names (without descriptions), perhaps they are enough to tell the model about what we want.


```python
from dataclasses import dataclass

@dataclass        
class Query():
    id: int
    query_summary: str
    query_text: str

# model instructions text, also known as system message
inst_text = "Extract information from customer queries."

# the input query, including the above text
in_text = "Each line is a customer query. Extract information about each query:\n\n" + queries

out = model.extract(list[Query],
                    in_text,
                    inst=inst_text)

for query in out:
    print(query)
```

    Query(id=1, query_summary='Trial period inquiry', query_text='Do you offer a trial period for your software before purchasing?')
    Query(id=2, query_summary='Technical issue', query_text="I'm experiencing a glitch with your app, it keeps freezing after the latest update.")
    Query(id=3, query_summary='Pricing inquiry', query_text='What are the different pricing plans available for your subscription service?')
    Query(id=4, query_summary='Password reset request', query_text='Can you provide instructions on how to reset my account password?')
    Query(id=5, query_summary='Compatibility inquiry', query_text="I'm unsure about the compatibility of your product with my device, can you advise?")
    Query(id=6, query_summary='Order tracking', query_text='How can I track my recent order and estimate its delivery date?')
    Query(id=7, query_summary='Loyalty program inquiry', query_text='Is there a customer loyalty program or rewards system for frequent buyers?')
    Query(id=8, query_summary='Refund policy inquiry', query_text="I'm interested in your online courses, but do you offer refunds if I'm not satisfied?")
    Query(id=9, query_summary='Warranty inquiry', query_text='Could you clarify the coverage and limitations of your product warranty?')
    Query(id=10, query_summary='Customer support inquiry', query_text='What are your customer support hours and how can I reach your team in case of emergencies?')


The summaries look good.

Let's now define tags and ask the model to classify each query into a tag. In the Tag class, we set its docstring to the rules we want for the classification. This is done in the docstring because Tag is not a dataclass, but derived from Enum.

No longer asking for the query_text in the Query class to keep output shorter.


```python
from enum import Enum

class Tag(str, Enum):
    """Queries can be classified into the following tags:
tech_support: queries related with technical problems.
billing: post-sale queries about billing cycle, or subscription termination.
account: queries about user account problems.
pre_sales: queries from prospective customers (who have not yet purchased).
other: all other query topics."""        
    TECH_SUPPORT = "tech_support"
    BILLING = "billing"
    PRE_SALES = "pre_sales"
    ACCOUNT = "account"
    OTHER = "other"

@dataclass        
class Query():
    id: int
    query_summary: str
    query_tag: Tag

out = model.extract(list[Query],
                    in_text,
                    inst=inst_text)

for query in out:
    print(query)
```

    Query(id=1, query_summary='Asking about trial period', query_tag='pre_sales')
    Query(id=2, query_summary='Reporting app issue', query_tag='tech_support')
    Query(id=3, query_summary='Inquiring about pricing plans', query_tag='billing')
    Query(id=4, query_summary='Requesting password reset instructions', query_tag='account')
    Query(id=5, query_summary='Seeking device compatibility advice', query_tag='pre_sales')
    Query(id=6, query_summary='Tracking order and delivery date', query_tag='other')
    Query(id=7, query_summary='Inquiring about loyalty program', query_tag='billing')
    Query(id=8, query_summary='Asking about refund policy', query_tag='pre_sales')
    Query(id=9, query_summary='Seeking warranty information', query_tag='other')
    Query(id=10, query_summary='Inquiring about customer support hours', query_tag='other')


The applied tags appear mostly reasonable. 

Of course, pre-sales tagging could be done automatically from a database of existing customer contacts, but the model is doing a good job of identifying questions likely to be pre-sales, like ids 1, 5 and 8 which are questions typically asked before buying/subscribing.

Also, note that classification is being done from a single phrase. More information in each customer query would certainly allow for fine-grained classification.

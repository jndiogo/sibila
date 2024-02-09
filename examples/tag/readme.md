In this example we'll summarize and classify customer queries with tags.

To use a local model, make sure you have its file in the folder "../../models/". You can use any GGUF format model - [see here how to download the OpenChat model used below](../setup_local_models/readme.md#setup-local-models). If you use a different one, don't forget to set its filename in the name variable below, after the text "llamacpp:".

To use an OpenAI model, make sure you defined the env variable OPENAI_API_KEY with a valid token and uncomment the line after "# to use an OpenAI model:".

Let's start by creating the model:


```python
from sibila import ModelDir

# delete any previous model
try: del model
except: ...

# to use a local model, assuming it's in ../../models/:
# add models folder config which also adds to ModelDir path
ModelDir.add("../../models/modeldir.json")
# set the model's filename - change to your own model
name = "llamacpp:openchat-3.5-1210.Q4_K_M.gguf"
model = ModelDir.create(name)

# to use an OpenAI model:
# model = ModelDir.create("openai:gpt-4")
```

These will be our queries, ten typical customer support questions:


```python
queries = """\
1. Do you offer a trial period for your software before purchasing?
2. I'm experiencing a glitch with your app, it keeps freezing after the latest update.
3. What are the different pricing plans available for your subscription service?"
4. Can you provide instructions on how to reset my account password?"
5. I'm unsure about the compatibility of your product with my device, can you advise?"
6. How can I track my recent order and estimate its delivery date?"
7. Is there a customer loyalty program or rewards system for frequent buyers?"
8. I'm interested in your online courses, but do you offer refunds if I'm not satisfied?"
9. Could you clarify the coverage and limitations of your product warranty?"
10. What are your customer support hours and how can I reach your team in case of emergencies?
"""
```

We'll start by summarizing each query. No need adding field descriptions, the field names should be enough to tell the model about what we want done.


```python
from pydantic import BaseModel, Field
from typing import List
from enum import Enum

class Query(BaseModel):
    id: int
    query_summary: str
    query_text: str
    
class QueryTags(BaseModel):
    queries: List[Query]

inst_text = "Extract information from customer queries."
in_text = "Each line is a customer query. Extract information about each query:\n\n" + queries

out = model.query_pydantic(QueryTags,
                           inst_text,
                           in_text)
for query in out.queries:
    print(query)
```

    id=1 query_summary='Trial period inquiry' query_text='Do you offer a trial period for your software before purchasing?'
    id=2 query_summary='Technical issue' query_text="I'm experiencing a glitch with your app, it keeps freezing after the latest update."
    id=3 query_summary='Pricing inquiry' query_text='What are the different pricing plans available for your subscription service?'
    id=4 query_summary='Password reset request' query_text='Can you provide instructions on how to reset my account password?'
    id=5 query_summary='Compatibility inquiry' query_text="I'm unsure about the compatibility of your product with my device, can you advise?"
    id=6 query_summary='Order tracking' query_text='How can I track my recent order and estimate its delivery date?'
    id=7 query_summary='Loyalty program inquiry' query_text='Is there a customer loyalty program or rewards system for frequent buyers?'
    id=8 query_summary='Refund policy inquiry' query_text="I'm interested in your online courses, but do you offer refunds if I'm not satisfied?"
    id=9 query_summary='Warranty inquiry' query_text='Could you clarify the coverage and limitations of your product warranty?'
    id=10 query_summary='Customer support hours' query_text='What are your customer support hours and how can I reach your team in case of emergencies?'


Summaries appear to be quite good.

Let's now define tags and ask the model to classify each query into a tag. In the Tag class, we set its docstring to the rules we want for the classification. This is done in the docstring because the class is not derived from BaseModel, so we cannot set a Field(description="...") for each item.

No longer asking for the query_text in the Query class to keep output shorter.


```python
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
    
class Query(BaseModel):
    id: int
    query_summary: str
    query_tag: Tag
    
class QueryTags(BaseModel):
    queries: List[Query]

out = model.query_pydantic(QueryTags,
                           inst_text,
                           in_text)
for query in out.queries:
    print(query)
```

    id=1 query_summary='Asking about trial period' query_tag=<Tag.PRE_SALES: 'pre_sales'>
    id=2 query_summary='Reporting app glitch after update' query_tag=<Tag.TECH_SUPPORT: 'tech_support'>
    id=3 query_summary='Inquiring about pricing plans' query_tag=<Tag.PRE_SALES: 'pre_sales'>
    id=4 query_summary='Requesting password reset instructions' query_tag=<Tag.ACCOUNT: 'account'>
    id=5 query_summary='Asking about device compatibility' query_tag=<Tag.PRE_SALES: 'pre_sales'>
    id=6 query_summary='Tracking recent order and delivery date' query_tag=<Tag.OTHER: 'other'>
    id=7 query_summary='Inquiring about customer loyalty program' query_tag=<Tag.BILLING: 'billing'>
    id=8 query_summary='Asking about refund policy for online courses' query_tag=<Tag.PRE_SALES: 'pre_sales'>
    id=9 query_summary='Seeking warranty coverage and limitations' query_tag=<Tag.OTHER: 'other'>
    id=10 query_summary='Inquiring about customer support hours' query_tag=<Tag.OTHER: 'other'>


The applied tags appear mostly reasonable. 

Of course, pre-sales tagging could be done automatically from a database of existing customer contacts, but the model is doing a good job of identifying questions likely to be pre-sales, like ids 1, 3, 5 and 8 which are questions typically asked before buying/subscribing.

Also, classification is being done from a single phrase. More text in each customer query could allow for fine grained classification.

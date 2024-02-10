# load env variables from a .env if available:
env_path = "../../.env"
import os
if os.path.isfile(env_path):
    from dotenv import load_dotenv
    assert load_dotenv(env_path, override=True, verbose=True)


from pydantic import BaseModel, Field
from typing import List
from enum import Enum

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


# model instructions text, also known as system message
inst_text = "Extract information from customer queries."

in_text = "Each line is a customer query. Extract information about each query:\n\n" + queries


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

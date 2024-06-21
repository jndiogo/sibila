if __name__ == "__main__":
    
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

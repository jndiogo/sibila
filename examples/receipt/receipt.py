if __name__ == "__main__":

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
    
    for line in info.lines:
        print(line)
    print("total:", info.total)

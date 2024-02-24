# load env variables like OPENAI_API_KEY from a .env file (if available)
try: from dotenv import load_dotenv; load_dotenv()
except: ...

if __name__ == "__main__":

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
    # model = ModelDir.create("openai:gpt-4")


    transcript = """\
Date: 10th April 2024
Time: 10:30 AM
Location: Conference Room A

Attendees:
    Arthur: Logistics Supervisor
    Bianca: Operations Manager
    Chris: Fleet Coordinator

Arthur: Good morning, team. Thanks for making it. We've got three matters to address quickly today.

Bianca: Morning, Arthur. Let's dive in.

Chris: Ready when you are.

Arthur: First off, we've been having complaints about late deliveries. This is very important, we're getting some bad reputation out there.

Bianca: Chris, I think you're the right person to take care of this. Can you investigate and report back by end of day? 

Chris: Absolutely, Bianca. I'll look into the reasons and propose solutions.

Arthur: Great. Second, Bianca, we need to update our driver training manual. Can you take the lead and have a draft by Friday?

Bianca: Sure thing, Arthur. I'll get started on that right away.

Arthur: Lastly, we need to schedule a meeting with our software vendor to discuss updates to our tracking system. This is a low-priority task but still important. I'll handle that. Any input on timing?

Bianca: How about next Wednesday afternoon?

Chris: Works for me.

Arthur: Sounds good. I'll arrange it. Thanks, Bianca, Chris. Let's keep the momentum going.

Bianca: Absolutely, Arthur.

Chris: Will do.
"""


    from pydantic import BaseModel, Field
    from enum import Enum

    class Attendee(BaseModel):
        name: str
        occupation: str

    class Priority(str, Enum):
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"
        
    class ActionItem(BaseModel):
        index: int = Field(description="Sequential index for the action item")
        name: str = Field(description="Action item name")
        priority: Priority = Field(description="Action item priority")
        due_by: str = Field(description="When should the item be complete")
        assigned_attendee: str = Field(description="Name of the attendee to which action item was assigned")

    class Meeting(BaseModel):
        meeting_date: str
        meeting_location: str
        attendees: list[Attendee]
        action_items: list[ActionItem]


    # model instructions text, also known as system message
    inst_text = "Extract information."

    # the input query, including the above transcript
    in_text = "Extract information from this meeting transcript:\n\n" + transcript

    out = model.extract(Meeting,
                        in_text,
                        inst=inst_text)

    print("Meeting:", out.meeting_date, "in", out.meeting_location)
    print("Attendees:")
    for att in out.attendees:
        print(att)
    print("Action items:")    
    for items in out.action_items:
        print(items)

Let's extract structured information from a meeting transcript, like attendees, action items and their priorities.

This is a quick meeting whose transcript is not very large, so a small local model should work well. See the [Tough meeting example](../tough_meeting/readme.md) for a larger and more complex transcription text.

To use a local model, make sure you have its file in the folder "../../models/". You can use any GGUF format model - [see here how to download the OpenChat model used below](https://jndiogo.github.io/sibila/setup-local-models/#default-model-used-in-the-examples-openchat). If you use a different one, don't forget to set its filename in the name variable below, after the text "llamacpp:".

If you prefer to use an OpenAI model, make sure you defined the env variable OPENAI_API_KEY with a valid token and uncomment the line after "# to use an OpenAI model:".

Available as a [Jupyter notebook](quick_meeting.ipynb) or [Python script](quick_meeting.py).

Let's create the model:


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

Here's the transcript:


```python
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

# model instructions text, also known as system message
inst_text = "Extract information."
```

Let's define two classes whose instances will receive the extracted information:
- Attendee: to store information about each meeting attendee
- Meeting: to keep meeting's date and location, list of participants and other info we'll see below

And let's ask the model to create objects that are instances of these classes:


```python
from pydantic import BaseModel, Field
from typing import List
from enum import Enum

# class definitions will be used to constrain the model output and initialize an instance object
class Attendee(BaseModel):
    name: str
    occupation: str

class Meeting(BaseModel):
    meeting_date: str
    meeting_location: str
    attendees: List[Attendee]

in_text = "Extract information from this meeting transcript:\n\n" + transcript

out = model.query_pydantic(Meeting,
                           inst_text,
                           in_text)
print(out)
```

    meeting_date='10th April 2024' meeting_location='Conference Room A' attendees=[Attendee(name='Arthur', occupation='Logistics Supervisor'), Attendee(name='Bianca', occupation='Operations Manager'), Attendee(name='Chris', occupation='Fleet Coordinator')]


A prettier display:


```python
print("Meeting:", out.meeting_date, "in", out.meeting_location)
print("Attendees:")
for att in out.attendees:
    print(att)
```

    Meeting: 10th April 2024 in Conference Room A
    Attendees:
    name='Arthur' occupation='Logistics Supervisor'
    name='Bianca' occupation='Operations Manager'
    name='Chris' occupation='Fleet Coordinator'


This information was correctly extracted.

Let's now request the action items mentioned in the meeting. Well create a new class ActionItem with an index and a name for the item.

We'll also add an action_items field to the Meeting class to hold the items list.


```python
class Attendee(BaseModel):
    name: str
    occupation: str

class ActionItem(BaseModel):
    index: int = Field(description="Sequential index for the action item")
    name: str = Field(description="Action item name")

class Meeting(BaseModel):
    meeting_date: str
    meeting_location: str
    attendees: List[Attendee]
    action_items: List[ActionItem]

out = model.query_pydantic(Meeting,
                           inst_text,
                           in_text)

print("Meeting:", out.meeting_date, "in", out.meeting_location)
print("Attendees:")
for att in out.attendees:
    print(att)
print("Action items:")    
for items in out.action_items:
    print(items)
```

    Meeting: 10th April 2024 in Conference Room A
    Attendees:
    name='Arthur' occupation='Logistics Supervisor'
    name='Bianca' occupation='Operations Manager'
    name='Chris' occupation='Fleet Coordinator'
    Action items:
    index=1 name='Investigate and report on late deliveries by end of day'
    index=2 name='Update driver training manual by Friday'
    index=3 name='Schedule a meeting with software vendor to discuss tracking system updates'


The extracted action items also look good.

Let's now extract more action item information:
- Priority for each item
- Due by... information
- Name of the attendee that was assigned for that item 

So, we create a Priority class holding three priority types - low to high. 

We also add three fields to the ActionItem class, to hold the new information: priority, due_by and assigned_attendee.


```python
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
    attendees: List[Attendee]
    action_items: List[ActionItem]

out = model.query_pydantic(Meeting,
                           inst_text,
                           in_text)

print("Meeting:", out.meeting_date, "in", out.meeting_location)
print("Attendees:")
for att in out.attendees:
    print(att)
print("Action items:")    
for items in out.action_items:
    print(items)
```

    Meeting: 10th April 2024 in Conference Room A
    Attendees:
    name='Arthur' occupation='Logistics Supervisor'
    name='Bianca' occupation='Operations Manager'
    name='Chris' occupation='Fleet Coordinator'
    Action items:
    index=1 name='Investigate late deliveries' priority=<Priority.HIGH: 'high'> due_by='end of day' assigned_attendee='Chris'
    index=2 name='Update driver training manual' priority=<Priority.MEDIUM: 'medium'> due_by='Friday' assigned_attendee='Bianca'
    index=3 name='Schedule meeting with software vendor' priority=<Priority.LOW: 'low'> due_by='next Wednesday afternoon' assigned_attendee='Arthur'


The new information was correctly extracted: priorities, due by and assigned attendees for each action item.

For an example of a harder, more complex transcript see the [Tough meeting example](../tough_meeting/readme.md).

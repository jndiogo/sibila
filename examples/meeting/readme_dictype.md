This is a version of the [meeting transcription example](readme.md) but using dictypes instead of Pydantic BaseClass objects.

Learn more about [dictype definitions here](https://jndiogo.github.io/sibila/api-reference/#dictype).

So let's look at extracting participants and action items from a meeting transcript.


Start by creating the model. As you'll see below, the transcript is large, with complex language, so we'll use OpenAI's GPT-4 this time. You can still use a local model by uncommenting the commented lines below.


```python
from sibila import ModelDir, GenConf

# delete any previous model
try: del model
except: ...

# to use a local model, assuming it's in ../../models/:
# add models folder config which also adds to ModelDir path
# ModelDir.add("../../models/modeldir.json")
# set the model's filename - change to your own model
# name = "llamacpp:openchat-3.5-1210.Q4_K_M.gguf"

# the transcript is large, so we'll create the model with a context length of 3072, which should be enough.
#model = ModelDir.create(name, ctx_len=3072)

# to use an OpenAI model:
model = ModelDir.create("openai:gpt-4", ctx_len=3072)
```

We'll use a sample meeting transcript from https://www.ctas.tennessee.edu/eli/sample-meeting-transcript


```python
transcript = """\
Chairman Wormsley (at the proper time and place, after taking the chair and striking the gavel on the table): This meeting of the CTAS County Commission will come to order. Clerk please call the role. (Ensure that a majority of the members are present.)

Chairman Wormsley: Each of you has received the agenda. I will entertain a motion that the agenda be approved.

Commissioner Brown: So moved.

Commissioner Hobbs: Seconded

Chairman Wormsley: It has been moved and seconded that the agenda be approved as received by the members. All those in favor signify by saying "Aye"?...Opposed by saying "No"?...The agenda is approved. You have received a copy of the minutes of the last meeting. Are there any corrections or additions to the meeting?

Commissioner McCroskey: Mister Chairman, my name has been omitted from the Special Committee on Indigent Care.

Chairman Wormsley: Thank you. If there are no objections, the minutes will be corrected to include the name of Commissioner McCroskey. Will the clerk please make this correction. Any further corrections? Seeing none, without objection the minutes will stand approved as read. (This is sort of a short cut way that is commonly used for approval of minutes and/or the agenda rather than requiring a motion and second.)

Chairman Wormsley: Commissioner Adkins, the first item on the agenda is yours.

Commissioner Adkins: Mister Chairman, I would like to make a motion to approve the resolution taking money from the Data Processing Reserve Account in the County Clerk's office and moving it to the equipment line to purchase a laptop computer.

Commissioner Carmical: I second the motion.

Chairman Wormsley: This resolution has a motion and second. Will the clerk please take the vote.

Chairman Wormsley: The resolution passes. We will now take up old business. At our last meeting, Commissioner McKee, your motion to sell property near the airport was deferred to this meeting. You are recognized.

Commissioner McKee: I move to withdraw that motion.

Chairman Wormsley: Commissioner McKee has moved to withdraw his motion to sell property near the airport. Seeing no objection, this motion is withdrawn. The next item on the agenda is Commissioner Rodgers'.

Commissioner Rodgers: I move adopton of the resolution previously provided to each of you to increase the state match local litigation tax in circuit, chancery, and criminal courts to the maximum amounts permissible. This resolution calls for the increases to go to the general fund.

Chairman Wormsley: Commissioner Duckett

Commissioner Duckett: The sheriff is opposed to this increase.

Chairman Wormsley: Commissioner, you are out of order because this motion has not been seconded as needed before the floor is open for discussion or debate. Discussion will begin after we have a second. Is there a second?

Commissioner Reinhart: For purposes of discussion, I second the motion.

Chairman Wormsley: Commissioner Rodgers is recognized.

Commissioner Rodgers: (Speaks about the data on collections, handing out all sorts of numerical figures regarding the litigation tax, and the county's need for additional revenue.)

Chairman Wormsley: Commissioner Duckett

Commissioner Duckett: I move an amendment to the motion to require 25 percent of the proceeds from the increase in the tax on criminal cases go to fund the sheriff's department.

Chairman Wormsley: Commissioner Malone

Commissioner Malone: I second the amendment.

Chairman Wormsley: A motion has been made and seconded to amend the motion to increase the state match local litigation taxes to the maximum amounts to require 25 percent of the proceeds from the increase in the tax on criminal cases in courts of record going to fund the sheriff's department. Any discussion? Will all those in favor please raise your hand? All those opposed please raise your hand. The amendment carries 17-2. We are now on the motion as amended. Any further discussion?

Commissioner Headrick: Does this require a two-thirds vote?

Chairman Wormsley: Will the county attorney answer that question?

County Attorney Fults: Since these are only courts of record, a majority vote will pass it. The two-thirds requirement is for the general sessions taxes.

Chairman Wormsley: Other questions or discussion? Commissioner Adams.

Commissioner Adams: Move for a roll call vote.

Commissioner Crenshaw: Second

Chairman Wormsley: The motion has been made and seconded that the state match local litigation taxes be increased to the maximum amounts allowed by law with 25 percent of the proceeds from the increase in the tax on criminal cases in courts of record going to fund the sheriff's department. Will all those in favor please vote as the clerk calls your name, those in favor vote "aye," those against vote "no." Nine votes for, nine votes against, one not voting. The increase fails. We are now on new business. Commissioner Adkins, the first item on the agenda is yours.

Commissioner Adkins: Each of you has previously received a copy of a resolution to increase the wheel tax by $10 to make up the state cut in education funding. I move adoption of this resolution.

Chairman Wormsley: Commissioner Thompson

Commissioner Thompson: I second.

Chairman Wormsley: It has been properly moved and seconded that a resolution increasing the wheel tax by $10 to make up the state cut in education funding be passed. Any discussion? (At this point numerous county commissioners speak for and against increasing the wheel tax and making up the education cuts. This is the first time this resolution is under consideration.) Commissioner Hayes is recognized.

Commissioner Hayes: I move previous question.

Commisioner Crenshaw: Second.

Chairman Wormsley: Previous question has been moved and seconded. As you know, a motion for previous question, if passed by a two-thirds vote, will cut off further debate and require us to vote yes or no on the resolution before us. You should vote for this motion if you wish to cut off further debate of the wheel tax increase at this point. Will all those in favor of previous question please raise your hand? Will all those against please raise your hand? The vote is 17-2. Previous question passes. We are now on the motion to increase the wheel tax by $10 to make up the state cut in education funding. Will all those in favor please raise your hand? Will all those against please raise your hand? The vote is 17-2. This increase passes on first passage. Is there any other new business? Since no member is seeking recognition, are there announcements? Commissioner Hailey.

Commissioner Hailey: There will be a meeting of the Budget Committee to look at solid waste funding recommendations on Tuesday, July 16 at noon here in this room.

Chairman Wormsley: Any other announcements? The next meeting of this body will be Monday, August 19 at 7 p.m., here in this room. Commissioner Carmical.

Commissioner Carmical: There will be a chili supper at County Elementary School on August 16 at 6:30 p.m. Everyone is invited.

Chairman Wormsley: Commissioner Austin.

Commissioner Austin: Move adjournment.

Commissioner Garland: Second.

Chairman Wormsley: Without objection, the meeting will stand adjourned.
"""

# this is the text with the model instructions, also known as system message.
inst_text = "Extract information and output in JSON format."
```

As you can see, this is a quite large transcript, filled with long names and complex phrases. Let's see how the model will handle it...

Let's start by extracting the names of the participants in the meeting.

We'll create meeting_type, with the participants field having a list of strings; and add a description of what we're looking to receive.

Instead of calling query_pydantic(), we call query_dictype().


```python
# dictype definition describing a list of strings in the field participants
meeting_type = {
    "participants": {"type": [str], "desc": "List of complete names of meeting participants"}
}

in_text = "Extract information from this meeting transcript:\n\n" + transcript

out = model.query_dictype(meeting_type,
                          inst_text,
                          in_text)
print(out)
```

    {'participants': ['Chairman Wormsley', 'Commissioner Brown', 'Commissioner Hobbs', 'Commissioner McCroskey', 'Commissioner Adkins', 'Commissioner Carmical', 'Commissioner McKee', 'Commissioner Rodgers', 'Commissioner Duckett', 'Commissioner Reinhart', 'Commissioner Malone', 'Commissioner Headrick', 'County Attorney Fults', 'Commissioner Adams', 'Commissioner Crenshaw', 'Commissioner Thompson', 'Commissioner Hayes', 'Commissioner Hailey', 'Commissioner Carmical', 'Commissioner Austin', 'Commissioner Garland']}


query_dictype() returned a Python dict with the structure and types that we specified in the dictype definition.

Let's print those names:


```python
# print the generated participants list:
for part in out["participants"]:
    print(part)
```

    Chairman Wormsley
    Commissioner Brown
    Commissioner Hobbs
    Commissioner McCroskey
    Commissioner Adkins
    Commissioner Carmical
    Commissioner McKee
    Commissioner Rodgers
    Commissioner Duckett
    Commissioner Reinhart
    Commissioner Malone
    Commissioner Headrick
    County Attorney Fults
    Commissioner Adams
    Commissioner Crenshaw
    Commissioner Thompson
    Commissioner Hayes
    Commissioner Hailey
    Commissioner Carmical
    Commissioner Austin
    Commissioner Garland


Some names appear twice ("Commissioner Carmical") and the "clerk", which is mentioned in the text, is not listed.

It's a matter of opinion if the clerk is an active participant, but let's try to fix the repeated names.

Let's try asking for a list of participants "without repeated entries", in the field's description:


```python
meeting_type = {
    "participants": {"type": [str], "desc": "List of complete names of meeting participants without repeated entries"}
}

out = model.query_dictype(meeting_type,
                          inst_text,
                          in_text)

for part in out["participants"]:
    print(part)
```

    Wormsley
    Brown
    Hobbs
    McCroskey
    Adkins
    Carmical
    McKee
    Rodgers
    Duckett
    Reinhart
    Malone
    Headrick
    Fults
    Adams
    Crenshaw
    Thompson
    Hayes
    Hailey
    Austin
    Garland


Didn't work as expected, repetition is gone but it dropped the titles, only the names are appearing.

Let's try asking for "names and titles":


```python
meeting_type = {
    "participants": {"type": [str], "desc": "List of names and titles of participants without repeated entries"}
}

out = model.query_dictype(meeting_type,
                          inst_text,
                          in_text)

for part in out["participants"]:
    print(part)
```

    Chairman Wormsley
    Commissioner Brown
    Commissioner Hobbs
    Commissioner McCroskey
    Commissioner Adkins
    Commissioner Carmical
    Commissioner McKee
    Commissioner Rodgers
    Commissioner Duckett
    Commissioner Reinhart
    Commissioner Malone
    Commissioner Headrick
    County Attorney Fults
    Commissioner Adams
    Commissioner Crenshaw
    Commissioner Thompson
    Commissioner Hayes
    Commissioner Hailey
    Commissioner Carmical
    Commissioner Austin
    Commissioner Garland


And now "Commissioner Carmical" is repeating again!

Let's move on, the point is that you can also do some prompt engineering with the description field. And this model shortcoming could be dealt with by post-processing the received list.
    
Let's now also request a list of action items mentioned in the transcript:


```python
action_item_type = {
    "index": {"type": int, "desc": "Sequential index for the action item"},
    "name": {"type": str, "desc": "Action item name"}
}

meeting_type = {
    "participants": {"type": [str], "desc": "List of complete names of meeting participants"},
    "action_items": {"type": [action_item_type], "desc": "List of action items in the meeting"}
}

out = model.query_dictype(meeting_type,
                          inst_text,
                          in_text)

print("Participants", "-" * 16)
for part in out["participants"]:
    print(part)
print("Action items", "-" * 16)
for ai in out["action_items"]:
    print(ai)
```

    Participants ----------------
    Chairman Wormsley
    Commissioner Brown
    Commissioner Hobbs
    Commissioner McCroskey
    Commissioner Adkins
    Commissioner Carmical
    Commissioner McKee
    Commissioner Rodgers
    Commissioner Duckett
    Commissioner Reinhart
    Commissioner Malone
    Commissioner Headrick
    County Attorney Fults
    Commissioner Adams
    Commissioner Crenshaw
    Commissioner Thompson
    Commissioner Hayes
    Commissioner Hailey
    Commissioner Carmical
    Commissioner Austin
    Commissioner Garland
    Action items ----------------
    {'index': 1, 'name': 'Approve the agenda'}
    {'index': 2, 'name': 'Correct the minutes to include Commissioner McCroskey in the Special Committee on Indigent Care'}
    {'index': 3, 'name': 'Approve the resolution to transfer funds for laptop purchase'}
    {'index': 4, 'name': 'Withdraw motion to sell property near the airport'}
    {'index': 5, 'name': 'Adopt the resolution to increase the state match local litigation tax'}
    {'index': 6, 'name': "Amend the motion to allocate 25% of increased tax to fund the sheriff's department"}
    {'index': 7, 'name': 'Vote on the motion to increase the state match local litigation taxes with amendment'}
    {'index': 8, 'name': 'Adopt the resolution to increase the wheel tax by $10'}
    {'index': 9, 'name': 'Hold a Budget Committee meeting on solid waste funding recommendations'}
    {'index': 10, 'name': 'Announce the chili supper at County Elementary School'}
    {'index': 11, 'name': 'Adjourn the meeting'}


These are reasonable action items.

Let's now also request a priority for each ActionItem - we'll create a string Enum class with three priority levels.


```python
action_item_type = {
    "index": {"type": int, "desc": "Sequential index for the action item"},
    "name": {"type": str, "desc": "Action item name"},
    "priority": {"type": ["high", "medium", "low"], "desc": "Action item priority"}
}

meeting_type = {
    "participants": {"type": [str], "desc": "List of complete names of meeting participants"},
    "action_items": {"type": [action_item_type], "desc": "List of action items in the meeting"}
}

out = model.query_dictype(meeting_type,
                          inst_text,
                          in_text)

print("Participants", "-" * 16)
for part in out["participants"]:
    print(part)
print("Action items", "-" * 16)
for ai in out["action_items"]:
    print(ai)
```

    Participants ----------------
    Chairman Wormsley
    Commissioner Brown
    Commissioner Hobbs
    Commissioner McCroskey
    Commissioner Adkins
    Commissioner Carmical
    Commissioner McKee
    Commissioner Rodgers
    Commissioner Duckett
    Commissioner Reinhart
    Commissioner Malone
    Commissioner Headrick
    County Attorney Fults
    Commissioner Adams
    Commissioner Crenshaw
    Commissioner Thompson
    Commissioner Hayes
    Commissioner Hailey
    Commissioner Carmical
    Commissioner Austin
    Commissioner Garland
    Action items ----------------
    {'index': 1, 'name': 'Approve the agenda', 'priority': 'high'}
    {'index': 2, 'name': 'Correct the minutes to include Commissioner McCroskey in the Special Committee on Indigent Care', 'priority': 'medium'}
    {'index': 3, 'name': 'Approve the resolution to transfer funds from the Data Processing Reserve Account to purchase a laptop', 'priority': 'medium'}
    {'index': 4, 'name': 'Withdraw motion to sell property near the airport', 'priority': 'low'}
    {'index': 5, 'name': 'Adopt the resolution to increase the state match local litigation tax', 'priority': 'high'}
    {'index': 6, 'name': "Amend the resolution to allocate 25 percent of the increased tax on criminal cases to the sheriff's department", 'priority': 'high'}
    {'index': 7, 'name': 'Vote on the resolution to increase the state match local litigation taxes with the amendment', 'priority': 'high'}
    {'index': 8, 'name': 'Adopt the resolution to increase the wheel tax by $10 for education funding', 'priority': 'high'}
    {'index': 9, 'name': 'Budget Committee meeting on solid waste funding recommendations', 'priority': 'medium'}
    {'index': 10, 'name': 'Chili supper at County Elementary School', 'priority': 'low'}


It's not clear from the meeting transcript text if these priorities are correct, but some items related to taxes are receiving high and medium priorities, looks reasonable that taxes are a priority. : )

It's interesting that comparing with the Pydantic example, action items are the same, but not listed with equal wording, and priorities do not match in all cases. I think there's not priority information stated in the meeting, even for a human it's hard to attribute priorities after reading the transcript.

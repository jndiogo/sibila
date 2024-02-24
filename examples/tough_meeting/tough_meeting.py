# load env variables like OPENAI_API_KEY from a .env file (if available)
try: from dotenv import load_dotenv; load_dotenv()
except: ...

if __name__ == "__main__":

    from sibila import Models, GenConf

    # delete any previous model
    try: del model
    except: ...

    # to use a local model, assuming it's in ../../models:
    # setup models folder:
    # Models.setup("../../models")
    # the transcript is large, so we'll create the model with a context length of 3072, which should be enough.
    # model = Models.create("llamacpp:openchat-3.5-1210.Q4_K_M.gguf", ctx_len=3072)

    # to use an OpenAI model:
    model = Models.create("openai:gpt-4", ctx_len=3072)


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


    from pydantic import BaseModel, Field
    from enum import Enum

    class ActionPriority(str, Enum):
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"
        
    class ActionItem(BaseModel):
        index: int = Field(description="Sequential index for the action item")
        name: str = Field(description="Action item name")
        priority: ActionPriority = Field(description="Action item priority")

    class Meeting(BaseModel):
        participants: list[str] = Field(description="List of complete names of meeting participants")
        action_items: list[ActionItem] = Field(description="List of action items in the meeting")


    # model instructions text, also known as system message
    inst_text = "Extract information and output in JSON format."

    # the input query, with the transcript
    in_text = "Extract information from this meeting transcript:\n\n" + transcript


    out = model.extract(Meeting,
                        in_text,
                        inst=inst_text)


    print("Participants", "-" * 16)
    for part in out.participants:
        print(part)
    print("Action items", "-" * 16)
    for ai in out.action_items:
        print(ai)
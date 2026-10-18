from typing import Annotated, List, Optional, Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class JDSlots(BaseModel):
    role_title: Optional[str] = Field(None, description="Specific role title, e.g., Senior Backend Engineer (strip vague terms if still unclear)")
    experience_years: Optional[str] = Field(None, description="Years of experience required, e.g., 3-5 years")
    work_location: Optional[str] = Field(None, description="Work location and mode, e.g., Bengaluru (Hybrid) or Remote")
    core_skills: List[str] = Field(default_factory=list, description="Must-have technical or domain skills")
    employment_type: Optional[str] = Field("Full-time", description="Full-time, Contract, or Internship")
    compensation_notes: Optional[str] = Field(None, description="Optional salary/CTC range or notice period constraint")


class FormFieldSpec(BaseModel):
    label: str = Field(description="Exact question label shown on the Typeform")
    field_type: Literal["short_text", "long_text", "multiple_choice", "number", "email", "website", "yes_no"] = Field(
        description="Typeform field type. Use 'website' for portfolio/LinkedIn/GitHub/resume links."
    )
    required: bool = Field(True, description="Whether candidate must answer this field")
    options: Optional[List[str]] = Field(None, description="Choices if field_type is multiple_choice")


class ExtractionOutput(BaseModel):
    updated_slots: JDSlots = Field(description="Merged JD slots combining existing state with new user input")
    missing_critical_info: List[str] = Field(
        default_factory=list,
        description="List of missing or overly vague critical JD parameters among: role_title, experience_years, work_location, core_skills"
    )
    approved_form_fields: List[FormFieldSpec] = Field(
        default_factory=list,
        description="Active fields currently included in the candidate application form"
    )
    ai_recommended_fields: List[FormFieldSpec] = Field(
        default_factory=list,
        description="2 to 3 smart, role-specific screening fields suggested by the AI that await user confirmation"
    )
    user_satisfied_to_publish: bool = Field(
        False,
        description="Set to True ONLY when all critical JD slots are present, form fields are populated, AND the user explicitly confirms/approves creating the form"
    )
    spoken_response: str = Field(
        description="Natural, conversational voice reply (strictly under 35 words). Ask clarifying questions if slots are missing, pitch recommended form fields once slots are ready, or confirm form creation."
    )
    draft_jd_markdown: Optional[str] = Field(
        None,
        description="Complete professional Markdown Job Description (Overview, Responsibilities, Requirements, Perks). Populate once role_title, experience_years, work_location, and core_skills are known."
    )


class HiringAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    jd_slots: dict
    form_fields: list
    recommended_fields: list
    missing_info: list
    draft_jd: Optional[str]
    form_url: Optional[str]
    spoken_response: str
    user_satisfied: bool
    stage: Literal["collecting_jd", "refining_form", "completed"]


class InitRequest(BaseModel):
    thread_id: Optional[str] = None
    use_voice: bool = True
    speaker: Optional[str] = None


class ChatRequest(BaseModel):
    thread_id: str
    text: str
    use_voice: bool = True
    speaker: Optional[str] = None

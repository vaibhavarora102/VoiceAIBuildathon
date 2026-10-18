from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END

from config import OPENAI_MODEL
from models import HiringAgentState, ExtractionOutput
from services import create_typeform_form
from agent.prompts import SYSTEM_PROMPT


def analyzer_node(state: HiringAgentState) -> dict:
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)
    structured_llm = llm.with_structured_output(ExtractionOutput)

    state_context = (
        f"CURRENT JD SLOTS: {state.get('jd_slots', {})}\n"
        f"CURRENT APPROVED FORM FIELDS: {state.get('form_fields', [])}\n"
        f"CURRENT AI RECOMMENDED FIELDS: {state.get('recommended_fields', [])}\n"
        f"HAS EXISTING JD DRAFT: {bool(state.get('draft_jd'))}"
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=state_context)
    ] + state["messages"]

    result: ExtractionOutput = structured_llm.invoke(messages)

    new_draft_jd = result.draft_jd_markdown or state.get("draft_jd")
    stage = "collecting_jd"
    if not result.missing_critical_info and new_draft_jd:
        stage = "refining_form"

    # Merge recommendations if approved_form_fields only had baseline
    approved_fields = [f.model_dump() for f in result.approved_form_fields]
    recommended_fields = [f.model_dump() for f in result.ai_recommended_fields]

    return {
        "jd_slots": result.updated_slots.model_dump(),
        "missing_info": result.missing_critical_info,
        "form_fields": approved_fields,
        "recommended_fields": recommended_fields,
        "draft_jd": new_draft_jd,
        "spoken_response": result.spoken_response,
        "user_satisfied": result.user_satisfied_to_publish,
        "stage": stage,
        "messages": [AIMessage(content=result.spoken_response)]
    }


def form_builder_node(state: HiringAgentState) -> dict:
    slots = state.get("jd_slots", {})
    role_title = slots.get("role_title") or "Open Role"
    jd_md = state.get("draft_jd") or ""
    fields = state.get("form_fields") or []

    # Ensure baseline fields exist if list is somehow empty
    if not fields:
        fields = [
            {"label": "Full Name", "field_type": "short_text", "required": True},
            {"label": "Email Address", "field_type": "email", "required": True},
            {"label": "Resume / LinkedIn URL", "field_type": "website", "required": True}
        ]

    form_url = create_typeform_form(role_title=role_title, jd_markdown=jd_md, fields=fields)
    final_speech = (
        f"Awesome! I have generated the Typeform for the {role_title} role with {len(fields)} screening fields. "
        f"Your live form link is ready on the dashboard!"
    )

    return {
        "form_url": form_url,
        "spoken_response": final_speech,
        "stage": "completed",
        "messages": [AIMessage(content=final_speech)]
    }


def route_after_analysis(state: HiringAgentState) -> Literal["form_builder", "__end__"]:
    if state.get("user_satisfied") and not state.get("missing_info"):
        return "form_builder"
    return END

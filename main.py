import os
import re
import uuid
import base64
import requests
from typing import Annotated, List, Optional, Literal
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TYPEFORM_API_KEY = os.getenv("TYPEFORM_API_KEY", "")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
SARVAM_SPEAKER = os.getenv("SARVAM_SPEAKER", "priya")

# =====================================================================
# 1. PYDANTIC SCHEMAS & LANGGRAPH STATE
# =====================================================================

class JDSlots(BaseModel):
    role_title: Optional[str] = Field(None, description="Specific role title, e.g., Senior Backend Engineer ( strip vague terms if still unclear)")
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

# =====================================================================
# 2. EXTERNAL TOOLS: TYPEFORM & SARVAM AI
# =====================================================================

def create_typeform_form(role_title: str, jd_markdown: str, fields: List[dict]) -> str:
    """Creates a live Typeform via the Typeform Create API and returns the public URL."""
    if not TYPEFORM_API_KEY or TYPEFORM_API_KEY.startswith("tfp_your"):
        # Fallback mock URL so local testing never crashes without a Typeform token
        mock_slug = uuid.uuid4().hex[:8]
        return f"https://form.typeform.com/to/mock-{mock_slug}?role={role_title.replace(' ', '+')}"

    tf_fields = []
    for idx, f in enumerate(fields):
        ftype = f.get("field_type", "short_text")
        # Ensure valid Typeform field type
        if ftype not in {"short_text", "long_text", "multiple_choice", "number", "email", "website", "yes_no"}:
            ftype = "short_text"

        field_payload = {
            "title": f.get("label", f"Question {idx + 1}"),
            "type": ftype,
            "validations": {"required": bool(f.get("required", True))}
        }
        if ftype == "multiple_choice":
            opts = f.get("options") or ["Yes", "No"]
            field_payload["properties"] = {
                "choices": [{"label": str(o)} for o in opts],
                "allow_multiple_selection": False
            }
        tf_fields.append(field_payload)

    payload = {
        "title": f"Application: {role_title}",
        "welcome_screens": [
            {
                "title": f"Apply for {role_title}",
                "properties": {
                    "show_button": True,
                    "button_text": "Start Application",
                    "description": "Please complete the short screening questions below."
                }
            }
        ],
        "fields": tf_fields
    }

    resp = requests.post(
        "https://api.typeform.com/forms",
        headers={
            "Authorization": f"Bearer {TYPEFORM_API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=20
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Typeform API error ({resp.status_code}): {resp.text}")

    data = resp.json()
    return data.get("_links", {}).get("display", f"https://form.typeform.com/to/{data.get('id')}")


def sarvam_speech_to_text(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """Calls Sarvam AI Speech-to-Text API."""
    if not SARVAM_API_KEY or SARVAM_API_KEY == "your-sarvam-subscription-key":
        raise HTTPException(status_code=400, detail="SARVAM_API_KEY is not configured in .env. Switch to Text Mode or add your key.")

    files = {
        "file": (filename, audio_bytes, "audio/wav")
    }
    data = {
        "model": SARVAM_STT_MODEL,
        "language_code": "en-IN"
    }
    resp = requests.post(
        "https://api.sarvam.ai/speech-to-text",
        headers={"api-subscription-key": SARVAM_API_KEY},
        files=files,
        data=data,
        timeout=30
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Sarvam STT failed: {resp.text}")
    return resp.json().get("transcript", "").strip()


def sarvam_text_to_speech(text: str, speaker: Optional[str] = None) -> Optional[str]:
    """Calls Sarvam AI Text-to-Speech API and returns base64 WAV audio."""
    if not SARVAM_API_KEY or SARVAM_API_KEY == "your-sarvam-subscription-key" or not text:
        return None

    # Clean text: remove markdown symbols (*, _, #, `, links) so voice synthesis sounds natural
    clean_text = re.sub(r'[*_#`]', '', text)
    clean_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean_text)
    clean_text = clean_text.strip()
    if not clean_text:
        return None

    chosen_speaker = speaker or SARVAM_SPEAKER or "priya"
    payload = {
        "inputs": [clean_text[:480]],
        "target_language_code": "en-IN",
        "speaker": chosen_speaker,
        "pitch": 0,
        "pace": 1.05,
        "loudness": 1.5,
        "speech_sample_rate": 16000,
        "enable_preprocessing": True,
        "model": SARVAM_TTS_MODEL
    }
    try:
        resp = requests.post(
            "https://api.sarvam.ai/text-to-speech",
            headers={
                "api-subscription-key": SARVAM_API_KEY,
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=30
        )
        if resp.status_code != 200:
            print(f"[Warning] Sarvam TTS error ({resp.status_code}): {resp.text}")
            return None

        audios = resp.json().get("audios", [])
        return audios[0] if audios else None
    except Exception as e:
        print(f"[Warning] Sarvam TTS exception: {e}")
        return None



# =====================================================================
# 3. LANGGRAPH NODES & WORKFLOW
# =====================================================================

SYSTEM_PROMPT = """You are an intelligent, concise Voice AI Technical Recruiter & Hiring Assistant.
Your goal is to help the user define a sharp Job Description (JD) and build a tailored candidate screening Typeform.

Follow this strict 3-stage workflow:

STAGE 1: CLARIFY & COLLECT CORE JD SLOTS
- Check if these 4 mandatory parameters are clear and specific:
  1. `role_title` (If the user says something broad like "engineer" or "developer" or "marketing person", treat it as unclear and ask which specialization, e.g., Backend, Frontend, DevOps, Growth).
  2. `experience_years` (e.g., "2-4 years", "Senior 5+ years").
  3. `work_location` (City + Remote/Hybrid/Onsite mode).
  4. `core_skills` (At least 2-3 specific tools, languages, or domain skills).
- While any of these 4 are missing or vague, list them in `missing_critical_info` and ask a friendly, concise voice question (MAX 2 questions at a time, strictly under 35 words in `spoken_response`).

STAGE 2: DRAFT JD & RECOMMEND SMART FORM FIELDS
- As soon as all 4 core JD slots are known:
  1. Generate a complete, well-structured Markdown Job Description in `draft_jd_markdown`.
  2. Populate `approved_form_fields` with essential baseline fields (Full Name [short_text], Email Address [email], Resume / LinkedIn URL [website], Years of Experience [number]).
  3. Populate `ai_recommended_fields` with 2-3 high-signal, role-specific screening fields (for example: GitHub Profile [website], Notice Period in Days [number], or a multiple-choice/short-text question testing their primary skill).
  4. In `spoken_response` (under 35 words), tell the user the JD draft is ready on screen, explicitly mention your 2-3 recommended form fields, and ask if they'd like to include them or add custom fields before generating the form.
- If the user asks to add, accept, modify, or remove specific fields, move accepted fields into `approved_form_fields` and update `ai_recommended_fields` accordingly.

STAGE 3: FINAL APPROVAL TO GENERATE FORM
- ONLY set `user_satisfied_to_publish = True` when:
  1. All 4 core JD slots are filled,
  2. `approved_form_fields` has the desired questions, AND
  3. The user explicitly states they are satisfied or asks to proceed/generate/make the form (e.g., "Looks good, create the form", "Include those recommendations and generate the link", "Okay satisfied with these inputs").
"""

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

    # If user says "accept recommendations and create form" in one go, merge recommendations if approved_form_fields only had baseline
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


def build_hiring_graph():
    builder = StateGraph(HiringAgentState)
    builder.add_node("analyzer", analyzer_node)
    builder.add_node("form_builder", form_builder_node)

    builder.add_edge(START, "analyzer")
    builder.add_conditional_edges(
        "analyzer",
        route_after_analysis,
        {
            "form_builder": "form_builder",
            END: END
        }
    )
    builder.add_edge("form_builder", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


hiring_graph = build_hiring_graph()

# =====================================================================
# 4. FASTAPI APPLICATION & ENDPOINTS
# =====================================================================

app = FastAPI(title="Voice AI Hiring & Form Builder Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InitRequest(BaseModel):
    thread_id: Optional[str] = None
    use_voice: bool = True
    speaker: Optional[str] = None

class ChatRequest(BaseModel):
    thread_id: str
    text: str
    use_voice: bool = True
    speaker: Optional[str] = None

@app.post("/api/init")
def init_session(req: InitRequest):
    """Initializes a new hiring conversation session and optionally returns Sarvam TTS greeting."""
    thread_id = req.thread_id or f"session-{uuid.uuid4().hex[:8]}"
    greeting = "Hi! I'm your AI hiring assistant. What exact role are you looking to hire for today?"

    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "messages": [AIMessage(content=greeting)],
        "jd_slots": JDSlots().model_dump(),
        "form_fields": [],
        "recommended_fields": [],
        "missing_info": ["role_title", "experience_years", "work_location", "core_skills"],
        "draft_jd": None,
        "form_url": None,
        "spoken_response": greeting,
        "user_satisfied": False,
        "stage": "collecting_jd"
    }
    hiring_graph.update_state(config, initial_state)

    audio_b64 = sarvam_text_to_speech(greeting, speaker=req.speaker) if req.use_voice else None
    return {
        "thread_id": thread_id,
        "transcript": None,
        "reply_text": greeting,
        "audio_base64": audio_b64,
        "state": initial_state
    }


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    """Handles text input with optional Sarvam TTS voice audio."""
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-proj-your"):
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is missing in .env")

    config = {"configurable": {"thread_id": req.thread_id}}
    output_state = hiring_graph.invoke(
        {"messages": [HumanMessage(content=req.text)]},
        config=config
    )

    reply_text = output_state.get("spoken_response", "")
    audio_b64 = sarvam_text_to_speech(reply_text, speaker=req.speaker) if req.use_voice else None

    return {
        "thread_id": req.thread_id,
        "transcript": req.text,
        "reply_text": reply_text,
        "audio_base64": audio_b64,
        "state": {
            "jd_slots": output_state.get("jd_slots"),
            "missing_info": output_state.get("missing_info"),
            "form_fields": output_state.get("form_fields"),
            "recommended_fields": output_state.get("recommended_fields"),
            "draft_jd": output_state.get("draft_jd"),
            "form_url": output_state.get("form_url"),
            "stage": output_state.get("stage")
        }
    }


@app.post("/api/voice")
async def voice_endpoint(
    thread_id: str = Form(...),
    use_voice: bool = Form(True),
    speaker: Optional[str] = Form(None),
    audio: UploadFile = File(...)
):
    """Handles recorded voice input -> Sarvam STT -> LangGraph -> Sarvam TTS."""
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-proj-your"):
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is missing in .env")

    audio_bytes = await audio.read()
    transcript = sarvam_speech_to_text(audio_bytes, filename=audio.filename or "audio.wav")
    if not transcript:
        clarification = "I didn't quite catch that. Could you please repeat the role details?"
        audio_b64 = sarvam_text_to_speech(clarification, speaker=speaker) if use_voice else None
        return {
            "thread_id": thread_id,
            "transcript": "",
            "reply_text": clarification,
            "audio_base64": audio_b64,
            "state": None
        }

    config = {"configurable": {"thread_id": thread_id}}
    output_state = hiring_graph.invoke(
        {"messages": [HumanMessage(content=transcript)]},
        config=config
    )

    reply_text = output_state.get("spoken_response", "")
    audio_b64 = sarvam_text_to_speech(reply_text, speaker=speaker) if use_voice else None

    return {
        "thread_id": thread_id,
        "transcript": transcript,
        "reply_text": reply_text,
        "audio_base64": audio_b64,
        "state": {
            "jd_slots": output_state.get("jd_slots"),
            "missing_info": output_state.get("missing_info"),
            "form_fields": output_state.get("form_fields"),
            "recommended_fields": output_state.get("recommended_fields"),
            "draft_jd": output_state.get("draft_jd"),
            "form_url": output_state.get("form_url"),
            "stage": output_state.get("stage")
        }
    }


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")

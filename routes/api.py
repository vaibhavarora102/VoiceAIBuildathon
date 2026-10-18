import uuid
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from langchain_core.messages import HumanMessage, AIMessage

from config import OPENAI_API_KEY
from models import JDSlots, InitRequest, ChatRequest
from services import sarvam_speech_to_text, sarvam_text_to_speech
from agent import hiring_graph

router = APIRouter(prefix="/api", tags=["Agent API"])


@router.post("/init")
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


@router.post("/chat")
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


@router.post("/voice")
def voice_endpoint(
    thread_id: str = Form(...),
    use_voice: bool = Form(True),
    speaker: Optional[str] = Form(None),
    audio: UploadFile = File(...)
):
    """Handles recorded voice input -> Sarvam STT -> LangGraph -> Sarvam TTS."""
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-proj-your"):
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is missing in .env")

    audio_bytes = audio.file.read()
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

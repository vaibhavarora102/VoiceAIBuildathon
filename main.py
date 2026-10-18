from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from routes import api_router

# Backward compatibility re-exports (for notebook.ipynb and external tests)
from services import (
    create_typeform_form,
    sarvam_speech_to_text,
    sarvam_text_to_speech,
)
from agent import (
    hiring_graph,
    build_hiring_graph,
    SYSTEM_PROMPT,
    analyzer_node,
    form_builder_node,
    route_after_analysis,
)
from models import (
    JDSlots,
    FormFieldSpec,
    ExtractionOutput,
    HiringAgentState,
    InitRequest,
    ChatRequest,
)

# Initialize FastAPI application
app = FastAPI(
    title="HireVoice AI — Voice Hiring & Form Builder Agent",
    description="Conversational Voice AI Agent powered by LangGraph, Sarvam AI, and Typeform.",
    version="2.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router)

# Mount static UI files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_ui():
    """Serves the main single-page conversational voice application."""
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

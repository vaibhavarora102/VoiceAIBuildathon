import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TYPEFORM_API_KEY = os.getenv("TYPEFORM_API_KEY", "")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")

# Model Configurations
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
SARVAM_SPEAKER = os.getenv("SARVAM_SPEAKER", "priya")

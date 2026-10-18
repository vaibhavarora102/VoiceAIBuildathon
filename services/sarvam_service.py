import re
import requests
from typing import Optional
from fastapi import HTTPException
from config import (
    SARVAM_API_KEY,
    SARVAM_STT_MODEL,
    SARVAM_TTS_MODEL,
    SARVAM_SPEAKER,
)


def sarvam_speech_to_text(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """Calls Sarvam AI Speech-to-Text API."""
    if not SARVAM_API_KEY or SARVAM_API_KEY == "your-sarvam-subscription-key":
        print("[Warning] SARVAM_API_KEY is not configured in .env")
        return ""

    # Prevent 400 error on zero-duration or tiny audio chunks (< 0.2s)
    if not audio_bytes or len(audio_bytes) < 6000:
        print(f"[Info] Audio buffer too short ({len(audio_bytes) if audio_bytes else 0} bytes), skipping STT")
        return ""

    files = {
        "file": (filename, audio_bytes, "audio/wav")
    }
    data = {
        "model": SARVAM_STT_MODEL,
        "language_code": "en-IN"
    }
    try:
        resp = requests.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": SARVAM_API_KEY},
            files=files,
            data=data,
            timeout=20
        )
        if resp.status_code != 200:
            print(f"[Warning] Sarvam STT returned error ({resp.status_code}): {resp.text}")
            return ""
        return resp.json().get("transcript", "").strip()
    except Exception as e:
        print(f"[Warning] Sarvam STT request exception: {e}")
        return ""



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

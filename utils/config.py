"""
Configuration management for Mental Health AI Interviewer.
Loads settings from environment variables and .env file.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_env_file():
    """Load variables from .env file into os.environ (simple parser)."""
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip().strip("\"'")
                if key not in os.environ:
                    os.environ[key] = value


_load_env_file()

# --- Gemini LLM ---
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME: str = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")

# --- Local LLM ---
LOCAL_LLM_MODEL_NAME: str = os.environ.get(
    "LOCAL_LLM_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"
)
LOCAL_LLM_MAX_NEW_TOKENS: int = int(os.environ.get("LOCAL_LLM_MAX_NEW_TOKENS", "256"))

# --- Model paths ---
DEPRESSION_MODEL_PATH: str = os.environ.get(
    "DEPRESSION_MODEL_PATH",
    str(BASE_DIR / "models" / "depression" / "phq_xgb.pkl"),
)
ANXIETY_MODEL_PATH: str = os.environ.get(
    "ANXIETY_MODEL_PATH",
    str(BASE_DIR / "models" / "anxiety" / "gbr_pipeline.joblib"),
)

# --- SER (Speech Emotion Recognition) ---
SER_MODEL_NAME: str = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
SER_LOAD_TIMEOUT: int = int(os.environ.get("SER_LOAD_TIMEOUT", "120"))

# --- Stress model ---
STRESS_HF_REPO: str = "forwarder1121/voice-based-stress-recognition"

# --- Audio ---
SAMPLE_RATE: int = int(os.environ.get("SAMPLE_RATE", "16000"))
ANXIETY_N_MFCC: int = int(os.environ.get("ANXIETY_N_MFCC", "40"))

# --- Server ---
HOST: str = os.environ.get("HOST", "0.0.0.0")
PORT: int = int(os.environ.get("PORT", "8000"))
CORS_ORIGINS: list = os.environ.get("CORS_ORIGINS", "*").split(",")

# --- Interview ---
TOTAL_QUESTIONS: int = int(os.environ.get("TOTAL_QUESTIONS", "5"))

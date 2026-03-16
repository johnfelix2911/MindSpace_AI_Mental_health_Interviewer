# Architecture

## System Overview

```
Browser ──HTTP──> FastAPI (main.py)
                    │
                    ├─ /submit_demographics  ──> session_store
                    ├─ /start                ──> questionnaire ──> llm_service (Gemini)
                    ├─ /next_question        ──> questionnaire ──> llm_service (Gemini)
                    ├─ /analyze_speech       ──> assessment_router
                    │                              ├─ depression_pipeline (local)
                    │                              ├─ anxiety_pipeline    (local)
                    │                              └─ stress_pipeline     (local)
                    ├─ /results              ──> recommendations + llm_service (Gemini)
                    └─ /exit_session         ──> session_store
```

## Inference Pipelines

### Depression
1. Audio → FFmpeg → WAV (16 kHz mono)
2. WAV → wav2vec2-large-xlsr-53-english (SER) → emotion scores per 4s window
3. Aggregate (mean, std, max) → feature vector
4. Feature vector → XGBoost (`phq_xgb.pkl`) → PHQ-8 score (0–24)

### Anxiety
1. Audio → FFmpeg → WAV (16 kHz mono)
2. WAV → librosa (40 MFCCs + deltas) + parselmouth (pitch, jitter) → 81 features
3. Features → GBR pipeline (`gbr_pipeline.joblib`) → anxiety score (0–24)

### Stress
1. Audio → torchaudio → WAV (16 kHz mono)
2. WAV → Wav2Vec2-base CNN feature extractor → 512-dim embedding
3. Embedding → StudentNet classifier → P(stressed) ∈ [0, 1]

## LLM Usage (Gemini)

Gemini API is used for two purposes only:
1. **Question generation** – personalized interview questions based on demographics + conversation history
2. **Recommendation generation** – final summary and suggested actions based on scores + demographics

All scoring/inference is entirely local. Gemini handles only natural language generation.

## Session Management

Simple in-memory dictionary keyed by session ID. Stores:
- Demographics
- Assessment type
- Conversation history
- Per-pipeline score lists
- Question state

For production scaling, replace `session_store.py` with Redis-backed implementation.

## Future: Local LLM Swap

The `services/llm_service.py` module isolates all Gemini calls. To switch to a local LLM:
1. Replace `_call_gemini(prompt)` with local model inference
2. Keep the same function signatures
3. No other files need to change

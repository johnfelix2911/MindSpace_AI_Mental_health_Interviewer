# MindSpace – Mental Health AI Interviewer

Integrated AI-powered mental health screening application that assesses **depression**, **anxiety**, and **stress** through conversational speech analysis.

## Architecture

| Layer | Technology |
|---|---|
| Frontend | HTML / CSS / JS (served by FastAPI) |
| Backend | Python FastAPI |
| LLM | Google Gemini API (questions & recommendations) |
| Depression inference | wav2vec2 SER → XGBoost (local) |
| Anxiety inference | MFCC + pitch/jitter → GBR pipeline (local) |
| Stress inference | Wav2Vec2-base embeddings → StudentNet (local) |

All ML inference runs **locally** — only Gemini API calls go online.

## User Flow

1. Fill demographic form & choose assessment type (depression / anxiety / stress / all)
2. Gemini generates personalized interview questions
3. User records spoken responses (5 questions)
4. Audio is analyzed silently by local ML pipelines in the background
5. Gemini generates a final recommendation based on scores + demographics
6. Results page shows scores, severity labels, and recommendations

## Quick Start

```bash
# 1. Clone and enter directory
cd Mental-Health-ai-interviewer

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure Gemini API key
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 5. Download/cache ML models (first time only)
python scripts/prepare_offline_assets.py

# 6. Verify setup
python scripts/verify_setup.py

# 7. Run the server
python main.py
```

Open http://localhost:8000 in your browser.

## Project Structure

```
Mental-Health-ai-interviewer/
  main.py                    # FastAPI application
  index.html                 # Interview chat UI
  demographic.html/css/js    # Demographic form

  models/
    depression/phq_xgb.pkl   # XGBoost depression model
    anxiety/gbr_pipeline.joblib  # Anxiety GBR pipeline

  pipelines/
    depression_pipeline.py   # wav2vec2 SER → XGBoost
    anxiety_pipeline.py      # MFCC/pitch → GBR
    stress_pipeline.py       # Wav2Vec2-base → StudentNet

  services/
    llm_service.py           # Gemini API (questions & recommendations)
    questionnaire.py         # Question flow management
    recommendations.py       # Score interpretation & severity
    assessment_router.py     # Routes analysis to correct pipeline(s)
    session_store.py         # In-memory session management

  utils/
    config.py                # Environment & config loading
    audio_common.py          # Shared audio conversion (ffmpeg/pydub/librosa)
    file_utils.py            # Temp file helpers

  scripts/
    download_ser_model.py    # Cache depression SER model
    download_stress_models.py # Cache stress models
    prepare_offline_assets.py # Run all model downloads
    verify_setup.py          # Check everything is ready
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Demographic form |
| GET | `/interview` | Interview chat page |
| GET | `/health` | Health check |
| GET | `/model_status` | ML model status |
| GET | `/llm_status` | Gemini availability |
| POST | `/submit_demographics` | Save demographics + assessment type |
| GET | `/start` | Begin interview session |
| POST | `/next_question` | Get next question (with transcript) |
| POST | `/analyze_speech` | Upload audio for analysis |
| GET | `/results` | Get final scores & recommendations |
| POST | `/exit_session` | Clear session |

## Requirements

- Python 3.10+
- FFmpeg (for audio conversion)
- Gemini API key
- ~3 GB disk for cached ML models

## Deployment

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for server deployment instructions.

"""
Mental Health AI Interviewer – FastAPI application.
Unified backend serving depression, anxiety, and stress assessments.
"""

import os
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from utils.config import CORS_ORIGINS, HOST, PORT, TOTAL_QUESTIONS
from services.session_store import get_or_create_default, delete_session
from services.questionnaire import get_first_question, get_next_question
from services.assessment_router import analyze_audio
from services.recommendations import compute_labels
from services.llm_service import (
    is_llm_available,
    is_local_llm_available,
    generate_recommendation,
)
from pipelines.depression_pipeline import (
    load_model as load_depression_model,
    is_demo_mode as depression_demo_mode,
    is_ser_cached,
    is_ser_loaded,
)
from utils.file_utils import save_upload_to_temp, cleanup, ext_from_content_type

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── FastAPI app ───────────────────────────────────────────────

app = FastAPI(
    title="Mental Health AI Interviewer",
    description="Integrated depression, anxiety, and stress screening via speech analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ──────────────────────────────────────────

class DemographicData(BaseModel):
    name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    country: Optional[str] = None
    role: Optional[str] = None
    stage: Optional[str] = None
    focus: Optional[str] = None
    sleep_duration: Optional[str] = None
    workload: Optional[str] = None
    screen_time: Optional[str] = None
    living_situation: Optional[str] = None
    support_system: Optional[str] = None
    stressors: Optional[List[str]] = []
    assessment_type: Optional[str] = "all"


class NextQuestionRequest(BaseModel):
    transcript: Optional[str] = None


# ── Startup ───────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    print("\n" + "="*55)
    print("  MindSpace — Mental Health AI Interviewer starting ...")
    print("="*55)
    load_depression_model()
    if depression_demo_mode():
        print("  [WARNING] Depression model running in DEMO MODE")
    else:
        print("  [OK] Depression XGBoost model ready")
    llm_ok = is_llm_available()
    print(f"  [{'OK' if llm_ok else 'WARN'}] Gemini LLM: {'available' if llm_ok else 'not configured (set GEMINI_API_KEY)'}")
    local_ok = is_local_llm_available()
    print(f"  [{'OK' if local_ok else 'INFO'}] Local LLM: "
          f"{'cached and available' if local_ok else 'not downloaded (run: python scripts/download_local_llm.py)'}")
    print(f"  [INFO] Server: http://{HOST}:{PORT}")
    print("="*55 + "\n")


# ── Static file routes ────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home():
    print(f"[ROUTE] GET /  → serving demographic.html")
    path = os.path.join(BASE_DIR, "demographic.html")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Mental Health AI Interviewer</h1><p>demographic.html not found</p>"


@app.get("/interview", response_class=HTMLResponse)
async def interview():
    print(f"[ROUTE] GET /interview  → serving index.html")
    path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Interview page not found</h1>"


@app.get("/demographic.css")
async def get_css():
    path = os.path.join(BASE_DIR, "demographic.css")
    if os.path.exists(path):
        return FileResponse(path, media_type="text/css")
    raise HTTPException(404, "CSS not found")


@app.get("/demographic.js")
async def get_js():
    path = os.path.join(BASE_DIR, "demographic.js")
    if os.path.exists(path):
        return FileResponse(path, media_type="application/javascript")
    raise HTTPException(404, "JS not found")


# ── Health / status endpoints ─────────────────────────────────

@app.get("/health")
async def health_check():
    print(f"[ROUTE] GET /health")
    return {
        "status": "healthy",
        "depression_model_loaded": not depression_demo_mode(),
    }


@app.get("/model_status")
async def model_status():
    print(f"[ROUTE] GET /model_status")
    from pipelines.depression_pipeline import _ser_pipeline_error, SER_MODEL_NAME
    ser_cached = is_ser_cached()
    ser_loaded = is_ser_loaded()
    print(f"[STATUS] SER cached: {ser_cached}  SER loaded: {ser_loaded}  Dep model: {not depression_demo_mode()}")
    return {
        "ser_model_cached": ser_cached,
        "ser_pipeline_loaded": ser_loaded,
        "ser_error": _ser_pipeline_error,
        "depression_model": not depression_demo_mode(),
        "message": "Models ready" if ser_loaded else "Run setup scripts to download models",
    }


@app.get("/llm_status")
async def llm_status():
    print(f"[ROUTE] GET /llm_status")
    available = is_llm_available()
    print(f"[STATUS] LLM available: {available}")
    return {
        "llm_available": available,
        "message": "Gemini LLM ready" if available else "Set GEMINI_API_KEY in .env",
    }


@app.get("/local_llm_status")
async def local_llm_status():
    print(f"[ROUTE] GET /local_llm_status")
    from utils.config import LOCAL_LLM_MODEL_NAME
    available = is_local_llm_available()
    print(f"[STATUS] Local LLM ({LOCAL_LLM_MODEL_NAME}) available: {available}")
    return {
        "local_llm_available": available,
        "model_name": LOCAL_LLM_MODEL_NAME,
        "message": "Local LLM ready" if available else "Run: python scripts/download_local_llm.py",
    }


# ── Demographics ──────────────────────────────────────────────

@app.post("/submit_demographics")
async def submit_demographics(data: DemographicData):
    print(f"\n[ROUTE] POST /submit_demographics")
    sid, session = get_or_create_default()

    session["demographics"] = data.dict(exclude={"assessment_type"})
    session["assessment_type"] = data.assessment_type or "all"

    # Log what was received
    name    = data.name or "(anonymous)"
    age     = data.age or "?"
    atype   = session["assessment_type"]
    print(f"[DEMO] Participant: {name}, age {age}")
    print(f"[DEMO] Role: {data.role or '?'}  |  Workload: {data.workload or '?'}  |  "
          f"Sleep: {data.sleep_duration or '?'}")
    print(f"[DEMO] Assessment type selected: {atype.upper()}")
    print(f"[DEMO] Stressors: {data.stressors or []}")

    # Persist for debugging
    try:
        path = os.path.join(BASE_DIR, "user_demographics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data.dict(), f, indent=2)
        print(f"[DEMO] Demographics saved to user_demographics.json")
    except Exception:
        pass

    return {
        "status": "success",
        "redirect_url": "/interview",
        "assessment_type": atype,
    }


# ── Interview flow ────────────────────────────────────────────

@app.get("/start")
async def start_interview(llm_mode: str = "gemini"):
    print(f"\n[ROUTE] GET /start  llm_mode={llm_mode}")
    sid, session = get_or_create_default()
    demographics    = session.get("demographics", {})
    assessment_type = session.get("assessment_type", "all")
    total_questions = TOTAL_QUESTIONS

    # Validate
    if llm_mode not in ("gemini", "local"):
        llm_mode = "gemini"

    # Persist chosen backend in session for the whole interview
    session["llm_mode"] = llm_mode

    print(f"[FLOW] Starting interview — assessment: {assessment_type.upper()}  "
          f"llm_mode: {llm_mode}  total questions: {total_questions}")

    first_q, q_mode = get_first_question(
        demographics, assessment_type, total_questions, llm_mode=llm_mode
    )
    print(f"[FLOW] Question mode: {q_mode.upper()}")

    session.update({
        "current_question": 0,
        "questions": [first_q],
        "conversation_history": [],
        "question_mode": q_mode,
        "total_questions": total_questions,
        "scores": {"depression": [], "anxiety": [], "stress": []},
    })

    print(f"[FLOW] Session initialised. Q1 sent to client.")
    return {
        "question": first_q,
        "question_number": 1,
        "total_questions": total_questions,
        "completed": False,
        "question_mode": q_mode,
        "assessment_type": assessment_type,
        "llm_mode": llm_mode,
    }


@app.post("/next_question")
async def next_question_route(request: NextQuestionRequest = None):
    print(f"\n[ROUTE] POST /next_question")
    sid, session    = get_or_create_default()
    current_idx     = session["current_question"]
    total_q         = session.get("total_questions", TOTAL_QUESTIONS)
    history         = session.get("conversation_history", [])
    q_mode          = session.get("question_mode", "llm")
    demographics    = session.get("demographics", {})
    assessment_type = session.get("assessment_type", "all")

    # Store the user's transcript for the question just answered
    transcript = request.transcript if request else None
    if current_idx < len(session["questions"]) and transcript:
        answered_q = session["questions"][current_idx]
        history.append({"question": answered_q, "answer": transcript})
        session["conversation_history"] = history
        short_t = transcript[:60] + "..." if len(transcript) > 60 else transcript
        print(f"[FLOW] Q{current_idx + 1} answered — transcript: \"{short_t}\"")
        print(f"[FLOW] Conversation history length: {len(history)}")
    else:
        print(f"[FLOW] Q{current_idx + 1} — no transcript provided (audio-only).")

    session["current_question"] += 1
    next_idx = session["current_question"]

    if next_idx >= total_q:
        print(f"[FLOW] All {total_q} questions answered — interview COMPLETE.")
        return {
            "question": "Thank you for completing the interview.",
            "question_number": next_idx + 1,
            "total_questions": total_q,
            "completed": True,
        }

    print(f"[FLOW] Requesting Q{next_idx + 1}/{total_q} ...")
    llm_mode = session.get("llm_mode", "gemini")
    next_q = get_next_question(
        demographics, history, next_idx + 1, total_q, q_mode, assessment_type,
        llm_mode=llm_mode,
    )
    session["questions"].append(next_q)

    print(f"[FLOW] Q{next_idx + 1} ready: \"{next_q[:60]}{'...' if len(next_q) > 60 else ''}\"")
    return {
        "question": next_q,
        "question_number": next_idx + 1,
        "total_questions": total_q,
        "completed": False,
    }


# ── Audio analysis ────────────────────────────────────────────

@app.post("/analyze_speech")
async def analyze_speech(audio: UploadFile = File(...), force_demo: bool = False):
    print(f"\n[ROUTE] POST /analyze_speech  force_demo={force_demo}")
    sid, session    = get_or_create_default()
    assessment_type = session.get("assessment_type", "all")
    current_q_idx   = session.get("current_question", 0)

    print(f"[AUDIO] Received audio: {audio.filename}  content_type: {audio.content_type}")
    print(f"[AUDIO] Current question index: {current_q_idx}  assessment: {assessment_type.upper()}")

    # Demo / quick-test mode — skip real inference, inject mock scores
    if force_demo:
        import random
        mock = round(random.uniform(3, 12), 1)
        for key in ("depression", "anxiety", "stress"):
            if assessment_type in (key, "all"):
                val = mock if key != "stress" else round(random.uniform(0.2, 0.8), 2)
                session["scores"][key].append(val)
                print(f"[AUDIO] DEMO MODE — {key} mock score injected: {val}")
        print(f"[AUDIO] Running scores so far: {session['scores']}")
        return {"status": "ok", "demo_mode": True, "message": "Quick test – mock scores stored"}

    # Real inference path
    ext     = ext_from_content_type(audio.content_type or "", audio.filename)
    content = await audio.read()
    size_kb = len(content) // 1024
    print(f"[AUDIO] Audio received — size: {size_kb} KB  ext: {ext}")

    tmp_path = save_upload_to_temp(content, suffix=ext)
    print(f"[AUDIO] Saved to temp file: {os.path.basename(tmp_path)}")

    try:
        print(f"[AUDIO] Dispatching to inference router ...")
        results = analyze_audio(tmp_path, assessment_type)

        # Append each result score to the running session scores list
        for key in ("depression", "anxiety", "stress"):
            if key in results:
                score = results[key].get("score", 0)
                session["scores"][key].append(score)
                print(f"[AUDIO] Score appended — {key}: {score}  "
                      f"(running list: {session['scores'][key]})")

        print(f"[AUDIO] Inference complete. Accumulated scores: {session['scores']}")
        return {"status": "ok", "demo_mode": False}
    except Exception as e:
        print(f"[AUDIO] ERROR during inference: {e}")
        return {"status": "error", "message": str(e), "demo_mode": True}
    finally:
        cleanup(tmp_path)
        print(f"[AUDIO] Temp file cleaned up.")


# ── Results ───────────────────────────────────────────────────

@app.get("/results")
async def get_results():
    print(f"\n[ROUTE] GET /results")
    sid, session    = get_or_create_default()
    assessment_type = session.get("assessment_type", "all")
    demographics    = session.get("demographics", {})
    llm_mode        = session.get("llm_mode", "gemini")

    print(f"[RESULTS] Computing final scores for assessment: {assessment_type.upper()}  llm_mode: {llm_mode}")
    print(f"[RESULTS] Raw session scores: {session.get('scores', {})}")

    # Average all per-question scores into a single value per assessment
    avg_scores = {}
    for key in ("depression", "anxiety", "stress"):
        if assessment_type in (key, "all"):
            vals = session["scores"].get(key, [])
            avg = round(sum(vals) / len(vals), 2) if vals else 0.0
            avg_scores[key] = avg
            print(f"[RESULTS] {key}: {vals} → average = {avg}")

    print(f"[RESULTS] Final averaged scores: {avg_scores}")

    labels = compute_labels(avg_scores)

    # Generate recommendation via the chosen backend
    if llm_mode == "local" and is_local_llm_available():
        print(f"[RESULTS] Generating AI recommendation via LOCAL LLM ...")
        recommendation = generate_recommendation(
            avg_scores, labels, demographics, assessment_type, llm_mode="local"
        )
    elif llm_mode == "gemini" and is_llm_available():
        print(f"[RESULTS] Generating AI recommendation via Gemini ...")
        recommendation = generate_recommendation(
            avg_scores, labels, demographics, assessment_type, llm_mode="gemini"
        )
    else:
        print(f"[RESULTS] Using hardcoded recommendation (no LLM available).")
        from services.llm_service import _hardcoded_recommendation
        recommendation = _hardcoded_recommendation(avg_scores, labels, assessment_type)

    print(f"[RESULTS] All done. Returning results to client.")
    return {
        "completed": True,
        "assessment_type": assessment_type,
        "scores": avg_scores,
        "labels": labels,
        "details": {},
        "recommendation": recommendation,
        "demo_mode": depression_demo_mode(),
    }


# ── Session management ────────────────────────────────────────

@app.post("/exit_session")
async def exit_session():
    print(f"[ROUTE] POST /exit_session  → clearing session")
    delete_session("default")
    return {"status": "ended", "message": "Session ended. Start a new session whenever you're ready."}


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)

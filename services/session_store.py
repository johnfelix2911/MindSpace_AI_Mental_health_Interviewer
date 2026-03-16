"""
Per-user session store.
Simple in-memory dict keyed by session_id.
For production, swap with Redis or a database backend.
"""

import uuid
from typing import Any


_sessions: dict[str, dict[str, Any]] = {}


def create_session() -> str:
    """Create a new session and return its ID."""
    sid = uuid.uuid4().hex[:12]
    _sessions[sid] = {
        "demographics": {},
        "assessment_type": "all",
        "current_question": 0,
        "total_questions": 5,
        "questions": [],
        "conversation_history": [],
        "scores": {
            "depression": [],
            "anxiety": [],
            "stress": [],
        },
        "question_mode": "llm",
        "llm_mode": "gemini",
    }
    print(f"[SESSION] New session created: {sid}")
    return sid


def get_session(sid: str) -> dict[str, Any] | None:
    """Retrieve session by ID, or None."""
    session = _sessions.get(sid)
    if session is None:
        print(f"[SESSION] Session not found: {sid}")
    return session


def get_or_create_default() -> tuple[str, dict[str, Any]]:
    """Return the 'default' session, creating if needed."""
    sid = "default"
    if sid not in _sessions:
        print(f"[SESSION] Creating default session ...")
        _sessions[sid] = {
            "demographics": {},
            "assessment_type": "all",
            "current_question": 0,
            "total_questions": 5,
            "questions": [],
            "conversation_history": [],
            "scores": {
                "depression": [],
                "anxiety": [],
                "stress": [],
            },
            "question_mode": "llm",
            "llm_mode": "gemini",
        }
        print(f"[SESSION] Default session initialised.")
    return sid, _sessions[sid]


def delete_session(sid: str) -> None:
    """Remove a session entirely."""
    if sid in _sessions:
        _sessions.pop(sid, None)
        print(f"[SESSION] Session '{sid}' deleted.")
    else:
        print(f"[SESSION] Delete called on non-existent session: {sid}")

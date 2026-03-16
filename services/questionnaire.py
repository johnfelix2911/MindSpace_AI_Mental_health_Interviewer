"""
Interview question-flow management.
Tracks question count, state, and delegates to llm_service for generation.
"""

from typing import Dict, List

from services.llm_service import (
    generate_next_question as _llm_next_question,
    is_llm_available,
    is_local_llm_available,
    HARDCODED_QUESTIONS,
)
from utils.config import TOTAL_QUESTIONS


def get_first_question(
    demographics: Dict,
    assessment_type: str = "all",
    total_questions: int = TOTAL_QUESTIONS,
    llm_mode: str = "gemini",
) -> tuple[str, str]:
    """
    Return (first_question_text, question_mode).
    question_mode is 'llm' or 'static'.
    """
    if llm_mode == "local":
        llm_ok = is_local_llm_available()
    else:
        llm_ok = is_llm_available()

    print(f"[QUESTIONNAIRE] Getting first question — mode: {llm_mode}  "
          f"available: {llm_ok}  assessment: {assessment_type}  total: {total_questions}")

    if llm_ok:
        print(f"[QUESTIONNAIRE] Generating Q1 via {llm_mode} ...")
        q = _llm_next_question(
            demographics, [], 1, total_questions, assessment_type,
            llm_mode=llm_mode,
        )
        print(f"[QUESTIONNAIRE] Q1 ({llm_mode}): {q[:80]}{'...' if len(q) > 80 else ''}")
        return q, "llm"

    print(f"[QUESTIONNAIRE] Using hardcoded Q1 (static mode).")
    return HARDCODED_QUESTIONS[0], "static"


def get_next_question(
    demographics: Dict,
    conversation_history: List[Dict],
    question_number: int,
    total_questions: int = TOTAL_QUESTIONS,
    question_mode: str = "llm",
    assessment_type: str = "all",
    llm_mode: str = "gemini",
) -> str:
    """Generate or retrieve the next question."""
    print(f"[QUESTIONNAIRE] Generating Q{question_number}/{total_questions}  "
          f"mode: {question_mode}  llm_mode: {llm_mode}  history_length: {len(conversation_history)}")

    if question_mode == "llm":
        if llm_mode == "local":
            ok = is_local_llm_available()
        else:
            ok = is_llm_available()

        if ok:
            print(f"[QUESTIONNAIRE] Calling {llm_mode} for Q{question_number} ...")
            q = _llm_next_question(
                demographics, conversation_history,
                question_number, total_questions, assessment_type,
                llm_mode=llm_mode,
            )
            print(f"[QUESTIONNAIRE] Q{question_number} ({llm_mode}): {q[:80]}{'...' if len(q) > 80 else ''}")
            return q

    idx = min(question_number - 1, len(HARDCODED_QUESTIONS) - 1)
    q   = HARDCODED_QUESTIONS[idx]
    print(f"[QUESTIONNAIRE] Q{question_number} (static, idx={idx}): {q[:80]}")
    return q

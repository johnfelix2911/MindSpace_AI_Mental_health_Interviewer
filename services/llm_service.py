"""
LLM service layer – supports Gemini API and local LLM (Qwen2.5-0.5B-Instruct).
All LLM calls are isolated here so the rest of the app is backend-agnostic.
"""

import json
import re
import os
from typing import Dict, List, Optional

from utils.config import (
    GEMINI_API_KEY, GEMINI_MODEL_NAME,
    LOCAL_LLM_MODEL_NAME, LOCAL_LLM_MAX_NEW_TOKENS,
)

# ---------------------------------------------------------------
# Lazy Gemini setup
# ---------------------------------------------------------------
_genai = None
_model = None
_available: Optional[bool] = None


def _ensure_genai():
    global _genai, _model, _available
    if _available is not None:
        return _available
    try:
        import google.generativeai as genai
        _genai = genai
        if not GEMINI_API_KEY:
            print(f"[LLM] WARNING — GEMINI_API_KEY not set. LLM unavailable.")
            _available = False
            return False
        genai.configure(api_key=GEMINI_API_KEY)
        _model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        _available = True
        print(f"[LLM] Gemini initialised — model: {GEMINI_MODEL_NAME}")
    except Exception as e:
        print(f"[LLM] WARNING — Gemini init failed: {e}")
        _available = False
    return _available


def is_llm_available() -> bool:
    return _ensure_genai()


def _call_gemini(prompt: str) -> str:
    """Send *prompt* to Gemini and return the text response."""
    print(f"[LLM] Sending request to Gemini ({GEMINI_MODEL_NAME}) ...")
    _ensure_genai()
    if not _model:
        raise RuntimeError("Gemini model not initialised")
    response = _model.generate_content(prompt)
    text = response.text.strip()
    print(f"[LLM] Gemini response received ({len(text)} chars).")
    return text


# ---------------------------------------------------------------
# Lazy LOCAL LLM setup
# ---------------------------------------------------------------
_local_tokenizer = None
_local_model = None
_local_available: Optional[bool] = None


def _is_local_model_cached() -> bool:
    """Lightweight check: is the local LLM in the HF cache? (no loading)"""
    # Method 1: huggingface_hub API
    try:
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(LOCAL_LLM_MODEL_NAME, "config.json")
        if result is not None and isinstance(result, str):
            return True
    except Exception:
        pass

    # Method 2: check cache directory manually
    try:
        cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "hub"
        )
        model_dir = os.path.join(
            cache_dir, "models--" + LOCAL_LLM_MODEL_NAME.replace("/", "--")
        )
        snapshots = os.path.join(model_dir, "snapshots")
        if os.path.isdir(snapshots) and os.listdir(snapshots):
            return True
    except Exception:
        pass

    return False


def is_local_llm_available() -> bool:
    """Check if the local LLM model is downloaded (does NOT load it)."""
    return _is_local_model_cached()


def _ensure_local_llm() -> bool:
    """Lazily load the local LLM into memory. Only called on first inference."""
    global _local_tokenizer, _local_model, _local_available
    if _local_available is not None:
        return _local_available

    if not _is_local_model_cached():
        print(f"[LLM-LOCAL] Model not cached: {LOCAL_LLM_MODEL_NAME}")
        _local_available = False
        return False

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        print(f"[LLM-LOCAL] Loading local model: {LOCAL_LLM_MODEL_NAME} ...")
        _local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_MODEL_NAME)
        _local_model = AutoModelForCausalLM.from_pretrained(
            LOCAL_LLM_MODEL_NAME,
            torch_dtype=torch.float32,
        ).to("cpu")
        _local_model.eval()
        _local_available = True
        print(f"[LLM-LOCAL] Model loaded successfully on CPU.")
    except Exception as e:
        print(f"[LLM-LOCAL] WARNING — Local LLM init failed: {e}")
        _local_available = False
    return _local_available


def _call_local_llm(prompt: str) -> str:
    """Run inference on the local LLM and return the text response."""
    import torch

    print(f"[LLM-LOCAL] Generating response (max {LOCAL_LLM_MAX_NEW_TOKENS} tokens) ...")
    if not _ensure_local_llm():
        raise RuntimeError("Local LLM model not loaded")

    messages = [
        {"role": "system", "content": "You are a helpful, compassionate mental health interviewer assistant. Follow instructions precisely and be concise."},
        {"role": "user", "content": prompt},
    ]

    text_input = _local_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _local_tokenizer(text_input, return_tensors="pt")

    with torch.no_grad():
        output = _local_model.generate(
            **inputs,
            max_new_tokens=LOCAL_LLM_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=_local_tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    generated = output[0][inputs["input_ids"].shape[1]:]
    result = _local_tokenizer.decode(generated, skip_special_tokens=True).strip()
    print(f"[LLM-LOCAL] Response received ({len(result)} chars).")
    return result


# ---------------------------------------------------------------
# Robust JSON parsing (handles small-model quirks)
# ---------------------------------------------------------------
def _parse_json_response(text: str):
    """Strip markdown fences and parse JSON with multiple fallback strategies."""
    t = text.strip()

    # Strategy 1: strip markdown code fences
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.startswith("json"):
            t = t[4:]
        t = t.strip()

    # Strategy 2: direct parse
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass

    # Strategy 3: find the first { ... } block via brace matching
    try:
        start = t.index("{")
        depth = 0
        end = start
        for i in range(start, len(t)):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        return json.loads(t[start:end])
    except (ValueError, json.JSONDecodeError):
        pass

    # Strategy 4: fix common small-model JSON issues
    try:
        fixed = t.replace("'", '"')
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        start = fixed.index("{")
        depth = 0
        end = start
        for i in range(start, len(fixed)):
            if fixed[i] == "{":
                depth += 1
            elif fixed[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        return json.loads(fixed[start:end])
    except Exception:
        pass

    raise json.JSONDecodeError("Could not parse JSON from response", t, 0)


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

def format_demographics(demographics: Dict) -> str:
    """Format demographics dict into a readable string for prompts."""
    if not demographics:
        return "No demographic information provided."
    field_labels = {
        "name": "Name", "age": "Age", "gender": "Gender",
        "country": "Location", "role": "Role", "stage": "Current Stage",
        "focus": "Primary Focus", "sleep_duration": "Sleep",
        "workload": "Workload", "screen_time": "Screen Time",
        "living_situation": "Living Situation",
        "support_system": "Support System",
    }
    parts = []
    for key, label in field_labels.items():
        val = demographics.get(key)
        if val:
            parts.append(f"- {label}: {val}")
    stressors = demographics.get("stressors")
    if isinstance(stressors, list) and stressors:
        parts.append(f"- Current Stressors: {', '.join(stressors)}")
    return "\n".join(parts) if parts else "No demographic information provided."


# ---------------------------------------------------------------
# Simplified prompt builders for the local 0.5B model
# ---------------------------------------------------------------

def _build_local_question_prompt(
    demographics: Dict,
    conversation_history: list,
    question_number: int,
    total_questions: int,
    assessment_type: str,
) -> str:
    demo_ctx = format_demographics(demographics)
    name = demographics.get("name", "")

    conv_text = ""
    if conversation_history:
        recent = conversation_history[-2:]
        for entry in recent:
            conv_text += f"Q: {entry.get('question', 'N/A')}\n"
            conv_text += f"A: {entry.get('answer', 'No response')}\n"

    assessment_label = {
        "depression": "depression",
        "anxiety": "anxiety",
        "stress": "stress",
        "all": "mental health",
    }.get(assessment_type, "mental health")

    focus_areas = [
        "current mood and feelings",
        "sleep and energy",
        "social connections",
        "work or study stress",
        "self-worth and future outlook",
    ]
    focus = focus_areas[min(question_number - 1, len(focus_areas) - 1)]

    name_instruction = ""
    if name and question_number in (1, 3):
        name_instruction = f"Use their name '{name}' in this question."
    elif name:
        name_instruction = "Do NOT use their name in this question."

    prompt = f"""Generate one short interview question about {assessment_label}.

Profile:
{demo_ctx}

{conv_text}

This is question {question_number} of {total_questions}. Topic: {focus}
{name_instruction}

Rules:
- Maximum 20 words
- Be warm and personal
- Reference their profile details
- No clinical language

Return ONLY the question text, nothing else."""

    return prompt


def _build_local_recommendation_prompt(
    scores: Dict[str, float],
    labels: Dict[str, str],
    demographics: Dict,
    assessment_type: str,
) -> str:
    demo_ctx = format_demographics(demographics)
    name = demographics.get("name", "")

    score_lines = []
    for key in ("depression", "anxiety", "stress"):
        if key in scores:
            score_lines.append(f"{key}: {scores[key]:.1f} ({labels.get(key, 'N/A')})")
    score_text = ", ".join(score_lines)

    name_note = f"Use the name '{name}' once." if name else "Do not use any name."

    prompt = f"""You are a mental health advisor. Write recommendations based on screening results.

Profile:
{demo_ctx}

Results: {score_text}

{name_note}

Return a JSON object with exactly this structure:
{{"summary": "2-3 sentences about results", "recommendations": ["action 1", "action 2", "action 3", "action 4"], "resources": ["resource 1", "resource 2"], "encouragement": "one warm sentence"}}

Each recommendation should be one short sentence starting with a verb.
Return ONLY valid JSON, no other text."""

    return prompt


# ---------------------------------------------------------------
# Question generation (dual-backend)
# ---------------------------------------------------------------

def generate_next_question(
    demographics: Dict,
    conversation_history: list,
    question_number: int,
    total_questions: int = 5,
    assessment_type: str = "all",
    llm_mode: str = "gemini",
) -> str:
    """Generate the next interview question using the selected LLM backend."""

    # ── Local LLM path ──
    if llm_mode == "local":
        if not _ensure_local_llm():
            print(f"[LLM-LOCAL] Not available, falling back to hardcoded Q{question_number}.")
            return _hardcoded_question(question_number)

        print(f"[LLM-LOCAL] Building prompt for Q{question_number}/{total_questions}")
        prompt = _build_local_question_prompt(
            demographics, conversation_history,
            question_number, total_questions, assessment_type
        )
        try:
            question = _call_local_llm(prompt).strip('"\'')
            # Take only the first line if the model outputs extra text
            question = question.split("\n")[0].strip()
            if len(question) > 10:
                print(f"[LLM-LOCAL] Question generated: {question[:80]}")
                return question
            print(f"[LLM-LOCAL] Response too short — using fallback.")
        except Exception as e:
            print(f"[LLM-LOCAL] ERROR: {e}")
        return _hardcoded_question(question_number)

    # ── Gemini path (original logic, unchanged) ──
    if not is_llm_available():
        print(f"[LLM] LLM not available — falling back to hardcoded Q{question_number}.")
        return _hardcoded_question(question_number)

    print(f"[LLM] Building prompt for Q{question_number}/{total_questions} "
          f"(assessment={assessment_type}, history={len(conversation_history)} turns) ...")

    demo_ctx = format_demographics(demographics)

    conv_text = ""
    if conversation_history:
        conv_text = "\n\nConversation so far:\n"
        for i, entry in enumerate(conversation_history, 1):
            conv_text += f"Q{i}: {entry.get('question', 'N/A')}\n"
            conv_text += f"A{i}: {entry.get('answer', 'No response')}\n\n"

    focus_areas = [
        "their current emotional state and daily mood",
        "sleep patterns, energy levels, or fatigue",
        "social connections, loneliness, or support systems",
        "work/study stress, motivation, or concentration",
        "self-worth, future outlook, or things they enjoy",
    ]
    focus = focus_areas[min(question_number - 1, len(focus_areas) - 1)]

    assessment_label = {
        "depression": "depression",
        "anxiety": "anxiety",
        "stress": "stress",
        "all": "depression, anxiety, and stress",
    }.get(assessment_type, "mental health")

    prompt = f"""You are a compassionate mental health interviewer conducting a screening for {assessment_label}.

User Profile:
{demo_ctx}
{conv_text}

This is question {question_number} of {total_questions}. Focus on: {focus}

Generate ONE short, highly personalized follow-up question.

CRITICAL REQUIREMENTS:
1. Be VERY SHORT - 10-20 words maximum
2. NAME USAGE RULES:
   - If a name IS in the profile: use their first name in ONLY questions 1 and 3. The rest should NOT use the name.
   - If NO name is provided: NEVER guess or invent a name. Use "you".
   - Current question is #{question_number}.
3. Reference SPECIFIC details from their profile or previous answers
4. DO NOT ask generic questions - make it personal
5. Build on what they've shared before
6. Be warm and conversational, like a caring friend
7. Avoid clinical language

Return ONLY the question text, nothing else. No quotes, no explanation."""

    try:
        question = _call_gemini(prompt).strip('"\'')
        if len(question) > 10:
            print(f"[LLM] Question generated: {question[:80]}{'...' if len(question) > 80 else ''}")
            return question
        print(f"[LLM] Response too short ({len(question)} chars) — using fallback.")
    except Exception as e:
        print(f"[LLM] ERROR during question generation: {e}")
    return _hardcoded_question(question_number)


# ---------------------------------------------------------------
# Recommendation generation (dual-backend)
# ---------------------------------------------------------------

def generate_recommendation(
    scores: Dict[str, float],
    labels: Dict[str, str],
    demographics: Dict,
    assessment_type: str = "all",
    llm_mode: str = "gemini",
) -> Dict:
    """Generate a final recommendation/summary via the selected LLM backend."""

    # ── Local LLM path ──
    if llm_mode == "local":
        if not _ensure_local_llm():
            print(f"[LLM-LOCAL] Not available, using hardcoded recommendation.")
            return _hardcoded_recommendation(scores, labels, assessment_type)

        print(f"[LLM-LOCAL] Generating personalised recommendation ...")
        print(f"[LLM-LOCAL] Scores: {scores}  |  Labels: {labels}")
        prompt = _build_local_recommendation_prompt(
            scores, labels, demographics, assessment_type
        )
        try:
            text = _call_local_llm(prompt)
            print(f"[LLM-LOCAL] Parsing recommendation JSON ...")
            rec = _parse_json_response(text)

            # Validate and fill defaults
            rec.setdefault("summary", "Assessment complete.")
            rec.setdefault("recommendations", [])
            rec.setdefault("resources", [])
            rec.setdefault("encouragement", "Thank you for taking this step.")
            rec["llm_generated"] = True

            if not isinstance(rec["recommendations"], list):
                rec["recommendations"] = [str(rec["recommendations"])]
            if not isinstance(rec["resources"], list):
                rec["resources"] = [str(rec["resources"])]

            print(f"[LLM-LOCAL] Recommendation ready — summary length: {len(rec.get('summary',''))} chars, "
                  f"recommendations: {len(rec.get('recommendations',[]))}, resources: {len(rec.get('resources',[]))}")
            return rec
        except Exception as e:
            print(f"[LLM-LOCAL] ERROR generating recommendation: {e}")
            return _hardcoded_recommendation(scores, labels, assessment_type)

    # ── Gemini path (original logic, unchanged) ──
    if not is_llm_available():
        print(f"[LLM] LLM not available — using hardcoded recommendation.")
        return _hardcoded_recommendation(scores, labels, assessment_type)

    print(f"[LLM] Generating personalised recommendation ...")
    print(f"[LLM] Scores: {scores}  |  Labels: {labels}")

    demo_ctx = format_demographics(demographics)

    score_lines = []
    for key in ("depression", "anxiety", "stress"):
        if key in scores:
            score_lines.append(f"- {key.title()}: {scores[key]:.1f} ({labels.get(key, 'N/A')})")
    score_text = "\n".join(score_lines)

    prompt = f"""You are a compassionate, evidence-informed mental health advisor writing personalized recommendations after a screening.

User Profile:
{demo_ctx}

Assessment Type: {assessment_type}
Screening Results:
{score_text}

NAME RULES:
- If a name IS in the profile: use their first name ONCE in the summary or encouragement.
- If NO name: use "you" or general phrasing.

Return a JSON object:
{{
    "summary": "<2-3 sentences acknowledging results warmly, using their context>",
    "recommendations": [
        "<action 1: short, specific, doable this week>",
        "<action 2>",
        "<action 3>",
        "<action 4>"
    ],
    "resources": [
        "<resource 1: name + brief description>",
        "<resource 2>"
    ],
    "encouragement": "<1 warm sentence, personal, not generic>"
}}

Style:
- Each recommendation: 1 sentence, under 20 words, starts with a verb
- Use their specific context (role, stressors, living situation)
- For high severity: lead recommendations with professional help
- For low severity: keep tone light and affirming
- Resources should be concrete (apps, hotlines, campus services)

Return ONLY valid JSON — no markdown, no code fences, no extra text."""

    try:
        text = _call_gemini(prompt)
        print(f"[LLM] Parsing recommendation JSON ...")
        rec = _parse_json_response(text)
        rec["llm_generated"] = True
        print(f"[LLM] Recommendation ready — summary length: {len(rec.get('summary',''))} chars, "
              f"recommendations: {len(rec.get('recommendations',[]))}, resources: {len(rec.get('resources',[]))}")
        return rec
    except Exception as e:
        print(f"[LLM] ERROR generating recommendation: {e}")
        return _hardcoded_recommendation(scores, labels, assessment_type)


# ---------------------------------------------------------------
# Hardcoded fallbacks
# ---------------------------------------------------------------

HARDCODED_QUESTIONS = [
    "Hello! I'm here to chat with you today. How have you been feeling lately?",
    "Over the past two weeks, have you felt little interest or pleasure in doing things you usually enjoy?",
    "How has your sleep been? Have you had trouble falling asleep, staying asleep, or sleeping too much?",
    "Have you been feeling tired or having little energy lately?",
    "How do you feel about yourself lately? Have you been feeling bad about yourself?",
]


def _hardcoded_question(question_number: int) -> str:
    idx = min(question_number - 1, len(HARDCODED_QUESTIONS) - 1)
    return HARDCODED_QUESTIONS[idx]


HARDCODED_RECOMMENDATIONS = {
    "minimal": {
        "summary": "Your responses suggest minimal or no significant symptoms.",
        "recommendations": [
            "Continue maintaining your current healthy lifestyle habits",
            "Practice regular self-care activities that bring you joy",
            "Stay connected with friends and family",
            "Consider keeping a gratitude journal",
        ],
        "resources": [
            "Mindfulness apps like Headspace or Calm",
            "Regular physical exercise (30 mins, 3-5 times per week)",
        ],
        "encouragement": "You're doing well — keep it up!",
    },
    "mild": {
        "summary": "Your responses suggest some mild symptoms that may benefit from attention.",
        "recommendations": [
            "Talk to a trusted friend or family member about how you feel",
            "Establish a regular daily routine with consistent sleep",
            "Engage in physical activities you enjoy",
            "Practice deep breathing or meditation",
        ],
        "resources": [
            "Self-help books on cognitive behavioral techniques",
            "Online mental wellness programs",
        ],
        "encouragement": "It's great that you're paying attention to your wellbeing.",
    },
    "moderate": {
        "summary": "Your responses indicate moderate symptoms. Professional support is recommended.",
        "recommendations": [
            "Schedule an appointment with a mental health professional",
            "Speak with your primary care physician",
            "Set small, achievable daily goals",
            "Avoid making major life decisions while feeling this way",
        ],
        "resources": [
            "Licensed therapists or counselors",
            "Your campus / workplace counseling center",
        ],
        "encouragement": "Seeking help is a sign of strength, not weakness.",
    },
    "severe": {
        "summary": "Your responses indicate significant symptoms requiring professional attention.",
        "recommendations": [
            "Please seek professional help as soon as possible",
            "Contact a crisis helpline if you feel overwhelmed",
            "Stay connected with people — avoid being alone for long periods",
            "Tell someone you trust how you're feeling right now",
        ],
        "resources": [
            "Crisis Text Line: Text HOME to 741741",
            "iCall: 9152987821 (India)",
            "National Suicide Prevention Lifeline: 988 (US)",
        ],
        "encouragement": "This is treatable and help is available. You've taken an important first step today.",
    },
}


def _hardcoded_recommendation(scores, labels, assessment_type):
    worst = "minimal"
    for lbl in labels.values():
        if lbl in ("severe", "moderately_severe"):
            worst = "severe"
            break
        if lbl == "moderate" and worst != "severe":
            worst = "moderate"
        if lbl == "mild" and worst in ("minimal",):
            worst = "mild"
    rec = HARDCODED_RECOMMENDATIONS.get(worst, HARDCODED_RECOMMENDATIONS["minimal"]).copy()
    rec["llm_generated"] = False
    return rec

"""
Score interpretation + recommendation helpers.
Severity classification for each assessment type.
"""

from typing import Dict


# --- Depression (PHQ-like, 0-24) ---
def depression_severity(score: float) -> str:
    if score < 5:
        return "minimal"
    if score < 10:
        return "mild"
    if score < 15:
        return "moderate"
    if score < 20:
        return "moderately_severe"
    return "severe"


# --- Anxiety (regressor, 0-24 scale) ---
def anxiety_severity(score: float) -> str:
    if score < 5:
        return "minimal"
    if score < 10:
        return "mild"
    if score < 15:
        return "moderate"
    return "severe"


# --- Stress (probability 0-1) ---
def stress_severity(score: float) -> str:
    if score < 0.3:
        return "minimal"
    if score < 0.5:
        return "mild"
    if score < 0.7:
        return "moderate"
    return "severe"


SEVERITY_FN = {
    "depression": depression_severity,
    "anxiety": anxiety_severity,
    "stress": stress_severity,
}


def compute_labels(scores: Dict[str, float]) -> Dict[str, str]:
    """Given a scores dict, compute severity labels."""
    print(f"[RECOM] Computing severity labels for scores: {scores}")
    labels = {}
    for key, val in scores.items():
        fn = SEVERITY_FN.get(key)
        if fn:
            labels[key] = fn(val)
            print(f"[RECOM] {key}: score={val:.2f}  →  severity={labels[key].upper()}")
    print(f"[RECOM] Labels: {labels}")
    return labels

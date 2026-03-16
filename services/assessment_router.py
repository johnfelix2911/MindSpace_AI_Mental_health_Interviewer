"""
Assessment router – dispatches audio analysis to the correct pipeline(s)
based on the chosen assessment type.
Returns a unified response shape.
"""

from typing import Dict, Any

from pipelines.depression_pipeline import predict_depression
from pipelines.anxiety_pipeline import predict_anxiety
from pipelines.stress_pipeline import predict_stress


def analyze_audio(audio_path: str, assessment_type: str) -> Dict[str, Any]:
    """
    Run inference on *audio_path* for the given *assessment_type*.
    Returns:
        {
            "depression": {"score": float, ...} | None,
            "anxiety":    {"score": float, ...} | None,
            "stress":     {"score": float, ...} | None,
        }
    """
    import os
    print(f"\n{'='*50}")
    print(f"[ROUTER] ▶ Assessment type: {assessment_type.upper()}")
    print(f"[ROUTER] Audio file: {os.path.basename(audio_path)}")

    results: Dict[str, Any] = {}

    run_dep = assessment_type in ("depression", "all")
    run_anx = assessment_type in ("anxiety",   "all")
    run_str = assessment_type in ("stress",    "all")

    print(f"[ROUTER] Pipelines queued — "
          f"depression: {run_dep}  |  anxiety: {run_anx}  |  stress: {run_str}")

    # ── Depression ──
    if run_dep:
        print(f"[ROUTER] → Handing off to depression pipeline ...")
        try:
            results["depression"] = predict_depression(audio_path)
            print(f"[ROUTER] ✓ Depression complete — score: {results['depression']['score']}")
        except Exception as e:
            print(f"[ROUTER] ✗ Depression pipeline ERROR: {e}")
            results["depression"] = {"score": 5.0, "error": str(e)}

    # ── Anxiety ──
    if run_anx:
        print(f"[ROUTER] → Handing off to anxiety pipeline ...")
        try:
            results["anxiety"] = predict_anxiety(audio_path)
            print(f"[ROUTER] ✓ Anxiety complete — score: {results['anxiety']['score']}")
        except Exception as e:
            print(f"[ROUTER] ✗ Anxiety pipeline ERROR: {e}")
            results["anxiety"] = {"score": 5.0, "error": str(e)}

    # ── Stress ──
    if run_str:
        print(f"[ROUTER] → Handing off to stress pipeline ...")
        try:
            results["stress"] = predict_stress(audio_path)
            print(f"[ROUTER] ✓ Stress complete — score: {results['stress']['score']}")
        except Exception as e:
            print(f"[ROUTER] ✗ Stress pipeline ERROR: {e}")
            results["stress"] = {"score": 0.5, "error": str(e)}

    summary = {k: v.get("score") for k, v in results.items()}
    print(f"[ROUTER] All pipelines done. Score summary: {summary}")
    print(f"{'='*50}\n")
    return results

"""
Prepare ALL models required for offline inference.

This master script checks each model's cache status before downloading.
Models that are already cached locally are SKIPPED — only missing models
are downloaded.

Models managed:
  1. Depression SER  — jonatasgrosman/wav2vec2-large-xlsr-53-english (~1.2 GB)
  2. Stress Wav2Vec2  — torchaudio WAV2VEC2_BASE (~360 MB)
  3. Stress StudentNet — forwarder1121/voice-based-stress-recognition (~5 MB)
  4. Local LLM        — Qwen/Qwen2.5-0.5B-Instruct (~1 GB)

Local model files (already in models/ folder, no download needed):
  - models/depression/phq_xgb.pkl      — XGBoost depression predictor
  - models/anxiety/gbr_pipeline.joblib  — GBR anxiety pipeline

Usage:
    python scripts/prepare_offline_assets.py
"""

import os
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
BASE_DIR = SCRIPTS_DIR.parent


def run_script(name: str) -> bool:
    """Run a download script as a subprocess."""
    path = SCRIPTS_DIR / name
    print(f"\n{'=' * 60}")
    print(f"  Running {name}")
    print(f"{'=' * 60}\n")
    result = subprocess.run([sys.executable, str(path)])
    return result.returncode == 0


def check_local_models():
    """Verify that the local .pkl / .joblib model files exist."""

    print(f"\n{'=' * 60}")
    print("  Checking local model files")
    print(f"{'=' * 60}\n")

    dep_model = BASE_DIR / "models" / "depression" / "phq_xgb.pkl"
    anx_model = BASE_DIR / "models" / "anxiety" / "gbr_pipeline.joblib"

    ok = True

    if dep_model.exists():
        size_kb = dep_model.stat().st_size // 1024
        print(f"[OK]   Depression model: {dep_model.name} ({size_kb} KB)")
    else:
        print(f"[MISS] Depression model NOT found at: {dep_model}")
        print("       Copy phq_xgb.pkl from depression-ai-interviewer/")
        ok = False

    if anx_model.exists():
        size_kb = anx_model.stat().st_size // 1024
        print(f"[OK]   Anxiety model:    {anx_model.name} ({size_kb} KB)")
    else:
        print(f"[MISS] Anxiety model NOT found at: {anx_model}")
        print("       Copy gbr_pipeline.joblib from anxiety-ai-interviewer/ML/")
        ok = False

    return ok


if __name__ == "__main__":

    all_ok = True

    # 1. Check local model files first (no download, just existence check)
    if not check_local_models():
        all_ok = False

    # 2. Depression SER model (checks cache internally, skips if present)
    if not run_script("download_ser_model.py"):
        all_ok = False

    # 3. Stress models (checks cache internally, skips if present)
    if not run_script("download_stress_models.py"):
        all_ok = False

    # 4. Local LLM for offline question generation (checks cache, skips if present)
    if not run_script("download_local_llm.py"):
        all_ok = False

    # --- Summary ---
    print(f"\n{'=' * 60}")
    if all_ok:
        print("  All assets ready!")
        print()
        print("  Local models:")
        print("    models/depression/phq_xgb.pkl      — depression XGBoost")
        print("    models/anxiety/gbr_pipeline.joblib  — anxiety GBR pipeline")
        print()
        print("  Cached models (HuggingFace / torchaudio):")
        print("    wav2vec2-large-xlsr-53-english       — depression SER")
        print("    WAV2VEC2_BASE                        — stress embeddings")
        print("    voice-based-stress-recognition       — stress classifier")
        print("    Qwen/Qwen2.5-0.5B-Instruct          — local LLM")
        print()
        print("  You can now run:  python main.py")
    else:
        print("  Some models are missing. Check output above.")
    print(f"{'=' * 60}")

    sys.exit(0 if all_ok else 1)

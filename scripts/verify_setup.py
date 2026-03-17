"""
Verify that the integrated Mental Health AI Interviewer is ready to run.
Checks: imports, model files, ffmpeg, HuggingFace cache, Gemini config.

Usage:
    python scripts/verify_setup.py
"""

import sys
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OK = True


def check(label, condition, fix=""):
    global OK
    status = "OK" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        OK = False
        if fix:
            print(f"         Fix: {fix}")


def main():
    global OK

    print("=" * 60)
    print("  Mental Health AI Interviewer - Setup Verification")
    print("=" * 60)

    print("\n1. Core Python imports")
    for mod in ("fastapi", "uvicorn", "numpy", "librosa", "xgboost", "joblib", "pandas", "parselmouth", "soundfile"):
        try:
            __import__(mod)
            check(mod, True)
        except ImportError:
            check(mod, False, f"pip install {mod}")

    print("\n2. PyTorch & Transformers")
    try:
        import torch
        check(f"torch {torch.__version__}", True)
    except ImportError:
        check("torch", False, "pip install torch torchaudio")
    try:
        import transformers
        check(f"transformers {transformers.__version__}", True)
    except ImportError:
        check("transformers", False, "pip install transformers")
    try:
        import torchaudio
        check(f"torchaudio {torchaudio.__version__}", True)
    except ImportError:
        check("torchaudio", False, "pip install torchaudio")

    print("\n3. Gemini LLM")
    try:
        import google.generativeai
        check("google-generativeai installed", True)
    except ImportError:
        check("google-generativeai", False, "pip install google-generativeai")

    env_path = BASE / ".env"
    has_env = env_path.exists()
    check(".env file exists", has_env, "Create .env and set GEMINI_API_KEY")
    if has_env:
        text = env_path.read_text()
        has_key = "GEMINI_API_KEY" in text and text.split("GEMINI_API_KEY")[1].strip().startswith("=")
        check("GEMINI_API_KEY set in .env", has_key, "Add your key to .env")

    print("\n4. Model files")
    dep_model = BASE / "models" / "depression" / "phq_xgb.pkl"
    check(f"Depression model ({dep_model.name})", dep_model.exists(), "Copy phq_xgb.pkl from depression-ai-interviewer")
    anx_model = BASE / "models" / "anxiety" / "gbr_pipeline.joblib"
    check(f"Anxiety model ({anx_model.name})", anx_model.exists(), "Copy gbr_pipeline.joblib from anxiety-ai-interviewer")

    print("\n5. HuggingFace model cache")
    try:
        from huggingface_hub import try_to_load_from_cache
        ser_cached = try_to_load_from_cache("jonatasgrosman/wav2vec2-large-xlsr-53-english", "model.safetensors")
        check("Depression SER model cached", ser_cached is not None and isinstance(ser_cached, str), "python scripts/download_ser_model.py")
    except Exception:
        check("Depression SER model cached", False, "python scripts/download_ser_model.py")

    try:
        from huggingface_hub import try_to_load_from_cache
        stress_cached = try_to_load_from_cache("forwarder1121/voice-based-stress-recognition", "config.json")
        check("Stress StudentNet cached", stress_cached is not None and isinstance(stress_cached, str), "python scripts/download_stress_models.py")
    except Exception:
        check("Stress StudentNet cached", False, "python scripts/download_stress_models.py")

    print("\n6. System tools")
    ffmpeg = shutil.which("ffmpeg")
    check("ffmpeg in PATH", ffmpeg is not None, "Install ffmpeg: https://ffmpeg.org/download.html")

    print("\n7. Frontend files")
    for name in ("index.html", "demographic.html", "demographic.css", "demographic.js"):
        check(name, (BASE / name).exists())

    print("\n" + "=" * 60)
    if OK:
        print("  All checks PASSED. Ready to run:")
        print("    cd Mental-Health-ai-interviewer")
        print("    python main.py")
    else:
        print("  Some checks FAILED. Fix the issues above and re-run.")
    print("=" * 60)
    sys.exit(0 if OK else 1)


if __name__ == "__main__":
    main()

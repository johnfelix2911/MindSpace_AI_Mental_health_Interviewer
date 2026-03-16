"""
Depression inference pipeline.
Uses wav2vec2 SER → aggregate features → XGBoost PHQ predictor.
Ported from depression-ai-interviewer with clean module interface.
"""

import os
import pickle
import numpy as np
import librosa
import warnings

warnings.filterwarnings("ignore")

from utils.config import (
    DEPRESSION_MODEL_PATH, SER_MODEL_NAME, SER_LOAD_TIMEOUT, SAMPLE_RATE,
)

# ── SER pipeline (wav2vec2) ──────────────────────────────────

_ser_pipeline = None
_ser_pipeline_error: str | None = None
WINDOW_SEC = 4
HOP_SEC = 2

try:
    import torch
    from transformers.pipelines import pipeline as hf_pipeline
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False
    torch = None        # type: ignore
    hf_pipeline = None  # type: ignore


def _is_model_cached(model_name: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(model_name, "model.safetensors")
        if result is not None and not isinstance(result, str):
            return False
        return result is not None
    except ImportError:
        pass
    try:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        model_dir = os.path.join(cache_dir, "models--" + model_name.replace("/", "--"))
        snapshots = os.path.join(model_dir, "snapshots")
        return os.path.isdir(snapshots) and bool(os.listdir(snapshots))
    except Exception:
        return False


def _get_ser_pipeline():
    global _ser_pipeline, _ser_pipeline_error
    if _ser_pipeline_error:
        print(f"[DEPRESSION][SER] Skipping load — previous error: {_ser_pipeline_error}")
        return None
    if _ser_pipeline is not None:
        return _ser_pipeline
    if not _TORCH_AVAILABLE:
        _ser_pipeline_error = "Torch not available"
        print(f"[DEPRESSION][SER] ERROR — {_ser_pipeline_error}")
        return None
    print(f"[DEPRESSION][SER] Checking cache for: {SER_MODEL_NAME}")
    if not _is_model_cached(SER_MODEL_NAME):
        _ser_pipeline_error = "SER model not cached. Run: python scripts/download_ser_model.py"
        print(f"[DEPRESSION][SER] WARNING — {_ser_pipeline_error}")
        return None
    try:
        device = 0 if torch.cuda.is_available() else -1
        print(f"[DEPRESSION][SER] Loading SER pipeline on device={'GPU' if device == 0 else 'CPU'} ...")
        os.environ["HF_HUB_OFFLINE"] = "1"
        _ser_pipeline = hf_pipeline("audio-classification", model=SER_MODEL_NAME, device=device)
        os.environ.pop("HF_HUB_OFFLINE", None)
        print(f"[DEPRESSION][SER] Pipeline loaded successfully.")
    except Exception as e:
        os.environ.pop("HF_HUB_OFFLINE", None)
        _ser_pipeline_error = str(e)
        print(f"[DEPRESSION][SER] ERROR loading pipeline: {e}")
    return _ser_pipeline


def _extract_emotion_features(wav_path: str) -> np.ndarray:
    """Sliding-window SER → (num_chunks, num_labels) matrix."""
    print(f"[DEPRESSION][SER] Extracting emotion features from: {os.path.basename(wav_path)}")
    pipe = _get_ser_pipeline()
    y, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    duration = len(y) / SAMPLE_RATE
    print(f"[DEPRESSION][SER] Audio loaded — duration: {duration:.1f}s  sample_rate: {SAMPLE_RATE}")

    if pipe is None:
        print(f"[DEPRESSION][SER] No SER pipeline — returning fallback features.")
        return np.array([[0.6, 0.4]])

    window = int(WINDOW_SEC * SAMPLE_RATE)
    hop    = int(HOP_SEC * SAMPLE_RATE)
    all_scores = []
    chunk_count = 0

    for start in range(0, len(y) - window + 1, hop):
        chunk = y[start : start + window]
        results = pipe(chunk, sampling_rate=SAMPLE_RATE)
        all_scores.append([r["score"] for r in results])
        chunk_count += 1

    if not all_scores:
        print(f"[DEPRESSION][SER] Audio too short for windowing — running on full clip.")
        results = pipe(y, sampling_rate=SAMPLE_RATE)
        all_scores.append([r["score"] for r in results])
        chunk_count = 1

    mat = np.array(all_scores)
    print(f"[DEPRESSION][SER] Processed {chunk_count} chunk(s) → feature matrix shape: {mat.shape}")
    return mat


def _aggregate_features(mat: np.ndarray) -> np.ndarray:
    """mean + std + max across chunks → flat feature vector."""
    feat = np.concatenate([mat.mean(0), mat.std(0), mat.max(0)]).astype(np.float32)
    print(f"[DEPRESSION][FEAT] Aggregated feature vector length: {len(feat)}")
    return feat


# ── XGBoost PHQ model ────────────────────────────────────────

_model = None
_demo_mode = False


def load_model(path: str | None = None):
    global _model, _demo_mode
    path = path or DEPRESSION_MODEL_PATH
    print(f"[DEPRESSION][MODEL] Loading XGBoost model from: {path}")
    if not os.path.exists(path):
        _demo_mode = True
        print(f"[DEPRESSION][MODEL] WARNING — model file not found. Running in demo mode.")
        return
    with open(path, "rb") as f:
        _model = pickle.load(f)
    _demo_mode = False
    print(f"[DEPRESSION][MODEL] XGBoost model loaded successfully.")


def is_ser_cached() -> bool:
    return _is_model_cached(SER_MODEL_NAME)


def is_ser_loaded() -> bool:
    return _ser_pipeline is not None


def is_demo_mode() -> bool:
    return _demo_mode


# ── Public API ────────────────────────────────────────────────

def predict_depression(audio_path: str) -> dict:
    """
    Predict PHQ-8 depression score from an audio file.

    Returns:
        {"score": float, "label": str, "demo_mode": bool}
    """
    from utils.audio_common import ensure_wav

    print(f"\n{'─'*50}")
    print(f"[DEPRESSION] ▶ Starting depression inference")
    print(f"[DEPRESSION] Input audio: {os.path.basename(audio_path)}")

    wav_path = ensure_wav(audio_path)
    print(f"[DEPRESSION] WAV ready: {os.path.basename(wav_path)}")

    try:
        mat  = _extract_emotion_features(wav_path)
        feat = _aggregate_features(mat)

        if _model is not None:
            pred = float(np.clip(_model.predict(feat.reshape(1, -1))[0], 0, 24))
            print(f"[DEPRESSION][SCORE] XGBoost prediction: {pred:.2f}")
        else:
            pred = float(np.clip(np.mean(feat) * 24, 0, 24))
            print(f"[DEPRESSION][SCORE] Demo mode prediction: {pred:.2f}")

        from services.recommendations import depression_severity
        label = depression_severity(pred)
        print(f"[DEPRESSION][SCORE] Final score: {pred:.2f}  |  Severity: {label.upper()}")
        print(f"{'─'*50}\n")

        return {
            "score": round(pred, 2),
            "label": label,
            "demo_mode": _model is None,
        }
    finally:
        if wav_path != audio_path:
            try:
                os.unlink(wav_path)
                print(f"[DEPRESSION] Temp WAV cleaned up.")
            except Exception:
                pass

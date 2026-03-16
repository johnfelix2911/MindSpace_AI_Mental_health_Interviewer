"""
Anxiety inference pipeline.
Uses handcrafted audio features (MFCC + pitch + jitter via parselmouth)
→ GBR pipeline (scikit-learn joblib).
Ported from anxiety-ai-interviewer/ML/app.py with clean module interface.
"""

import os
import warnings
from typing import List

import joblib
import numpy as np
import pandas as pd
import librosa
import parselmouth
from parselmouth.praat import call

from utils.config import ANXIETY_MODEL_PATH, SAMPLE_RATE, ANXIETY_N_MFCC

warnings.filterwarnings("ignore")

# ── Model ─────────────────────────────────────────────────────

_model = None


def load_model(path: str | None = None):
    global _model
    path = path or ANXIETY_MODEL_PATH
    print(f"[ANXIETY][MODEL] Loading GBR pipeline from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Anxiety model not found: {path}")
    _model = joblib.load(path)
    print(f"[ANXIETY][MODEL] GBR pipeline loaded successfully.")


def _get_model():
    global _model
    if _model is None:
        print(f"[ANXIETY][MODEL] Model not loaded yet — loading on demand ...")
        load_model()
    return _model


def _get_feature_columns(model) -> List[str]:
    """Recover feature order from the sklearn pipeline."""
    if hasattr(model, "named_steps"):
        for _, step in model.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    raise RuntimeError("Cannot infer feature columns from the anxiety pipeline.")


# ── Feature extraction ────────────────────────────────────────

def _extract_audio_features(
    wav_path: str,
    feature_columns: List[str],
    sr: int = SAMPLE_RATE,
    n_mfcc: int = 13,
) -> pd.DataFrame:
    """
    Extract MFCC + delta + delta2 + pitch + jitter features from a WAV file.
    Returns a single-row DataFrame aligned to *feature_columns*.
    """
    print(f"[ANXIETY][FEAT] Loading audio from: {os.path.basename(wav_path)}")
    y, sr_loaded = librosa.load(wav_path, sr=sr, mono=True)
    duration = len(y) / sr_loaded
    print(f"[ANXIETY][FEAT] Audio loaded — duration: {duration:.1f}s  sr: {sr_loaded}  n_mfcc: {n_mfcc}")

    # MFCC + deltas
    print(f"[ANXIETY][FEAT] Computing MFCCs ({n_mfcc} coefficients) + delta + delta2 ...")
    mfcc       = librosa.feature.mfcc(y=y, sr=sr_loaded, n_mfcc=n_mfcc)
    mfcc_delta  = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # Pitch & jitter via Praat
    print(f"[ANXIETY][FEAT] Extracting pitch + jitter via Praat ...")
    snd        = parselmouth.Sound(wav_path)
    pitch      = snd.to_pitch()
    pitch_vals = pitch.selected_array["frequency"]
    pitch_vals = pitch_vals[pitch_vals > 0]

    pitch_mean = float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 0.0
    pitch_std  = float(np.std(pitch_vals))  if len(pitch_vals) > 0 else 0.0

    pp           = call(snd, "To PointProcess (periodic, cc)", 75, 500)
    jitter_local = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)

    print(f"[ANXIETY][FEAT] Pitch — mean: {pitch_mean:.1f} Hz  std: {pitch_std:.1f} Hz")
    print(f"[ANXIETY][FEAT] Jitter (local): {jitter_local:.6f}")

    # Aggregate statistics
    features = {}

    def add_stats(prefix: str, arr: np.ndarray):
        features[f"{prefix}_mean"] = float(np.mean(arr))
        features[f"{prefix}_std"]  = float(np.std(arr))

    for i in range(n_mfcc):
        add_stats(f"mfcc_{i+1}",       mfcc[i])
        add_stats(f"delta_mfcc_{i+1}", mfcc_delta[i])
        add_stats(f"delta2_mfcc_{i+1}", mfcc_delta2[i])

    features["pitch_mean"]   = pitch_mean
    features["pitch_std"]    = pitch_std
    features["jitter_local"] = float(jitter_local) if jitter_local is not None else 0.0

    row = [features.get(col, 0.0) for col in feature_columns]
    print(f"[ANXIETY][FEAT] Feature vector assembled — {len(row)} features aligned to model columns.")
    return pd.DataFrame([row], columns=feature_columns)


# ── Public API ────────────────────────────────────────────────

def predict_anxiety(audio_path: str) -> dict:
    """
    Predict anxiety score from an audio file.

    Returns:
        {"score": float, "label": str}
    """
    from utils.audio_common import ensure_wav

    print(f"\n{'─'*50}")
    print(f"[ANXIETY] ▶ Starting anxiety inference")
    print(f"[ANXIETY] Input audio: {os.path.basename(audio_path)}")

    model          = _get_model()
    feature_columns = _get_feature_columns(model)
    print(f"[ANXIETY][MODEL] Feature columns expected: {len(feature_columns)}")

    wav_path = ensure_wav(audio_path, sample_rate=SAMPLE_RATE)
    print(f"[ANXIETY] WAV ready: {os.path.basename(wav_path)}")

    try:
        X      = _extract_audio_features(wav_path, feature_columns, sr=SAMPLE_RATE, n_mfcc=ANXIETY_N_MFCC)
        y_pred = model.predict(X)
        pred   = float(np.clip(round(y_pred[0]), 0, 24))

        from services.recommendations import anxiety_severity
        label = anxiety_severity(pred)
        print(f"[ANXIETY][SCORE] GBR raw prediction: {y_pred[0]:.4f}  →  clipped: {pred}")
        print(f"[ANXIETY][SCORE] Final score: {pred:.1f}  |  Severity: {label.upper()}")
        print(f"{'─'*50}\n")

        return {
            "score": pred,
            "label": label,
        }
    finally:
        if wav_path != audio_path:
            try:
                os.unlink(wav_path)
                print(f"[ANXIETY] Temp WAV cleaned up.")
            except Exception:
                pass

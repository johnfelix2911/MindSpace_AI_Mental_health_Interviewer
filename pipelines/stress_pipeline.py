"""
Stress inference pipeline.
Uses Wav2Vec2-base CNN features (512-dim) → StudentNet classifier.
Ported from stress-ai-interviewer with clean module interface.
"""

import os
import warnings
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")

_detector = None
_init_error: str | None = None

_torch = None
_torchaudio = None


def _lazy_imports():
    global _torch, _torchaudio
    if _torch is None:
        import torch
        import torchaudio
        _torch = torch
        _torchaudio = torchaudio


# ── Embedding extractor ──────────────────────────────────────

class _EmbeddingExtractor:
    """Wav2Vec2-base CNN feature extractor → 512-dim embeddings."""

    def __init__(self, device):
        _lazy_imports()
        print(f"[STRESS][EMBED] Loading torchaudio WAV2VEC2_BASE ...")
        bundle = _torchaudio.pipelines.WAV2VEC2_BASE
        self.model = bundle.get_model().to(device)
        self.model.eval()
        self.sample_rate = bundle.sample_rate   # 16000
        self.device = device
        print(f"[STRESS][EMBED] WAV2VEC2_BASE loaded. Target sr: {self.sample_rate}")

    def extract(self, audio_path: str, aggregate: str = "mean"):
        _lazy_imports()
        print(f"[STRESS][EMBED] Loading audio: {os.path.basename(audio_path)}")
        waveform, sr = _torchaudio.load(audio_path)
        duration = waveform.shape[-1] / sr
        print(f"[STRESS][EMBED] Waveform shape: {tuple(waveform.shape)}  sr: {sr}  duration: {duration:.1f}s")

        if waveform.shape[0] > 1:
            print(f"[STRESS][EMBED] Downmixing stereo → mono")
            waveform = _torch.mean(waveform, dim=0, keepdim=True)
        if sr != self.sample_rate:
            print(f"[STRESS][EMBED] Resampling {sr} → {self.sample_rate} Hz ...")
            waveform = _torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

        waveform = waveform.to(self.device)
        print(f"[STRESS][EMBED] Running CNN feature extractor (aggregate={aggregate}) ...")

        with _torch.no_grad():
            embedding, _ = self.model.feature_extractor(waveform, length=None)
            if aggregate == "mean":
                embedding = _torch.mean(embedding, dim=1)
            elif aggregate == "max":
                embedding = _torch.max(embedding, dim=1)[0]
            elif aggregate == "first":
                embedding = embedding[:, 0, :]
            elif aggregate == "last":
                embedding = embedding[:, -1, :]

        if embedding.shape[0] == 1:
            embedding = embedding.squeeze(0)

        print(f"[STRESS][EMBED] Embedding shape: {tuple(embedding.shape)}")
        return embedding


# ── StudentNet wrapper ────────────────────────────────────────

class _StressClassifier:
    """Binary stressed / not-stressed classifier."""

    REPO_ID = "forwarder1121/voice-based-stress-recognition"

    def __init__(self, device):
        _lazy_imports()
        from transformers import AutoConfig, AutoModelForAudioClassification

        self.device = device
        print(f"[STRESS][CLASSIFIER] Loading StudentNet from HF: {self.REPO_ID} ...")
        self.config = AutoConfig.from_pretrained(self.REPO_ID, trust_remote_code=True)
        self.model  = AutoModelForAudioClassification.from_pretrained(
            self.REPO_ID, trust_remote_code=True, torch_dtype="auto",
        ).to(device)
        self.model.eval()
        print(f"[STRESS][CLASSIFIER] StudentNet loaded on {device}.")

    def predict(self, embedding):
        _lazy_imports()
        import torch.nn.functional as F
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        embedding = embedding.to(self.device)
        print(f"[STRESS][CLASSIFIER] Running softmax classifier ...")
        with _torch.no_grad():
            logits = self.model(embedding).logits
            probs  = F.softmax(logits, dim=-1)

        pred_idx       = _torch.argmax(probs, dim=-1).item()
        label          = "stressed" if pred_idx == 1 else "not_stressed"
        p_stressed     = probs[0, 1].item()
        p_not_stressed = probs[0, 0].item()

        print(f"[STRESS][CLASSIFIER] Probs — not_stressed: {p_not_stressed:.4f}  stressed: {p_stressed:.4f}")
        print(f"[STRESS][CLASSIFIER] Prediction: {label.upper()}  confidence: {probs[0, pred_idx].item():.4f}")

        return {
            "label":       label,
            "confidence":  probs[0, pred_idx].item(),
            "not_stressed": p_not_stressed,
            "stressed":    p_stressed,
        }


class _StressDetector:
    """Combines embedding extraction + classification."""

    def __init__(self, device: Optional[str] = None):
        _lazy_imports()
        if device is None:
            dev = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        else:
            dev = _torch.device(device)
        self.device = dev
        print(f"[STRESS] Initialising stress detector on: {dev} ...")
        self.extractor  = _EmbeddingExtractor(dev)
        self.classifier = _StressClassifier(dev)
        print(f"[STRESS] Stress detector fully ready.")

    def predict(self, audio_path: str, aggregate: str = "mean") -> dict:
        embedding = self.extractor.extract(audio_path, aggregate)
        return self.classifier.predict(embedding)


def _get_detector() -> _StressDetector:
    global _detector, _init_error
    if _init_error:
        print(f"[STRESS] Skipping — previous init error: {_init_error}")
        raise RuntimeError(_init_error)
    if _detector is None:
        print(f"[STRESS] Detector not loaded — initialising now ...")
        try:
            _detector = _StressDetector()
        except Exception as e:
            _init_error = str(e)
            print(f"[STRESS] ERROR during init: {e}")
            raise
    return _detector


def is_stress_model_ready() -> bool:
    try:
        _lazy_imports()
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(_StressClassifier.REPO_ID, "config.json")
        return result is not None and isinstance(result, str)
    except Exception:
        return False


# ── Public API ────────────────────────────────────────────────

def predict_stress(audio_path: str) -> dict:
    """
    Predict stress from an audio file.
    Returns: {"score": float, "label": str, "confidence": float}
    score = P(stressed), range [0, 1].
    """
    from utils.audio_common import ensure_wav

    print(f"\n{'─'*50}")
    print(f"[STRESS] ▶ Starting stress inference")
    print(f"[STRESS] Input audio: {os.path.basename(audio_path)}")

    wav_path = ensure_wav(audio_path)
    print(f"[STRESS] WAV ready: {os.path.basename(wav_path)}")

    try:
        det    = _get_detector()
        result = det.predict(wav_path)
        score  = round(result["stressed"], 4)
        label  = result["label"]
        conf   = round(result["confidence"], 4)

        print(f"[STRESS][SCORE] Final stress probability: {score:.4f}  |  Label: {label.upper()}  |  Confidence: {conf:.4f}")
        print(f"{'─'*50}\n")

        return {"score": score, "label": label, "confidence": conf}
    finally:
        if wav_path != audio_path:
            try:
                os.unlink(wav_path)
                print(f"[STRESS] Temp WAV cleaned up.")
            except Exception:
                pass

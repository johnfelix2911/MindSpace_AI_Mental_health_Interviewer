"""
Download and cache stress detection models:
  1. Wav2Vec2-base  (~360 MB, via torchaudio)
  2. StudentNet classifier (~5 MB, from HuggingFace)

This script checks if each model is ALREADY cached locally before
attempting any download.  If already present, it skips that model.

Cache locations:
  - Wav2Vec2-base: ~/.cache/torch/hub/checkpoints/  (torchaudio managed)
  - StudentNet:    ~/.cache/huggingface/hub/models--forwarder1121--voice-based-stress-recognition/

Usage:
    python scripts/download_stress_models.py
"""

import os
import sys


# ---------------------------------------------------------------------------
# Cache-check helpers
# ---------------------------------------------------------------------------

def is_studentnet_cached() -> bool:
    """
    Check whether the StudentNet stress classifier is already in
    the local HuggingFace cache.
    """
    repo = "forwarder1121/voice-based-stress-recognition"

    # Method 1: huggingface_hub API
    try:
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(repo, "config.json")
        if result is not None and isinstance(result, str):
            return True
    except ImportError:
        pass
    except Exception:
        pass

    # Method 2: check cache directory manually
    try:
        cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "hub"
        )
        model_dir = os.path.join(
            cache_dir, "models--" + repo.replace("/", "--")
        )
        snapshots = os.path.join(model_dir, "snapshots")
        if os.path.isdir(snapshots) and os.listdir(snapshots):
            return True
    except Exception:
        pass

    return False


def is_wav2vec2_base_cached() -> bool:
    """
    Check whether torchaudio's Wav2Vec2-base checkpoint is already
    downloaded.  torchaudio stores checkpoints under
    ~/.cache/torch/hub/checkpoints/ (or the TORCH_HOME equivalent).
    """
    try:
        # The filename torchaudio downloads for WAV2VEC2_BASE
        expected_file = "wav2vec2_fairseq_base_ls960_asr_ls960.pth"

        # Standard torch hub cache paths
        torch_home = os.environ.get(
            "TORCH_HOME",
            os.path.join(os.path.expanduser("~"), ".cache", "torch"),
        )
        checkpoint_dir = os.path.join(torch_home, "hub", "checkpoints")

        if os.path.isfile(os.path.join(checkpoint_dir, expected_file)):
            return True
    except Exception:
        pass

    return False


# ---------------------------------------------------------------------------
# Download functions (only called when NOT cached)
# ---------------------------------------------------------------------------

def download_wav2vec2_base() -> bool:
    """Download the Wav2Vec2-base model via torchaudio."""

    print("=== Wav2Vec2-base (torchaudio, ~360 MB) ===")
    print("Downloading...")
    try:
        import torchaudio

        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        model = bundle.get_model()  # triggers download if needed
        print(f"[OK] Wav2Vec2-base ready  (sample_rate={bundle.sample_rate})")
        del model
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def download_studentnet() -> bool:
    """Download the StudentNet stress classifier from HuggingFace."""

    repo = "forwarder1121/voice-based-stress-recognition"
    print(f"\n=== StudentNet ({repo}, ~5 MB) ===")
    print("Downloading...")
    try:
        from transformers import AutoConfig, AutoModelForAudioClassification

        AutoConfig.from_pretrained(repo, trust_remote_code=True)
        AutoModelForAudioClassification.from_pretrained(
            repo, trust_remote_code=True, torch_dtype="auto",
        )
        print("[OK] StudentNet classifier cached.")
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify() -> bool:
    """Load both stress models from cache to confirm they work."""

    print("\n=== Verification ===")
    try:
        import torch
        import torchaudio
        from transformers import AutoModelForAudioClassification

        # Wav2Vec2-base
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        _ = bundle.get_model()
        print("[OK] Wav2Vec2-base loads from cache.")

        # StudentNet
        _ = AutoModelForAudioClassification.from_pretrained(
            "forwarder1121/voice-based-stress-recognition",
            trust_remote_code=True, torch_dtype="auto",
        )
        print("[OK] StudentNet loads from cache.")

        print("[OK] All stress models verified.")
        return True
    except Exception as e:
        print(f"[WARN] Verification failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    all_ok = True

    # --- Wav2Vec2-base ---
    if is_wav2vec2_base_cached():
        print("[SKIP] Wav2Vec2-base already cached locally.")
    else:
        if not download_wav2vec2_base():
            all_ok = False

    # --- StudentNet ---
    if is_studentnet_cached():
        print("[SKIP] StudentNet stress classifier already cached locally.")
    else:
        if not download_studentnet():
            all_ok = False

    # --- Verify ---
    if all_ok:
        verify()

    sys.exit(0 if all_ok else 1)

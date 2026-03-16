"""
Download and cache the wav2vec2 SER model for depression inference.

This script checks if the model is ALREADY cached in the local
HuggingFace cache directory before attempting any download.
If already present, it skips the download and just verifies the model loads.

Model: jonatasgrosman/wav2vec2-large-xlsr-53-english (~1.2 GB)
Cache: ~/.cache/huggingface/hub/models--jonatasgrosman--wav2vec2-large-xlsr-53-english/

Usage:
    python scripts/download_ser_model.py
"""

import os
import sys

MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-english"


# ---------------------------------------------------------------------------
# Check if the model is already cached locally
# ---------------------------------------------------------------------------
def is_model_cached() -> bool:
    """
    Check whether the SER model has already been downloaded to the
    local HuggingFace cache.  Tries the huggingface_hub API first,
    then falls back to inspecting the cache directory on disk.
    """

    # Method 1: huggingface_hub API (most reliable)
    try:
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(MODEL_NAME, "model.safetensors")
        # Returns a file path string if cached, or a sentinel / None if not
        if result is not None and isinstance(result, str):
            return True
    except ImportError:
        pass  # huggingface_hub not installed; fall through
    except Exception:
        pass

    # Method 2: manually check the standard HF cache directory
    try:
        cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "hub"
        )
        model_dir = os.path.join(
            cache_dir, "models--" + MODEL_NAME.replace("/", "--")
        )
        snapshots = os.path.join(model_dir, "snapshots")
        if os.path.isdir(snapshots) and os.listdir(snapshots):
            return True
    except Exception:
        pass

    return False


# ---------------------------------------------------------------------------
# Download the model (only called when NOT already cached)
# ---------------------------------------------------------------------------
def download() -> bool:
    """Download the SER model from HuggingFace Hub."""

    print(f"Downloading SER model: {MODEL_NAME}")
    print("This is ~1.2 GB and may take several minutes...\n")

    # Try huggingface_hub snapshot_download first (supports resume)
    try:
        from huggingface_hub import snapshot_download
        cache_dir = snapshot_download(
            repo_id=MODEL_NAME,
            resume_download=True,
            max_workers=1,  # single thread = more stable on slow connections
        )
        print(f"\n[OK] Model cached at: {cache_dir}")
        return True
    except ImportError:
        print("[INFO] huggingface_hub not installed, trying transformers...")
    except Exception as e:
        print(f"[WARN] snapshot_download failed: {e}\nTrying alternative...\n")

    # Fallback: download via transformers API directly
    try:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        print("Downloading feature extractor...")
        AutoFeatureExtractor.from_pretrained(MODEL_NAME)

        print("Downloading model weights (this is the large file)...")
        AutoModelForAudioClassification.from_pretrained(MODEL_NAME)

        print(f"\n[OK] Model '{MODEL_NAME}' cached successfully.")
        return True
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Try a VPN if HuggingFace is slow in your region")
        print("  3. pip install huggingface_hub")
        return False


# ---------------------------------------------------------------------------
# Verify the model can be loaded from cache
# ---------------------------------------------------------------------------
def verify() -> bool:
    """Load the model from cache into a pipeline to confirm it works."""

    print("\nVerifying model loads from cache...")
    try:
        from transformers import pipeline as hf_pipeline

        pipe = hf_pipeline(
            "audio-classification", model=MODEL_NAME, device=-1  # CPU
        )
        print("[OK] SER model loads correctly from cache.")
        del pipe
        return True
    except Exception as e:
        print(f"[WARN] Verification failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # Step 1: Check if already cached
    if is_model_cached():
        print(f"[SKIP] SER model already cached locally: {MODEL_NAME}")
        print("       No download needed.\n")
        verify()
        sys.exit(0)

    # Step 2: Not cached — download it
    ok = download()
    if ok:
        verify()
    sys.exit(0 if ok else 1)

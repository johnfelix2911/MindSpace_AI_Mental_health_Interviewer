"""
Download and cache the local LLM for offline question generation.

Model: Qwen/Qwen2.5-0.5B-Instruct (~1 GB)
Cache: ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/

Usage:
    python scripts/download_local_llm.py
"""

import os
import sys

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


# ---------------------------------------------------------------------------
# Check if the model is already cached locally
# ---------------------------------------------------------------------------
def is_model_cached() -> bool:
    """Check whether the local LLM is already in the HuggingFace cache."""

    # Method 1: huggingface_hub API
    try:
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(MODEL_NAME, "config.json")
        if result is not None and isinstance(result, str):
            return True
    except ImportError:
        pass
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
    """Download the local LLM from HuggingFace Hub."""

    print(f"Downloading local LLM: {MODEL_NAME}")
    print("This is ~1 GB and may take several minutes...\n")

    # Try huggingface_hub snapshot_download first (supports resume)
    try:
        from huggingface_hub import snapshot_download
        cache_dir = snapshot_download(
            repo_id=MODEL_NAME,
            resume_download=True,
            max_workers=1,
        )
        print(f"\n[OK] Model cached at: {cache_dir}")
        return True
    except ImportError:
        print("[INFO] huggingface_hub not installed, trying transformers...")
    except Exception as e:
        print(f"[WARN] snapshot_download failed: {e}\nTrying alternative...\n")

    # Fallback: download via transformers API directly
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print("Downloading tokenizer...")
        AutoTokenizer.from_pretrained(MODEL_NAME)

        print("Downloading model weights...")
        AutoModelForCausalLM.from_pretrained(MODEL_NAME)

        print(f"\n[OK] Model '{MODEL_NAME}' cached successfully.")
        return True
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Try a VPN if HuggingFace is slow in your region")
        print("  3. pip install huggingface_hub transformers torch")
        return False


# ---------------------------------------------------------------------------
# Verify the model can be loaded from cache
# ---------------------------------------------------------------------------
def verify() -> bool:
    """Load the model from cache and run a quick test generation."""

    print("\nVerifying model loads from cache...")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32, device_map="cpu"
        )

        # Quick sanity test
        messages = [{"role": "user", "content": "Say hello in one sentence."}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=20)

        generated = output[0][inputs["input_ids"].shape[1]:]
        result = tokenizer.decode(generated, skip_special_tokens=True)
        print(f"[OK] Test generation: {result[:80]}")

        del model, tokenizer
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
        print(f"[SKIP] Local LLM already cached: {MODEL_NAME}")
        print("       No download needed.\n")
        verify()
        sys.exit(0)

    # Step 2: Not cached — download it
    ok = download()
    if ok:
        verify()
    sys.exit(0 if ok else 1)

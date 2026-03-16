"""
Shared audio utilities: format conversion, WAV normalisation.
Used by all three inference pipelines.
"""

import os
import subprocess
import tempfile
import numpy as np
import librosa
import warnings

warnings.filterwarnings("ignore")

from utils.config import SAMPLE_RATE

# FFmpeg discovery paths (Windows-friendly)
FFMPEG_PATHS = [
    "ffmpeg",
    r"C:\Users\SOUREN~1\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe",
    r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
    r"C:\ffmpeg\bin\ffmpeg.exe",
]


def find_ffmpeg() -> str | None:
    """Return path to a working ffmpeg binary, or None."""
    for path in FFMPEG_PATHS:
        try:
            if os.path.isfile(path):
                return path
            kwargs = {}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            result = subprocess.run(
                [path, "-version"], capture_output=True, timeout=5, **kwargs,
            )
            if result.returncode == 0:
                return path
        except Exception:
            continue
    return None


def convert_to_wav_ffmpeg(input_path: str, output_path: str,
                          sample_rate: int = SAMPLE_RATE) -> bool:
    """Convert any audio to mono WAV via ffmpeg. Returns True on success."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print(f"[AUDIO] ffmpeg not found — skipping ffmpeg conversion.")
        return False
    try:
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        cmd = [
            ffmpeg, "-y", "-i", input_path,
            "-ar", str(sample_rate), "-ac", "1", "-f", "wav", output_path,
        ]
        print(f"[AUDIO] ffmpeg: converting {os.path.basename(input_path)} → WAV @ {sample_rate}Hz ...")
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=30, **kwargs)
        success = result.returncode == 0 and os.path.exists(output_path)
        if success:
            size_kb = os.path.getsize(output_path) // 1024
            print(f"[AUDIO] ffmpeg conversion OK — output: {os.path.basename(output_path)} ({size_kb} KB)")
        else:
            print(f"[AUDIO] ffmpeg conversion FAILED (rc={result.returncode})")
        return success
    except Exception as e:
        print(f"[AUDIO] ffmpeg conversion exception: {e}")
        return False


def convert_to_wav_pydub(input_path: str, output_path: str,
                         sample_rate: int = SAMPLE_RATE) -> bool:
    """Fallback conversion via pydub."""
    try:
        from pydub import AudioSegment
        print(f"[AUDIO] pydub: converting {os.path.basename(input_path)} → WAV @ {sample_rate}Hz ...")
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(sample_rate).set_channels(1)
        audio.export(output_path, format="wav")
        size_kb = os.path.getsize(output_path) // 1024
        print(f"[AUDIO] pydub conversion OK — output: {os.path.basename(output_path)} ({size_kb} KB)")
        return True
    except Exception as e:
        print(f"[AUDIO] pydub conversion FAILED: {e}")
        return False


def ensure_wav(audio_path: str, sample_rate: int = SAMPLE_RATE) -> str:
    """
    Ensure *audio_path* is a 16 kHz mono WAV file.
    Returns path to a WAV file (original or a new temp file).
    Caller must clean up temp file when `returned_path != audio_path`.
    """
    ext = os.path.splitext(audio_path)[1].lower()
    print(f"[AUDIO] ensure_wav: input={os.path.basename(audio_path)}  ext={ext}  target_sr={sample_rate}")

    if ext == ".wav":
        try:
            _, sr = librosa.load(audio_path, sr=None, mono=False)
            if sr == sample_rate:
                print(f"[AUDIO] Already a {sample_rate}Hz WAV — no conversion needed.")
                return audio_path
            print(f"[AUDIO] WAV exists but sr={sr} ≠ target {sample_rate} — resampling ...")
        except Exception:
            pass

    tmp_wav = tempfile.mktemp(suffix=".wav")
    print(f"[AUDIO] Temp WAV target: {os.path.basename(tmp_wav)}")

    if convert_to_wav_ffmpeg(audio_path, tmp_wav, sample_rate):
        return tmp_wav

    print(f"[AUDIO] ffmpeg failed — trying pydub fallback ...")
    if convert_to_wav_pydub(audio_path, tmp_wav, sample_rate):
        return tmp_wav

    # Last resort: librosa
    print(f"[AUDIO] pydub failed — using librosa last-resort conversion ...")
    try:
        y, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
        import soundfile as sf
        sf.write(tmp_wav, y, sample_rate)
        size_kb = os.path.getsize(tmp_wav) // 1024
        print(f"[AUDIO] librosa conversion OK — {size_kb} KB")
        return tmp_wav
    except Exception as e:
        raise ValueError(f"Could not convert audio to WAV: {e}")


def load_audio_array(audio_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio file as a float32 mono numpy array at the given sample rate."""
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    return y.astype(np.float32)

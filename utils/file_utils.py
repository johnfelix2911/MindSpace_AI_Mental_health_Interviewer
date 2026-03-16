"""
File / temp-file helpers.
"""

import os
import tempfile


def save_upload_to_temp(content: bytes, suffix: str = ".webm") -> str:
    """Write upload bytes to a temporary file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
    except Exception:
        os.close(fd)
        raise
    return path


def cleanup(path: str) -> None:
    """Silently delete a file if it exists."""
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except Exception:
        pass


def ext_from_content_type(content_type: str, filename: str | None) -> str:
    """Derive a file extension from content-type or filename."""
    if filename:
        ext = os.path.splitext(filename)[1].lower()
        if ext:
            return ext

    ct_map = {
        "audio/webm": ".webm",
        "video/webm": ".webm",
        "audio/ogg": ".ogg",
        "audio/wav": ".wav",
        "audio/wave": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
    }
    return ct_map.get(content_type, ".webm")

"""Model downloader — fetches GGUF models from Hugging Face on first use."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)

# Default model: TinyLlama 1.1B Chat (GGUF Q4_K_M) — Apache 2.0 licensed
MODEL_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
MODEL_FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_URL = (
    f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILENAME}"
)
MODEL_SIZE_MB = 720  # approximate

# Fallback: Qwen2.5-0.5B-Instruct GGUF (Apache 2.0)
FALLBACK_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
FALLBACK_FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
FALLBACK_URL = (
    f"https://huggingface.co/{FALLBACK_REPO}/resolve/main/{FALLBACK_FILENAME}"
)
FALLBACK_SIZE_MB = 350


def _models_dir() -> Path:
    """Return the directory where downloaded models are cached."""
    p = Path.home() / ".local" / "share" / "momentum" / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_model_path(model_name: str = "tinyllama") -> Path:
    """Return the expected local path for the given model."""
    models_dir = _models_dir()
    if model_name == "tinyllama":
        return models_dir / MODEL_FILENAME
    elif model_name == "qwen":
        return models_dir / FALLBACK_FILENAME
    else:
        return models_dir / MODEL_FILENAME


def is_model_downloaded(model_name: str = "tinyllama") -> bool:
    """Check if the model file exists locally."""
    return get_model_path(model_name).exists()


def model_size_mb(model_name: str = "tinyllama") -> int:
    """Return approximate model size in MB."""
    if model_name == "qwen":
        return FALLBACK_SIZE_MB
    return MODEL_SIZE_MB


def _download_file(
    url: str,
    dest: Path,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Download *url* to *dest* with optional progress reporting."""
    log.info("Downloading %s to %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, dir=dest.parent) as tmp:
        tmp_path = Path(tmp.name)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Momentum/0.1"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    tmp.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and total > 0:
                        progress_callback(downloaded, total)
            shutil.move(tmp_path, dest)
            log.info("Download complete: %s", dest)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise


def ensure_model(
    model_name: str = "tinyllama",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Download the model if not already cached. Returns the local path."""
    path = get_model_path(model_name)
    if path.exists():
        log.debug("Model already cached at %s", path)
        return path

    if model_name == "qwen":
        url = FALLBACK_URL
    else:
        url = MODEL_URL

    _download_file(url, path, progress_callback)
    return path


def delete_model(model_name: str = "tinyllama") -> bool:
    """Remove a downloaded model from disk. Returns True if deleted."""
    path = get_model_path(model_name)
    if path.exists():
        path.unlink()
        log.info("Deleted model: %s", path)
        return True
    return False


def list_downloaded_models() -> list[dict[str, object]]:
    """List all downloaded GGUF models with metadata."""
    models_dir = _models_dir()
    if not models_dir.exists():
        return []
    results: list[dict[str, object]] = []
    for f in models_dir.iterdir():
        if f.suffix == ".gguf":
            results.append({
                "path": str(f),
                "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
            })
    return results
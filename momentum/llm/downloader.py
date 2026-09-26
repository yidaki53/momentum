"""Model downloader — fetches GGUF models from Hugging Face on first use.

Models are described by :data:`MODELS`, a registry of dataclass records rather
than loose module constants. Each entry lists one or more download sources, so
adding a mirror or a model does not mean editing branching logic.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from momentum.config import app_data_root
from momentum.ui.update_check import certifi_ssl_context

log = logging.getLogger(__name__)

_HF = "https://huggingface.co"


@dataclass(frozen=True)
class ModelSpec:
    """A downloadable GGUF model.

    Attributes:
        name: Stable identifier used in config (``llm_model``).
        filename: Name the file is cached under locally.
        size_mb: Approximate download size, for UI estimates only.
        license: SPDX-ish licence identifier shown in Settings.
        sources: ``(repo, filename)`` pairs. The first is the canonical
            source; any others are mirrors, used by later commits.
        sha256: Optional expected digest. Verified when present.
    """

    name: str
    filename: str
    size_mb: int
    license: str
    sources: tuple[tuple[str, str], ...]
    sha256: Optional[str] = None
    blurb: str = field(default="", compare=False)

    @property
    def url(self) -> str:
        """Canonical download URL (the first source)."""
        repo, filename = self.sources[0]
        return f"{_HF}/{repo}/resolve/main/{filename}"

    def mirror_urls(self) -> list[str]:
        """Every candidate URL, canonical first."""
        return [f"{_HF}/{repo}/resolve/main/{fn}" for repo, fn in self.sources]


# TinyLlama 1.1B Chat (GGUF Q4_K_M) - Apache 2.0, the default coach model.
_TINYLLAMA = ModelSpec(
    name="tinyllama",
    filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    size_mb=720,
    license="Apache-2.0",
    sources=(
        (
            "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        ),
        # Mirror maintained after TheBloke's repos were retired.
        (
            "bartowski/TinyLlama-1.1B-Chat-v1.0-GGUF",
            "TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf",
        ),
    ),
    blurb="TinyLlama 1.1B Chat - fast, well-tuned for its size.",
)

# Qwen2.5 0.5B Instruct GGUF (Apache 2.0) - smaller, for low-memory devices.
_QWEN = ModelSpec(
    name="qwen",
    filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
    size_mb=350,
    license="Apache-2.0",
    sources=(("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),),
    blurb="Qwen2.5 0.5B Instruct - smallest footprint.",
)

MODELS: dict[str, ModelSpec] = {
    _TINYLLAMA.name: _TINYLLAMA,
    _QWEN.name: _QWEN,
}

DEFAULT_MODEL = _TINYLLAMA.name

# Backwards-compatible module constants, kept because the UI and tests read
# these names directly. They are derived from the registry so the two cannot
# drift apart.
MODEL_REPO = _TINYLLAMA.sources[0][0]
MODEL_FILENAME = _TINYLLAMA.filename
MODEL_URL = _TINYLLAMA.url
MODEL_SIZE_MB = _TINYLLAMA.size_mb
FALLBACK_REPO = _QWEN.sources[0][0]
FALLBACK_FILENAME = _QWEN.filename
FALLBACK_URL = _QWEN.url
FALLBACK_SIZE_MB = _QWEN.size_mb


def get_spec(model_name: str) -> ModelSpec:
    """Return the :class:`ModelSpec` for *model_name*, defaulting sensibly.

    Unknown names fall back to the default model rather than raising: the
    coach is optional and must never take the app down over a stale config.
    """
    return MODELS.get(model_name, MODELS[DEFAULT_MODEL])


def available_models() -> list[ModelSpec]:
    """Return every model the app can download, for the Settings picker."""
    return list(MODELS.values())


def _models_dir() -> Path:
    """Return the directory where downloaded models are cached.

    On Android this lives under the app-private data root (so models survive
    updates and are not world-readable); on desktop it stays at the XDG-style
    ``~/.local/share/momentum/models`` path (unchanged behaviour).
    """
    p = app_data_root() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_model_path(model_name: str = "tinyllama") -> Path:
    """Return the expected local path for the given model."""
    return _models_dir() / get_spec(model_name).filename


def is_model_downloaded(model_name: str = "tinyllama") -> bool:
    """Check if the model file exists locally."""
    return get_model_path(model_name).exists()


def model_size_mb(model_name: str = "tinyllama") -> int:
    """Return approximate model size in MB."""
    return get_spec(model_name).size_mb


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
            with urllib.request.urlopen(
                req, timeout=30, context=certifi_ssl_context()
            ) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    tmp.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total)
            if total > 0 and downloaded != total:
                raise OSError(
                    f"Incomplete model download: {downloaded} of {total} bytes"
                )
            tmp.flush()
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

    _download_file(get_spec(model_name).url, path, progress_callback)
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
            results.append(
                {
                    "path": str(f),
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
                }
            )
    return results

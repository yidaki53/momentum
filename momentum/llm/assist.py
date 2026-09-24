"""Optional, non-blocking assistance built on the local AI Coach engine.

The rest of Momentum must remain useful when the model is absent or the user
has disabled the coach.  This module is therefore the single gate for optional
AI-generated snippets: callers get a cached, background-threaded result when
the coach is ready, and a clear fallback/error otherwise.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from momentum import config as cfg
from momentum.llm.context import build_user_context
from momentum.llm.downloader import is_model_downloaded
from momentum.llm.engine import get_engine, is_llm_available
from momentum.llm.prompts import SYSTEM_PROMPT

log = logging.getLogger(__name__)

# This cache is intentionally process-local.  Generated snippets are ephemeral
# UI copy; chat messages and the user's source data remain in SQLite.
_CACHE: dict[str, str] = {}
_CACHE_LOCK = threading.Lock()


def is_enabled(config: object | None = None) -> bool:
    """Return whether the user has enabled the optional coach."""
    conf = config if config is not None else cfg.load_config()
    return bool(getattr(conf, "llm_enabled", True))


def is_ready(config: object | None = None) -> bool:
    """Return whether an AI snippet can be generated without setup UI."""
    conf = config if config is not None else cfg.load_config()
    return bool(
        is_enabled(conf)
        and is_llm_available()
        and is_model_downloaded(getattr(conf, "llm_model", "tinyllama"))
    )


def status(config: object | None = None) -> dict[str, object]:
    """Return a UI-friendly, non-secret coach status."""
    conf = config if config is not None else cfg.load_config()
    enabled = is_enabled(conf)
    downloaded = is_model_downloaded(getattr(conf, "llm_model", "tinyllama"))
    return {
        "enabled": enabled,
        "engine": is_llm_available(),
        "downloaded": downloaded,
        "ready": enabled and is_llm_available() and downloaded,
    }


def clear_cache() -> None:
    """Forget generated optional snippets (used when settings change)."""
    with _CACHE_LOCK:
        _CACHE.clear()


def request_assistance(
    instruction: str,
    *,
    conn: object | None = None,
    config: object | None = None,
    cache_key: str | None = None,
    max_tokens: int = 180,
    temperature: float = 0.6,
    on_done: Callable[[str], None] | None = None,
    on_error: Callable[[Exception], None] | None = None,
) -> threading.Thread | None:
    """Generate a short personalised snippet in the background.

    ``on_done`` and ``on_error`` are invoked on the worker thread.  Mobile
    callers must marshal callbacks to Kivy's UI thread with ``Clock``.  A
    disabled/unready coach reports an error through ``on_error`` and does not
    touch the database or attempt a model download.
    """
    conf = config if config is not None else cfg.load_config()
    if not is_ready(conf):
        error = RuntimeError("The AI Coach is disabled or not ready.")
        if on_error:
            on_error(error)
        return None

    key = cache_key or instruction.strip()
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
    if cached is not None:
        if on_done:
            on_done(cached)
        return None

    context = ""
    if conn is not None:
        try:
            context = build_user_context(conn)  # type: ignore[arg-type]
        except Exception:
            log.debug("Could not build optional coach context", exc_info=True)
    user_prompt = instruction.strip()
    if context:
        user_prompt += f"\n\nRelevant Momentum data:\n{context}"

    def _run() -> None:
        try:
            engine = get_engine(getattr(conf, "llm_model", "tinyllama"))
            text = engine.generate(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            ).strip()
            if not text:
                raise RuntimeError("The coach returned an empty response.")
            with _CACHE_LOCK:
                _CACHE[key] = text
            if on_done:
                on_done(text)
        except Exception as exc:
            log.debug("Optional AI assistance failed", exc_info=True)
            if on_error:
                on_error(exc)

    thread = threading.Thread(target=_run, name="momentum-assist", daemon=True)
    thread.start()
    return thread

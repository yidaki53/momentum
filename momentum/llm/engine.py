"""LLM inference engine — wraps llama-cpp-python for local model inference.

The ``llama_cpp`` native dependency is imported lazily/guarded so this module
remains import-safe on builds where it is not available (e.g. the Android APK
when the optional llama-cpp-python recipe is absent). Use ``is_llm_available()``
to check at runtime; ``LlmEngine.load()`` raises a clear error if the native
backend is missing.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, Optional

try:  # Native dependency; absent on Android builds without the recipe.
    from llama_cpp import Llama

    LLM_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on Android/no-native builds
    Llama = None
    LLM_AVAILABLE = False

from momentum.llm.downloader import ensure_model

log = logging.getLogger(__name__)

_engine_instance: Optional[LlmEngine] = None
_engine_lock = threading.Lock()


def is_llm_available() -> bool:
    """Return True when the native llama-cpp-python backend is importable."""
    return LLM_AVAILABLE


class LlmEngine:
    """Manages a local GGUF model for text generation."""

    def __init__(
        self,
        model_path: Path,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_threads = n_threads or max(1, _guess_cpu_threads())
        self._verbose = verbose
        self._llama: Optional[Llama] = None
        self._lock = threading.Lock()

    def load(self) -> None:
        """Load the model into memory. Call once before generate()."""
        if self._llama is not None:
            return
        if not LLM_AVAILABLE or Llama is None:
            raise RuntimeError(
                "llama-cpp-python is not available on this build; "
                "the AI Coach UI is ready but on-device inference is not."
            )
        log.info(
            "Loading model %s (ctx=%d, threads=%d)",
            self._model_path,
            self._n_ctx,
            self._n_threads,
        )
        self._llama = Llama(
            model_path=str(self._model_path),
            n_ctx=self._n_ctx,
            n_threads=self._n_threads,
            verbose=self._verbose,
        )
        log.info("Model loaded successfully")

    def unload(self) -> None:
        """Unload the model from memory."""
        with self._lock:
            if self._llama is not None:
                # llama-cpp-python doesn't have an explicit unload,
                # but deleting the reference allows GC
                self._llama = None
                log.info("Model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._llama is not None

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        stop: Optional[list[str]] = None,
        stream: bool = False,
    ) -> str:
        """Generate a response from the model.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = deterministic).
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            stop: Optional list of stop sequences.
            stream: If True, use streaming (returns full text at end).

        Returns:
            The generated text.
        """
        if self._llama is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        with self._lock:
            response = self._llama.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop or [],
                stream=stream,
            )

        if stream:
            # For streaming, accumulate chunks
            full_text = ""
            for chunk in response:
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    full_text += content
            return full_text.strip()
        else:
            # ``response`` is Any (llama-cpp-python has no stubs); coerce to str
            # so the declared return type holds.
            return str(
                response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )

    def generate_async(
        self,
        messages: list[dict[str, str]],
        on_token: Callable[[str], None],
        on_done: Callable[[str], None],
        on_error: Callable[[Exception], None],
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> threading.Thread:
        """Generate a response asynchronously, streaming tokens to *on_token*.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            on_token: Called with each token string as it's generated.
            on_done: Called with the full response text when complete.
            on_error: Called with the exception if generation fails.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            The background thread.
        """

        def _run() -> None:
            try:
                full_text = ""
                if self._llama is None:
                    raise RuntimeError("Model not loaded. Call load() first.")

                with self._lock:
                    response = self._llama.create_chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True,
                    )

                for chunk in response:
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        full_text += content
                        on_token(content)

                on_done(full_text.strip())
            except Exception as exc:
                log.exception("Async generation failed")
                on_error(exc)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread


def _guess_cpu_threads() -> int:
    """Guess a reasonable number of CPU threads for inference."""
    import os

    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        pass
    try:
        import multiprocessing

        return multiprocessing.cpu_count()
    except NotImplementedError:
        return 4


def get_engine(
    model_name: str = "tinyllama",
    n_ctx: int = 2048,
    force_reload: bool = False,
) -> LlmEngine:
    """Get or create the singleton LLM engine.

    The model will be downloaded first if not already cached.
    """
    global _engine_instance

    with _engine_lock:
        if _engine_instance is not None and not force_reload:
            return _engine_instance

        model_path = ensure_model(model_name)
        _engine_instance = LlmEngine(
            model_path=model_path,
            n_ctx=n_ctx,
        )
        _engine_instance.load()
        return _engine_instance


# Re-exported for callers that only want to probe availability.
__all__ = [
    "LlmEngine",
    "get_engine",
    "reset_engine",
    "is_llm_available",
    "LLM_AVAILABLE",
]


def reset_engine() -> None:
    """Unload and reset the singleton engine."""
    global _engine_instance
    with _engine_lock:
        if _engine_instance is not None:
            _engine_instance.unload()
            _engine_instance = None

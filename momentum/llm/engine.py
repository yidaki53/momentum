"""LLM inference engine — wraps llama-cpp-python for local model inference.

The ``llama_cpp`` native dependency is imported lazily/guarded so this module
remains import-safe on builds where it is not available (e.g. the Android APK
when the optional llama-cpp-python recipe is absent). Use ``is_llm_available()``
to check at runtime; ``LlmEngine.load()`` raises a clear error if the native
backend is missing.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Optional, cast

from momentum.build_info import BUILD_VARIANT
from momentum.llm.downloader import ensure_model


def _android_native_dir() -> Optional[Path]:
    try:
        from jnius import autoclass  # type: ignore[import-not-found]

        activity = autoclass("org.kivy.android.PythonActivity").mActivity
        if activity is None:
            return None
        native_dir = activity.getApplicationInfo().nativeLibraryDir
        return Path(native_dir) if native_dir else None
    except Exception:
        log.debug("Could not locate Android nativeLibraryDir", exc_info=True)
        return None


def _preload_android_libraries(native_dir: Path) -> None:
    """Load llama/ggml dependencies globally before importing llama_cpp.

    ``llama_cpp.py`` uses ``LLAMA_CPP_LIB_PATH`` for libllama, while
    ``_ggml.py`` uses ``llama_cpp/lib`` for libggml.  On Android both copies
    can be present, but the dynamic loader still needs the dependency chain
    registered globally.  Loading the known names in dependency order avoids
    the intermittent ``undefined symbol``/``cannot locate`` import failure.
    """
    names: tuple[str, ...] = ("c++_shared", "ggml-base", "ggml-cpu", "ggml", "llama")
    if BUILD_VARIANT == "vulkan":
        names = (
            "c++_shared",
            "ggml-base",
            "ggml-cpu",
            "ggml-vulkan",
            "ggml",
            "llama",
        )
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    for name in names:
        path = native_dir / f"lib{name}.so"
        if not path.exists():
            continue
        try:
            ctypes.CDLL(str(path), mode=mode)
        except OSError as exc:
            log.debug("Could not preload %s: %s", path, exc)


def _prefer_android_native_lib_dir() -> None:
    """Configure llama-cpp-python's native loader on Android.

    The Python package contains its own ``llama_cpp/lib`` directory in the
    APK bundle. Prefer that complete directory for both llama and ggml, and
    preload the staged APK copies as dependencies. The APK native directory
    remains the fallback for builds that do not retain package-side libraries.

    Do not trust an inherited ``LLAMA_CPP_LIB_PATH`` blindly: p4a/launcher
    environments have been observed to provide a stale or incomplete path.
    Replace it on Android when the package-side directory is complete.
    """
    on_android = "ANDROID_ARGUMENT" in os.environ or hasattr(sys, "getandroidapilevel")
    if not on_android:
        return

    package_root = Path(__file__).resolve().parent.parent.parent / "llama_cpp" / "lib"
    package_complete = (package_root / "libllama.so").exists() and (
        package_root / "libggml.so"
    ).exists()
    existing = os.environ.get("LLAMA_CPP_LIB_PATH")
    if existing and Path(existing) == package_root and package_complete:
        return
    if package_complete:
        os.environ["LLAMA_CPP_LIB_PATH"] = str(package_root)
        _preload_android_libraries(package_root)
    else:
        native_dir = _android_native_dir()
        if native_dir is not None and (native_dir / "libllama.so").exists():
            os.environ["LLAMA_CPP_LIB_PATH"] = str(native_dir)
            _preload_android_libraries(native_dir)
    log.debug("llama native path configured: %s", os.environ.get("LLAMA_CPP_LIB_PATH"))


log = logging.getLogger(__name__)

_prefer_android_native_lib_dir()

LLM_IMPORT_ERROR = ""

# ``Llama`` is the llama-cpp-python class when the native backend imports, and
# ``None`` when it does not. Declaring it as ``Any`` lets both bindings
# type-check; the ``Optional[Any]`` annotations below are the consequence.
Llama: Any
try:  # Native loaders can raise several platform-specific exception types.
    from llama_cpp import Llama as _LlamaImpl

    Llama = _LlamaImpl
    LLM_AVAILABLE = True
except Exception as exc:  # keep the coach UI usable and expose the cause
    Llama = None
    LLM_AVAILABLE = False
    LLM_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def _first_choice(payload: Any) -> Optional[Mapping[str, Any]]:
    """Return the first choice mapping in a chat-completion payload, or None.

    llama-cpp-python returns a plain dict, but ``choices`` is missing on a
    malformed payload and empty when generation stops immediately, so callers
    must not index it unguarded.
    """
    if not isinstance(payload, Mapping):
        return None
    choices = payload.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
        return None
    if not choices:
        return None
    first = choices[0]
    return first if isinstance(first, Mapping) else None


def _message_text(response: Any) -> str:
    """Extract assistant text from a non-streaming completion.

    ``content`` is ``None`` when the model stops on a tool call or the context
    fills before emitting text, so it must be coerced rather than ``.strip()``ed.
    """
    choice = _first_choice(response)
    if choice is None:
        return ""
    message = choice.get("message")
    if not isinstance(message, Mapping):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _delta_text(chunk: Any) -> str:
    """Extract incremental text from a streaming chunk (``""`` when absent).

    The first chunk of a real llama-cpp-python stream carries only
    ``{"role": "assistant", "content": None}``; every other field is optional.
    """
    choice = _first_choice(chunk)
    if choice is None:
        return ""
    delta = choice.get("delta")
    if not isinstance(delta, Mapping):
        return ""
    content = delta.get("content")
    return content if isinstance(content, str) else ""


def native_diagnostics() -> dict[str, object]:
    """Return non-secret loader details for the Settings diagnostics panel."""
    package_root = Path(__file__).resolve().parent.parent.parent / "llama_cpp" / "lib"
    configured = os.environ.get("LLAMA_CPP_LIB_PATH")
    return {
        "available": LLM_AVAILABLE,
        "import_error": LLM_IMPORT_ERROR,
        "configured_path": configured or "",
        "package_path": str(package_root),
        "package_libllama": (package_root / "libllama.so").exists(),
        "package_libggml": (package_root / "libggml.so").exists(),
    }


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
        self._llama: Optional[Any] = None
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
            n_gpu_layers=-1 if BUILD_VARIANT == "vulkan" else 0,
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
                messages=cast("Any", messages),
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop or [],
                stream=stream,
            )

        if stream:
            # For streaming, accumulate chunks. ``_delta_text`` tolerates the
            # role-only and content-less chunks llama-cpp-python emits.
            full_text = ""
            for chunk in cast("Iterable[Any]", response):
                full_text += _delta_text(chunk)
            return full_text.strip()
        return _message_text(response).strip()

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
                        messages=cast("Any", messages),
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True,
                    )

                for chunk in cast("Iterable[Any]", response):
                    content = _delta_text(chunk)
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
    "native_diagnostics",
    "LLM_AVAILABLE",
]


def reset_engine() -> None:
    """Unload and reset the singleton engine."""
    global _engine_instance
    with _engine_lock:
        if _engine_instance is not None:
            _engine_instance.unload()
            _engine_instance = None

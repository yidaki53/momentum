"""LLM inference engine — wraps llama-cpp-python for local model inference.

The ``llama_cpp`` native dependency is imported lazily/guarded so this module
remains import-safe on builds where it is not available (e.g. the Android APK
when the optional llama-cpp-python recipe is absent). Use ``is_llm_available()``
to check at runtime; ``LlmEngine.load()`` raises a clear error if the native
backend is missing.
"""

from __future__ import annotations

import ctypes
import importlib.util
import logging
import os
import sys
import threading
import time
import traceback
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Optional, cast

from momentum.build_info import BUILD_VARIANT
from momentum.llm.downloader import ensure_model

log = logging.getLogger(__name__)


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


def _normalize_android_platform() -> None:
    """Coerce Android CPython's ``sys.platform`` to the Linux loader path.

    On Android, CPython sets ``sys.platform`` to ``"android"``. The upstream
    ``llama_cpp`` loader treats only the Linux/FreeBSD/Darwin/Windows branches as
    valid and raises ``RuntimeError("Unsupported platform")`` for Android even
    when the packaged libraries are present and correctly preloaded. Treating the
    Android runtime as Linux for this import guard is the minimal fix required to
    let the native library load proceed on-device.
    """
    if sys.platform == "android":
        sys.platform = "linux"
        log.debug("normalized sys.platform from android to linux for llama-cpp import")


_normalize_android_platform()


_PRELOAD_ORDER: tuple[str, ...] = (
    "c++_shared",
    "ggml-base",
    "ggml-cpu",
    "ggml-vulkan",
    "ggml",
    "llama",
)

# Diagnostics surfaced through ``native_diagnostics()``. Populated while this
# module imports, i.e. before any caller has a chance to install a log handler.
NATIVE_PRELOAD_FAILURES: list[str] = []
NATIVE_LIB_DIR: str = ""


def _package_lib_dir() -> Path:
    """Return the wheel-side directory llama-cpp-python ships libraries in.

    Resolving this relative to ``__file__`` is wrong on Android: the app's own
    package is unpacked from ``assets/private.tar`` into the app home, so
    ``momentum/llm/engine.pyc`` sits nowhere near llama-cpp, while the wheel
    lives in the bundled ``site-packages`` tree. Asking the import system is
    correct on both platforms. ``find_spec`` locates the module without
    executing it, so this cannot trigger the native load it exists to configure.
    """
    try:
        spec = importlib.util.find_spec("llama_cpp")
    except (ImportError, ValueError):  # pragma: no cover - exotic finders
        spec = None
    if spec is not None and spec.origin:
        return Path(spec.origin).resolve().parent / "lib"
    return Path(__file__).resolve().parent.parent.parent / "llama_cpp" / "lib"


def _preload_android_libraries(primary: Path, native_dir: Optional[Path]) -> list[str]:
    """Load the llama/ggml dependency chain globally before importing llama_cpp.

    ``llama_cpp.llama_cpp`` resolves ``libllama.so`` from
    ``LLAMA_CPP_LIB_PATH`` with a plain ``ctypes.CDLL`` of the absolute path,
    so every ``DT_NEEDED`` entry of that library must already be resolvable in
    this process' linker namespace. The libraries are loaded in dependency
    order with ``RTLD_GLOBAL`` so the linker finds them by soname instead of
    having to search for them.

    Each name is looked up in ``primary`` first and then in ``native_dir``,
    because ``libc++_shared.so`` is staged by p4a into the APK native dir only
    and never into the wheel's ``llama_cpp/lib``. Returns human-readable
    descriptions of the libraries that could not be loaded.
    """
    directories = [primary]
    if native_dir is not None and native_dir != primary:
        directories.append(native_dir)

    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    failures: list[str] = []
    for name in _PRELOAD_ORDER:
        candidates = [d / "lib{}.so".format(name) for d in directories]
        candidates = [p for p in candidates if p.exists()]
        if not candidates:
            continue  # legitimately absent, e.g. ggml-vulkan in a CPU build
        for path in candidates:
            try:
                ctypes.CDLL(str(path), mode=mode)
                break
            except OSError as exc:
                failures.append("{}: {}".format(path, exc))
                log.debug("Could not preload %s: %s", path, exc)
    return failures


def _prefer_android_native_lib_dir() -> None:
    """Configure llama-cpp-python's native loader on Android.

    The APK native library directory is preferred over the wheel's own
    ``llama_cpp/lib``. ``llama_cpp`` loads ``libllama.so`` with a single
    ``ctypes.CDLL`` of an absolute path, and that library's ``DT_NEEDED`` list
    contains the bare sonames ``libggml.so``, ``libggml-base.so``,
    ``libggml-cpu.so`` and ``libc++_shared.so`` while carrying no
    ``DT_RUNPATH``. Android's linker only resolves those bare sonames inside
    the app's native library directory, and ``libc++_shared.so`` -- staged
    there by the recipe's ``need_stl_shared`` -- is *never* copied into the
    wheel directory. Pointing the loader at the wheel directory therefore
    fails with ``dlopen failed: library "libc++_shared.so" not found`` even
    though every file appears to be present.

    The wheel directory stays as the fallback for builds that ship the native
    libraries only there. Do not trust an inherited ``LLAMA_CPP_LIB_PATH``
    blindly either: p4a launcher environments have been observed to provide a
    stale or incomplete path.
    """
    global NATIVE_LIB_DIR

    on_android = "ANDROID_ARGUMENT" in os.environ or hasattr(sys, "getandroidapilevel")
    if not on_android:
        return

    package_root = _package_lib_dir()
    native_dir = _android_native_dir()

    native_ok = native_dir is not None and (native_dir / "libllama.so").exists()
    package_ok = (package_root / "libllama.so").exists() and (
        package_root / "libggml.so"
    ).exists()

    if native_ok:
        chosen = native_dir
    elif package_ok:
        chosen = package_root
    else:
        chosen = None

    if chosen is not None:
        os.environ["LLAMA_CPP_LIB_PATH"] = str(chosen)
        NATIVE_LIB_DIR = str(chosen)

    # Preload from whichever directory is in play, falling back to the APK
    # native dir per library so libc++_shared.so is found either way.
    NATIVE_PRELOAD_FAILURES.extend(
        _preload_android_libraries(
            chosen if chosen is not None else package_root, native_dir
        )
    )
    log.debug(
        "llama native path configured: %s (preload failures: %d)",
        os.environ.get("LLAMA_CPP_LIB_PATH"),
        len(NATIVE_PRELOAD_FAILURES),
    )


def _instrument_event(msg: str) -> None:
    """Emit a concise, timestamped import-time message to the standard
    logger so it appears clearly in logcat when imports happen on-device.
    """
    try:
        ts = time.time()
        logging.getLogger("momentum.llm.imports").info(
            "[LLM-IMPORT] %s | ts=%.3f", msg, ts
        )
    except Exception:
        # Best-effort only; do not raise during import.
        pass


_instrument_event("prefer_android_native_lib_dir:start")
_prefer_android_native_lib_dir()
_instrument_event(
    f"prefer_android_native_lib_dir:done preload_failures={len(NATIVE_PRELOAD_FAILURES)} selected={NATIVE_LIB_DIR or 'unset'}"
)

LLM_IMPORT_ERROR = ""
LLM_IMPORT_TRACEBACK = ""

# ``Llama`` is the llama-cpp-python class when the native backend imports, and
# ``None`` when it does not. Declaring it as ``Any`` lets both bindings
# type-check; the ``Optional[Any]`` annotations below are the consequence.
Llama: Any
try:  # Native loaders can raise several platform-specific exception types.
    _instrument_event("llama_cpp:import:start")
    t0 = time.time()
    from llama_cpp import Llama as _LlamaImpl

    Llama = _LlamaImpl
    LLM_AVAILABLE = True
    _instrument_event(f"llama_cpp:import:ok elapsed={time.time() - t0:.3f}s")
except Exception as exc:  # keep the coach UI usable and expose the cause
    Llama = None
    LLM_AVAILABLE = False
    # The traceback is what actually identifies a native load failure: a bare
    # ``type: message`` drops the ``dlopen`` chain that caused it, and on a
    # release APK nothing else reaches logcat, so keep it for diagnostics.
    LLM_IMPORT_TRACEBACK = traceback.format_exc()
    LLM_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    _instrument_event(
        f"llama_cpp:import:fail elapsed={0:.3f}s error={LLM_IMPORT_ERROR}"
    )
    log.warning(
        "llama_cpp import failed (%s); native dir=%s preload failures=%s",
        LLM_IMPORT_ERROR,
        NATIVE_LIB_DIR or "unset",
        "; ".join(NATIVE_PRELOAD_FAILURES) or "none",
    )
    log.debug("llama_cpp import traceback:\n%s", LLM_IMPORT_TRACEBACK)


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
    package_root = _package_lib_dir()
    configured = os.environ.get("LLAMA_CPP_LIB_PATH")
    native_dir = _android_native_dir()
    native_root = Path(native_dir) if native_dir is not None else None
    return {
        "available": LLM_AVAILABLE,
        "variant": BUILD_VARIANT,
        "import_error": LLM_IMPORT_ERROR,
        "import_traceback": LLM_IMPORT_TRACEBACK,
        "configured_path": configured or "",
        "package_path": str(package_root),
        "package_libllama": (package_root / "libllama.so").exists(),
        "package_libggml": (package_root / "libggml.so").exists(),
        "apk_native_path": str(native_root) if native_root is not None else "",
        "apk_libllama": bool(
            native_root is not None and (native_root / "libllama.so").exists()
        ),
        "apk_libcxx_shared": bool(
            native_root is not None and (native_root / "libc++_shared.so").exists()
        ),
        "selected_dir": NATIVE_LIB_DIR,
        "preload_failures": list(NATIVE_PRELOAD_FAILURES),
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

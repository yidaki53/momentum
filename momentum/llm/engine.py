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


# ---------------------------------------------------------------------------
# Memory budget
#
# ``llama_cpp.Llama`` maps the whole GGUF into the process and then asks ggml for
# KV-cache and compute scratch on top of that. On a phone that is ~720 MB of
# weights plus a few hundred MB of buffers, and when the allocation fails the
# failure surfaces *inside* llama.cpp as a native SIGSEGV rather than a Python
# MemoryError. A segfault cannot be caught by ``try/except``, so it takes the
# whole app down with no message -- which is exactly the observed behaviour:
# the AI Coach tap on the help page killed Momentum outright.
#
# Estimating the footprint up front lets us refuse the load with an ordinary,
# catchable error the UI can render, instead of letting the device reap the app.
# The estimate is deliberately conservative; over-estimating only costs the user
# a clear message, while under-estimating costs them the app.
# ---------------------------------------------------------------------------

# ggml scratch space (activations, logits, copy buffers) measured on the GGUF
# sizes we ship. Roughly a fifth of the weights, which matches observed builds.
_COMPUTE_BUFFER_RATIO = 0.25

# KV cache is 2 tensors (keys + values) of n_ctx * n_layers * n_kv_heads *
# head_dim elements at 2 bytes (f16). 2 * 22 * 4 * 64 = 11264 bytes per token.
_KV_BYTES_PER_TOKEN = 2 * 22 * 4 * 64 * 2

# Headroom for the Python/Kivy process itself, the SQLite page cache, and the
# buffers ggml frees and re-requests during a decode step.
_RESIDENT_HEADROOM_MB = 160

# Devices with less free memory than this are not viable for on-device
# inference at all, whatever model is selected.
_MIN_HEADROOM_MB = 96


def _available_memory_mb() -> Optional[int]:
    """Return ``MemAvailable`` in MiB, or None when it cannot be determined.

    ``MemAvailable`` is the kernel's own estimate of memory that can be handed
    out without swapping, which is exactly the quantity that matters when the
    question is "can this process still claim another few hundred MB". Android
    exposes the same ``/proc/meminfo`` interface as Linux.
    """
    try:
        with open("/proc/meminfo", "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except Exception:  # unreadable/absent procfs, e.g. some sandboxes
        log.debug("Could not read /proc/meminfo", exc_info=True)
    return None


def estimate_load_mb(model_path: Path, n_ctx: int) -> int:
    """Return the peak RSS this load is expected to need, in MiB.

    Weights dominate: llama.cpp memory-maps the file, so the resident cost of
    the model itself is its size on disk. KV cache scales linearly with the
    context window, which is why a small ``n_ctx`` is the single cheapest lever
    on a memory-constrained phone.
    """
    try:
        weights_mb = Path(model_path).stat().st_size / (1024 * 1024)
    except OSError:
        weights_mb = 0.0
    kv_mb = max(0, n_ctx) * _KV_BYTES_PER_TOKEN / (1024 * 1024)
    compute_mb = weights_mb * _COMPUTE_BUFFER_RATIO
    return int(weights_mb + kv_mb + compute_mb + _RESIDENT_HEADROOM_MB)


class InsufficientMemoryError(RuntimeError):
    """Raised when loading the model would exceed the memory budget.

    A plain ``RuntimeError`` subclass so that every existing ``except
    Exception`` around the coach already handles it: the user gets a message
    instead of losing the app to a native crash.
    """


def check_memory_budget(model_path: Path, n_ctx: int) -> Optional[str]:
    """Return a user-facing refusal message, or None when the load can proceed.

    Returns the message rather than raising so that callers which only want to
    pre-flight (the settings screen, diagnostics) can inspect the verdict
    without handling an exception.
    """
    available_mb = _available_memory_mb()
    if available_mb is None:
        return None  # cannot measure; do not block a load that might well fit

    needed_mb = estimate_load_mb(model_path, n_ctx)
    headroom_mb = available_mb - needed_mb

    if headroom_mb < 0:
        return (
            f"The AI Coach needs about {needed_mb} MB to run "
            f"({Path(model_path).name}), but only {available_mb} MB of memory "
            f"is free on this device. Close other apps and free some space, or "
            f"choose the smaller Qwen model in Settings."
        )
    if headroom_mb < _MIN_HEADROOM_MB:
        # Technically loadable, but leaving the kernel this little room is what
        # produced the observed LOW_MEMORY kills mid-generation.
        return (
            f"Only about {available_mb} MB of memory is free and the AI Coach "
            f"needs roughly {needed_mb} MB. Loading it now would very likely "
            f"crash the app. Close other apps first, or switch to the smaller "
            f"Qwen model in Settings."
        )
    return None


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


def _default_n_ctx() -> int:
    """Return the context window to use for this platform.

    The window must comfortably exceed the prompt or ``llama_cpp`` refuses
    outright with "Requested tokens (...) exceed context window". Momentum's own
    prompts are large by design: ``CHAT_SYSTEM_PROMPT`` inlines the whole
    executive-dysfunction knowledge base and is ~1450 tokens on its own, before
    any user context or chat history.

    An earlier revision used 512 on Android to save KV-cache memory, which made
    *every* message fail -- the prompt alone never fit. The window is now 2048
    everywhere. Shrinking it further is not a useful lever: the cache at 2048 is
    only ~44 MB against a 720 MB model, so the saving is noise while the risk of
    silently rejecting the prompt is not. ``_fit_max_tokens`` handles the
    long-conversation case properly instead of by starving the window.
    """
    return 2048


# Tokens reserved for the model to finish its reply. Every request is capped at
# this share of the window so the prompt can never crowd out the answer.
_GENERATION_RESERVE = 256

# llama-cpp-python tokenises the prompt itself, so an exact count is not
# available before the call. Estimating from character length with a generous
# divisor keeps the estimate safely above the real token count, which is the
# direction that matters: over-estimating truncates the reply, while
# under-estimating raises "exceed context window".
_CHARS_PER_TOKEN_ESTIMATE = 3.0


def _estimate_prompt_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    """Return an upper-bound estimate of the prompt's token count.

    Deliberately pessimistic. llama-cpp-python formats chat messages with a
    template that adds per-message role tokens, and those cannot be counted
    without the tokenizer, so the estimate includes a small per-message
    allowance on top of the character-derived figure.
    """
    characters = 0
    for message in messages:
        content = message.get("content") if isinstance(message, Mapping) else None
        characters += len(content) if isinstance(content, str) else 0
    text_tokens = characters / _CHARS_PER_TOKEN_ESTIMATE
    # ~4 tokens per message covers the role markers llama-cpp adds.
    return int(text_tokens) + (len(messages) * 4) + 32


def _fit_max_tokens(
    n_ctx: int,
    messages: Sequence[Mapping[str, Any]],
    requested: int,
) -> int:
    """Clamp *requested* completion tokens so prompt + completion fit *n_ctx*.

    ``llama_cpp.Llama.create_chat_completion`` raises ``ValueError`` when the
    prompt alone fills the window, and the caller has no way to recover from
    that. Shrinking the requested completion here -- and, when even the minimum
    does not fit, dropping the oldest turns until it does -- is what makes the
    "Requested tokens (...) exceed context window" failure unreachable from
    Momentum's own code.
    """
    prompt_tokens = _estimate_prompt_tokens(messages)
    available = n_ctx - prompt_tokens

    if available <= 0:
        return 0  # caller reports a clear, actionable error

    return max(1, min(requested, available - 1))


# Windows the app offers. 2048 is the minimum that fits Momentum's prompts;
# larger values trade KV-cache memory for more history and longer answers.
MIN_CONTEXT_TOKENS = 2048
MAX_CONTEXT_TOKENS = 8192


def _configured_n_ctx() -> int:
    """Return the context window from config, clamped to a sane range.

    A local model can be given a bigger window than the desktop default, which
    is how the user gets more history and longer replies from the same weights.
    The value is clamped because a window below the prompt size makes every
    request fail, and an unbounded one can exhaust memory.
    """
    try:
        from momentum import config as cfg

        requested = int(getattr(cfg.load_config(), "llm_context_tokens", 0) or 0)
    except Exception:
        requested = 0
        log.debug("Could not read configured context window", exc_info=True)

    if requested <= 0:
        return _default_n_ctx()
    return max(MIN_CONTEXT_TOKENS, min(MAX_CONTEXT_TOKENS, requested))


def _short_repr(value: Any, limit: int = 160) -> str:
    """Return a compact repr of a streaming chunk, for on-device tracing."""
    try:
        text = repr(value)
    except Exception:  # pragma: no cover - exotic objects
        return "<unrepresentable>"
    return text if len(text) <= limit else text[:limit] + "..."


def _trace(message: str) -> None:
    """Emit a timestamped diagnostic line to stdout.

    On Android the APK's logging is not wired to a handler, so ``log.debug``
    vanishes and leaves nothing to diagnose a stalled generation with. ``print``
    reaches logcat on a debuggable build, which is what makes these traces
    usable in the field.
    """
    try:
        print(f"[LLM] {message}", flush=True)
    except Exception:  # never let diagnostics break generation
        pass


class LlmEngine:
    """Manages a local GGUF model for text generation."""

    def __init__(
        self,
        model_path: Path,
        n_ctx: Optional[int] = None,
        n_threads: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        self._model_path = model_path
        self._n_ctx = _default_n_ctx() if n_ctx is None else n_ctx
        self._n_threads = n_threads or max(1, _guess_cpu_threads())
        self._verbose = verbose
        self._llama: Optional[Any] = None
        self._lock = threading.Lock()
        # Completion tokens granted for the most recent request, published for
        # the UI's progress bar.
        self.last_budget: int = 0

    def load(self) -> None:
        """Load the model into memory. Call once before generate().

        The memory budget is checked *before* constructing ``Llama``: an
        allocation that llama.cpp cannot satisfy aborts natively with SIGSEGV,
        which no Python handler can intercept, so refusing up front is the only
        way to keep a failed load from taking the app down with it.
        """
        if self._llama is not None:
            return
        if not LLM_AVAILABLE or Llama is None:
            raise RuntimeError(
                "llama-cpp-python is not available on this build; "
                "the AI Coach UI is ready but on-device inference is not."
            )
        refusal = check_memory_budget(self._model_path, self._n_ctx)
        if refusal is not None:
            raise InsufficientMemoryError(refusal)
        log.info(
            "Loading model %s (ctx=%d, threads=%d)",
            self._model_path,
            self._n_ctx,
            self._n_threads,
        )
        _trace(
            f"loading {Path(self._model_path).name} n_ctx={self._n_ctx} "
            f"threads={self._n_threads}"
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

    def _prepare_request(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> tuple[list[dict[str, str]], int]:
        """Return messages and a completion budget that fit the context window.

        Momentum's prompts already fill most of the window, so a caller asking
        for 512 tokens would trip llama-cpp's "exceed context window" guard on
        every single request. Two adjustments keep that unreachable:

        1. the completion is clamped to whatever the prompt left behind, and
        2. when the prompt alone is too large, the oldest turns are dropped
           (never the system prompt) until it fits.

        Returning ``max_tokens == 0`` means even a trimmed prompt does not fit,
        which the caller turns into a message the user can act on.
        """
        budget = _fit_max_tokens(self._n_ctx, messages, max_tokens)
        trimmed = list(messages)

        while budget <= 0 and len(trimmed) > 1:
            # Drop the oldest turn after the system message (index 0).
            del trimmed[1]
            budget = _fit_max_tokens(self._n_ctx, trimmed, max_tokens)
            log.debug(
                "trimmed prompt to %d messages to fit n_ctx=%d",
                len(trimmed),
                self._n_ctx,
            )

        return trimmed, budget

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

        fitted_messages, budget = self._prepare_request(messages, max_tokens)
        if budget <= 0:
            raise RuntimeError(
                "This conversation is too long for the model's context window. "
                "Clear the chat and start again."
            )

        with self._lock:
            response = self._llama.create_chat_completion(
                messages=cast("Any", fitted_messages),
                max_tokens=budget,
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

    _GENERATION_LOCK_TIMEOUT_S = 120.0

    def _acquire_generation_slot(self) -> None:
        """Take the generation lock, or fail loudly instead of hanging.

        One non-reentrant lock serialises every generation, and it is held for
        the whole decode. A chat message and an automatic screen insight racing
        for it used to mean the loser waited in silence behind the winner, which
        is what left the chat screen stuck on "thinking" with nothing in the
        log. Waiting is bounded, and a refusal is reported, so the UI always
        recovers.
        """
        deadline = time.time() + self._GENERATION_LOCK_TIMEOUT_S
        while True:
            if self._lock.acquire(timeout=1.0):
                return
            if time.time() >= deadline:
                _trace("generation lock timeout - another request is stuck")
                raise RuntimeError(
                    "The coach is busy with another request that has not "
                    "finished. Wait a moment and try again."
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

                fitted_messages, budget = self._prepare_request(messages, max_tokens)
                # Published so the UI can show a determinate progress bar
                # against the budget actually granted, rather than the
                # requested one the prompt may have shrunk.
                self.last_budget = budget
                if budget <= 0:
                    raise RuntimeError(
                        "This conversation is too long for the model's context "
                        "window. Clear the chat and start again."
                    )

                # On-device tracing. The ``momentum`` logger is not configured
                # in the APK, so these prints are the only evidence available
                # when a generation stalls on a phone.
                _trace(
                    f"start n_ctx={self._n_ctx} threads={self._n_threads} "
                    f"budget={budget} messages={len(fitted_messages)}"
                )
                started = time.time()
                self._acquire_generation_slot()
                try:
                    response = self._llama.create_chat_completion(
                        messages=cast("Any", fitted_messages),
                        max_tokens=budget,
                        temperature=temperature,
                        stream=True,
                    )
                finally:
                    # The native call hands back a lazy generator, so the lock is
                    # released before any decoding happens. Holding it across the
                    # whole decode meant one slow generation blocked every other
                    # request behind it, silently.
                    self._lock.release()
                _trace(
                    f"create_chat_completion returned after {time.time() - started:.2f}s"
                )

                chunks = 0
                for chunk in cast("Iterable[Any]", response):
                    chunks += 1
                    if chunks == 1:
                        _trace(
                            f"first chunk after {time.time() - started:.2f}s: "
                            f"{type(chunk).__name__} {_short_repr(chunk)}"
                        )
                    content = _delta_text(chunk)
                    if content:
                        full_text += content
                        on_token(content)

                _trace(
                    f"stream complete: {chunks} chunks, {len(full_text)} chars, "
                    f"{time.time() - started:.2f}s"
                )
                on_done(full_text.strip())
            except Exception as exc:
                _trace(f"FAILED {type(exc).__name__}: {exc}\n" + traceback.format_exc())
                log.exception("Async generation failed")
                on_error(exc)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread


def _guess_cpu_threads() -> int:
    """Return a safe number of inference threads for this device.

    ggml parallelises with a spin barrier across its worker threads. On Android,
    and especially on an emulated CPU topology, asking for every reported core
    makes that barrier unreliable: the pool can livelock and the decode never
    returns. Capping the count keeps the threadpool well inside what the device
    actually schedules concurrently, which is the difference between a reply
    and a hang.

    The cap leaves headroom for the UI and Python threads rather than saturating
    every core with ggml workers.
    """
    import os

    available: int | None = None
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        try:
            import multiprocessing

            available = multiprocessing.cpu_count()
        except NotImplementedError:
            available = 4

    on_android = "ANDROID_ARGUMENT" in os.environ or hasattr(sys, "getandroidapilevel")
    cap = 4 if on_android else max(1, (available or 4))
    return max(1, min(available or cap, cap))


def get_engine(
    model_name: str = "tinyllama",
    n_ctx: Optional[int] = None,
    force_reload: bool = False,
) -> LlmEngine:
    """Get or create the singleton LLM engine.

    ``n_ctx`` defaults to the user's configured window (see
    ``AppConfig.llm_context_tokens``) and falls back to
    :func:`_default_n_ctx` when unset. A larger window buys longer chat history
    and longer answers at the cost of KV-cache memory, which the memory budget
    in :func:`check_memory_budget` accounts for -- so raising it is a real
    trade-off rather than a free win.

    Callers must run this off the UI thread. It performs the whole native load,
    which takes seconds and claims hundreds of megabytes; doing it inline in a
    Kivy callback freezes the main looper and leaves the screen showing its
    "thinking" state with no way to redraw or react.
    """
    global _engine_instance

    with _engine_lock:
        if _engine_instance is not None and not force_reload:
            return _engine_instance

        model_path = ensure_model(model_name)
        if n_ctx is None:
            n_ctx = _configured_n_ctx()
        _engine_instance = LlmEngine(
            model_path=model_path,
            n_ctx=n_ctx,
        )
        _engine_instance.load()
        return _engine_instance


# Re-exported for callers that only want to probe availability.
__all__ = [
    "LlmEngine",
    "InsufficientMemoryError",
    "check_memory_budget",
    "estimate_load_mb",
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

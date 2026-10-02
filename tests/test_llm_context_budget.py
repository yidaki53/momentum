"""Regression tests for the context-window budget.

Sending "hi" to the coach on Android failed with "Requested tokens (...)
exceed context window". The cause was a 512-token context window introduced to
save KV-cache memory, while ``CHAT_SYSTEM_PROMPT`` inlines the entire
executive-dysfunction knowledge base and is ~1450 tokens by itself. llama-cpp
rejects a request when the prompt alone fills the window, so every message
failed regardless of its length.

These tests pin the window size and the budget arithmetic that keeps the
failure unreachable.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

import momentum.llm.engine as engine_mod
from momentum.llm import prompts
from momentum.llm.engine import LlmEngine, _estimate_prompt_tokens, _fit_max_tokens


class _SlowFakeLlama:
    """A stand-in for ``Llama`` that yields chunks a real device would.

    Each chunk costs a little time, standing in for a token of decoding, and the
    counter records how far a generation got so a test can prove it stopped
    early instead of running to its full budget.
    """

    def __init__(self, chunks: int = 180, delay: float = 0.01) -> None:
        self._chunks = chunks
        self._delay = delay
        self.chunks_consumed = 0

    def create_chat_completion(self, **_kwargs):
        def _stream():
            for _ in range(self._chunks):
                time.sleep(self._delay)
                self.chunks_consumed += 1
                yield {
                    "choices": [{"delta": {"content": "word "}, "finish_reason": None}]
                }

        return _stream()


def _chat_messages(history_turns: int = 6) -> list[dict[str, str]]:
    """Build the same message list the coach screen sends."""
    history = [
        {"role": "user", "content": "I could not start my report today."},
        {"role": "assistant", "content": "That is common. Try two minutes."},
    ] * (history_turns // 2)
    return prompts.build_chat_prompt("hi", "Task focus: 25 minutes.", history)


def test_context_window_fits_the_system_prompt() -> None:
    """The bare system prompt plus a greeting must fit the default window.

    This is the exact regression: at 512 tokens the prompt alone overflowed, so
    every message was rejected before it was read.
    """
    n_ctx = engine_mod._default_n_ctx()
    messages = [
        {"role": "system", "content": prompts.CHAT_SYSTEM_PROMPT},
        {"role": "user", "content": "hi"},
    ]

    assert _estimate_prompt_tokens(messages) < n_ctx, (
        "the system prompt alone must fit inside the context window"
    )


def test_context_window_is_not_starved_on_android(monkeypatch) -> None:
    """Android must not use a window smaller than the prompts require.

    The 512-token setting looked like a memory saving but made the coach
    unusable, so the window is now the same on every platform.
    """
    monkeypatch.setenv("ANDROID_ARGUMENT", "1")
    assert engine_mod._default_n_ctx() == 2048


def test_real_chat_request_gets_a_usable_budget() -> None:
    """A realistic conversation yields a budget instead of a hard failure."""
    engine = LlmEngine(model_path=Path("/nonexistent.gguf"), n_ctx=2048)
    messages = _chat_messages()

    fitted, budget = engine._prepare_request(messages, max_tokens=512)

    assert budget > 0, "a normal conversation must not be rejected"
    assert _estimate_prompt_tokens(fitted) + budget <= 2048


def test_budget_never_exceeds_the_requested_amount() -> None:
    """A short prompt keeps the full requested completion."""
    engine = LlmEngine(model_path=Path("/nonexistent.gguf"), n_ctx=2048)
    _, budget = engine._prepare_request(
        [{"role": "user", "content": "hi"}], max_tokens=256
    )

    assert budget == 256


def test_overlong_conversation_trims_oldest_turns() -> None:
    """A huge history is trimmed, keeping the system prompt and newest turn."""
    engine = LlmEngine(model_path=Path("/nonexistent.gguf"), n_ctx=2048)
    messages = [
        {"role": "system", "content": prompts.CHAT_SYSTEM_PROMPT},
        *({"role": "user", "content": "x" * 2000} for _ in range(8)),
        {"role": "user", "content": "hi"},
    ]

    fitted, budget = engine._prepare_request(messages, max_tokens=512)

    assert len(fitted) < len(messages), "old turns must be dropped"
    assert budget > 0
    assert fitted[0]["content"] == prompts.CHAT_SYSTEM_PROMPT, "system prompt kept"
    assert fitted[-1]["content"] == "hi", "newest message kept"


def test_fit_max_tokens_reports_zero_when_nothing_fits() -> None:
    """A prompt that cannot fit yields zero so the caller can explain."""
    huge = [{"role": "user", "content": "x" * 500_000}]

    assert _fit_max_tokens(2048, huge, 512) == 0


@pytest.mark.parametrize("n_ctx", [2048])
def test_prompt_plus_budget_never_exceeds_window(n_ctx: int) -> None:
    """The invariant that makes the llama-cpp error unreachable."""
    for turns in (0, 2, 6, 20):
        messages = _chat_messages(history_turns=max(2, turns))
        engine = LlmEngine(model_path=Path("/nonexistent.gguf"), n_ctx=n_ctx)
        fitted, budget = engine._prepare_request(messages, max_tokens=512)
        if budget > 0:
            assert _estimate_prompt_tokens(fitted) + budget <= n_ctx


# ---------------------------------------------------------------------------
# Generation concurrency
# ---------------------------------------------------------------------------


def test_inference_is_single_flight() -> None:
    """Only one generation may run at a time, and losers are told.

    Screen insights fire automatically, so without a process-wide guard a
    background request can hold the engine while a chat message waits behind
    it -- the "stuck on thinking" symptom, with nothing in the log.
    """
    import sys
    import threading

    engine_mod = sys.modules["momentum.llm.engine"]

    # Take the slot in this thread. (Do not "reset" it by releasing first: an
    # uncontended Semaphore(1) released once has a count of 2, which would let
    # the contender through.)
    engine_mod._acquire_inference_slot()
    errors: list[Exception] = []

    def _contender():
        try:
            engine_mod._acquire_inference_slot()
        except Exception as exc:  # noqa: BLE001 - the point is that it raises
            errors.append(exc)

    original_wait = engine_mod.INFERENCE_WAIT_S
    engine_mod.INFERENCE_WAIT_S = 0.2
    try:
        thread = threading.Thread(target=_contender, daemon=True)
        thread.start()
        thread.join(10)
    finally:
        engine_mod.INFERENCE_WAIT_S = original_wait
        engine_mod._release_inference_slot()

    assert len(errors) == 1
    assert "already working" in str(errors[0])


def test_both_generation_paths_take_the_single_slot() -> None:
    """Chat (streaming) and insights (sync) must not bypass the guard."""
    import inspect
    import sys

    engine_mod = sys.modules["momentum.llm.engine"]
    for name in ("generate", "generate_async"):
        source = inspect.getsource(getattr(engine_mod.LlmEngine, name))
        assert "_acquire_inference_slot(" in source, name
        assert "_release_inference_slot()" in source, name


def test_android_thread_count_is_capped(monkeypatch) -> None:
    """ggml's spin barrier can livelock when given every reported core."""
    import sys

    engine_mod = sys.modules["momentum.llm.engine"]
    monkeypatch.setenv("ANDROID_ARGUMENT", "1")
    monkeypatch.setattr(engine_mod.os, "sched_getaffinity", lambda _pid: set(range(16)))

    assert engine_mod._guess_cpu_threads() <= 4


def test_chat_system_prompt_is_small_enough_for_a_phone() -> None:
    """The chat prompt must stay far below the context window.

    Inlining the whole knowledge base cost ~1750 tokens per turn: prefill alone
    took 10.8s on device, every decoded token attended over ~1900 tokens, and
    the reply budget collapsed to 126 tokens -- so the user saw a spinner
    instead of an answer. A system prompt that dominates the window is a
    performance bug, not a safety margin.
    """
    from momentum.llm import prompts

    tokens = _estimate_prompt_tokens(
        [{"role": "system", "content": prompts.CHAT_SYSTEM_PROMPT}]
    )
    assert tokens < 700, f"chat system prompt is {tokens} tokens; too large"

    # And it must leave room in a 2048 window for an actual conversation.
    headroom = 2048 - tokens
    assert headroom > 1000, "not enough room left for history and the reply"


def test_user_data_is_labelled_as_data_not_instructions() -> None:
    """The app-data block must be fenced and marked as data.

    Delivered as an unlabelled block of app formatting, the small model on the
    phone read it as an instruction and narrated the template back instead of
    replying: "hi" came back as "[This user is now creating a task ... they
    will write ... such as "Todya's Focus: [task]"]", inventing a label that
    exists nowhere in the codebase.
    """
    from momentum.llm import prompts

    block = prompts.build_chat_prompt("hi", "Active task: Draft intro", [])
    data = next(m for m in block if "Draft intro" in m["content"])

    assert data["content"].startswith("This is the user's app data.")
    assert "not instructions" in data["content"]
    assert "--- begin user data ---" in data["content"]
    assert "--- end user data ---" in data["content"]


def test_chat_system_prompt_forbids_narration() -> None:
    from momentum.llm import prompts

    prompt = prompts.CHAT_SYSTEM_PROMPT.lower()
    assert "square brackets" in prompt
    assert "reply only to what the user just said" in prompt


def test_chat_temperature_is_low_enough_for_small_models() -> None:
    """At 0.7 the smallest models drift into narration and invented formats."""
    import inspect

    from momentum.llm.engine import LlmEngine

    default = inspect.signature(LlmEngine.generate_async).parameters["temperature"]
    assert default.default <= 0.5, default.default
    """Compacting the prompt must not drop safety or behaviour rules.

    The task marker is covered separately in tests/test_coach_tasks.py, where
    it is deliberately kept out of the standing prompt.
    """
    from momentum.llm import prompts

    prompt = prompts.CHAT_SYSTEM_PROMPT.lower()
    for required in (
        "executive dysfunction",
        "never diagnose",
        "not a replacement for professional help",
    ):
        assert required in prompt, required


def _engine_with(fake) -> "LlmEngine":
    """Build an engine around a fake llama, bypassing the real model load."""
    import threading

    engine = object.__new__(LlmEngine)
    engine._llama = fake
    engine._model_path = Path("unused")
    engine._n_ctx = 2048
    engine._n_threads = 2
    engine._verbose = False
    engine._lock = threading.Lock()
    engine.last_budget = 0
    return engine


def test_background_generation_yields_the_slot_to_a_chat_reply() -> None:
    """A chat reply must not queue behind a long background snippet.

    Observed on the phone: a screen insight took the inference slot at 22:36:57
    for its 180-token budget, and 20 seconds later the user's "hi" was rejected
    with "inference slot busy - declining this request". Background work now
    checks between tokens and abandons itself so the reply goes through.
    """
    import threading

    fake = _SlowFakeLlama()
    engine = _engine_with(fake)
    # A chat arrives a moment into the snippet, as it would on a real screen.
    threading.Timer(0.15, engine_mod._FOREGROUND_WAITING.set).start()
    started = time.time()

    with pytest.raises(engine_mod.InferencePreempted):
        engine.generate(
            [{"role": "user", "content": "hi"}], max_tokens=180, background=True
        )
    # It stopped near the start rather than decoding all 180 chunks (~1.8s).
    assert fake.chunks_consumed < 60, fake.chunks_consumed
    assert time.time() - started < 1.0
    # And the slot is free again for whoever comes next.
    assert engine_mod._INFERENCE_SLOT.acquire(timeout=1)
    engine_mod._INFERENCE_SLOT.release()


def test_a_chat_reply_never_yields_its_own_slot() -> None:
    """Preemption is for background work only; a reply must finish."""
    import threading

    fake = _SlowFakeLlama()
    engine = _engine_with(fake)
    # The flag is set for the whole run, as it is while a reply holds the slot.
    engine_mod._FOREGROUND_WAITING.set()
    threading.Timer(2.0, engine_mod._FOREGROUND_WAITING.clear).start()
    try:
        text = engine.generate([{"role": "user", "content": "hi"}], max_tokens=180)
    finally:
        engine_mod._FOREGROUND_WAITING.clear()
    assert text.strip(), "reply finished rather than bailing out"
    assert fake.chunks_consumed == 180, fake.chunks_consumed


def test_chat_outwaits_a_background_prefill(monkeypatch) -> None:
    """A background job in prefill cannot yield yet; the chat must not give up."""
    import sys
    import threading

    engine_mod = sys.modules["momentum.llm.engine"]
    monkeypatch.setattr(engine_mod, "INFERENCE_WAIT_S", 0.2)
    engine_mod._acquire_inference_slot(background=True)
    threading.Timer(0.8, engine_mod._release_inference_slot).start()

    engine_mod._acquire_inference_slot()
    engine_mod._release_inference_slot()
    assert not engine_mod._BACKGROUND_HOLDS_SLOT.is_set()


def test_background_yields_before_prefill_when_a_chat_starts_with_it() -> None:
    """On the phone both started together after the model load; the snippet
    won the slot and its prefill made the reply wait over a minute."""
    import threading

    fake = _SlowFakeLlama()
    engine = _engine_with(fake)
    threading.Timer(0.05, engine_mod._FOREGROUND_WAITING.set).start()
    try:
        with pytest.raises(engine_mod.InferencePreempted):
            engine.generate([{"role": "user", "content": "hi"}], background=True)
    finally:
        engine_mod._FOREGROUND_WAITING.clear()
    assert fake.chunks_consumed == 0

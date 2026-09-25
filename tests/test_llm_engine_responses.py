"""Response-shape handling in the AI Coach engine.

The native ``llama_cpp`` backend is unavailable in most test environments, so
these tests drive ``LlmEngine`` with a stub that replays real llama-cpp-python
payload shapes. Several payload fields are genuinely optional at runtime: the
first streaming chunk carries only ``{"role": ..., "content": None}``, and
``content`` is ``None`` when generation stops on a tool call or a full context.
Unguarded indexing here raised AttributeError / IndexError / TypeError inside
the coach, so each shape is pinned below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from momentum.llm import engine
from momentum.llm.engine import LlmEngine


class _StubLlama:
    """Minimal stand-in for ``llama_cpp.Llama`` returning a canned payload."""

    def __init__(self, payload: Any) -> None:
        self._payload = payload
        self.calls: list[dict[str, Any]] = []

    def create_chat_completion(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._payload


def _engine_returning(payload: Any) -> LlmEngine:
    """Build an unloaded engine whose ``_llama`` replays *payload*."""
    eng = LlmEngine(model_path=Path("/fake.gguf"))
    eng._llama = _StubLlama(payload)
    return eng


_MESSAGES: list[dict[str, str]] = [{"role": "user", "content": "hello"}]


class TestGenerate:
    """Non-streaming completion handling."""

    def test_extracts_message_content(self) -> None:
        eng = _engine_returning(
            {"choices": [{"message": {"role": "assistant", "content": "hi there"}}]}
        )
        assert eng.generate(_MESSAGES) == "hi there"

    def test_strips_surrounding_whitespace(self) -> None:
        eng = _engine_returning({"choices": [{"message": {"content": "  spaced  "}}]})
        assert eng.generate(_MESSAGES) == "spaced"

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param(
                {"choices": [{"message": {"role": "assistant", "content": None}}]},
                id="content-is-none",
            ),
            pytest.param(
                {"choices": [{"message": {"role": "assistant"}}]},
                id="content-key-absent",
            ),
            pytest.param({"choices": []}, id="empty-choices"),
            pytest.param({"choices": [{}]}, id="choice-without-message"),
            pytest.param({}, id="no-choices-key"),
            pytest.param({"choices": "not-a-list"}, id="choices-not-a-list"),
        ],
    )
    def test_degenerate_payloads_return_empty_string(self, payload: Any) -> None:
        """These shapes used to raise instead of yielding ''.

        ``assist.request_assistance`` turns an empty string into a clean
        "the coach returned an empty response" error, which is the intended
        path; an unhandled exception is not.
        """
        assert _engine_returning(payload).generate(_MESSAGES) == ""


class TestGenerateStreaming:
    """Streaming chunk accumulation."""

    def test_accumulates_content_chunks(self) -> None:
        chunks = [
            {"choices": [{"delta": {"role": "assistant", "content": None}}]},
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
        ]
        eng = _engine_returning(iter(chunks))
        assert eng.generate(_MESSAGES, stream=True) == "Hello world"


class TestGenerateAsync:
    """Asynchronous streaming used by the chat UI."""

    def _run(self, chunks: list[Any]) -> tuple[list[str], list[str], list[Exception]]:
        tokens: list[str] = []
        done: list[str] = []
        errors: list[Exception] = []
        eng = _engine_returning(iter(chunks))
        thread = eng.generate_async(
            _MESSAGES, tokens.append, done.append, errors.append
        )
        thread.join(timeout=10)
        assert not thread.is_alive()
        return tokens, done, errors

    def test_streams_tokens_then_reports_completion(self) -> None:
        tokens, done, errors = self._run(
            [
                {"choices": [{"delta": {"role": "assistant", "content": None}}]},
                {"choices": [{"delta": {"content": "Hi"}}]},
                {"choices": [{"delta": {"content": "!"}}]},
            ]
        )
        assert errors == []
        assert tokens == ["Hi", "!"]
        assert done == ["Hi!"]

    def test_non_string_content_does_not_abort_the_stream(self) -> None:
        """An int ``content`` used to raise TypeError mid-stream.

        It arrived as a stray first chunk, which discarded an otherwise valid
        reply and surfaced a spurious error in the chat UI.
        """
        tokens, done, errors = self._run(
            [
                {"choices": [{"delta": {"content": 123}}]},
                {"choices": [{"delta": {"content": "recovered"}}]},
            ]
        )
        assert errors == []
        assert tokens == ["recovered"]
        assert done == ["recovered"]


class TestImportGuard:
    """The guarded import must still expose a patchable ``Llama`` symbol."""

    def test_llama_symbol_is_patchable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sentinel = object()
        monkeypatch.setattr(engine, "Llama", sentinel)
        assert engine.Llama is sentinel

    def test_generate_raises_before_load(self) -> None:
        eng = LlmEngine(model_path=Path("/fake.gguf"))
        with pytest.raises(RuntimeError, match="Model not loaded"):
            eng.generate(_MESSAGES)

    def test_lock_is_released_after_a_degenerate_payload(self) -> None:
        """A degenerate payload must not wedge the engine's lock."""
        eng = _engine_returning({"choices": [{"message": {"content": None}}]})
        eng.generate(_MESSAGES)
        assert eng._lock.acquire(blocking=False)
        eng._lock.release()


class TestHelpers:
    """Direct coverage of the payload-normalisation helpers."""

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            ({"choices": [{"message": {"content": "x"}}]}, "x"),
            ({"choices": [{"message": {"content": None}}]}, ""),
            ({"choices": []}, ""),
            ({}, ""),
            (None, ""),
            (42, ""),
        ],
    )
    def test_message_text(self, payload: Any, expected: str) -> None:
        assert engine._message_text(payload) == expected

    @pytest.mark.parametrize(
        ("chunk", "expected"),
        [
            ({"choices": [{"delta": {"content": "x"}}]}, "x"),
            ({"choices": [{"delta": {"role": "assistant", "content": None}}]}, ""),
            ({"choices": []}, ""),
            ({}, ""),
            (None, ""),
        ],
    )
    def test_delta_text(self, chunk: Any, expected: str) -> None:
        assert engine._delta_text(chunk) == expected

    @pytest.mark.parametrize(
        "payload", [{"choices": {}}, {"choices": []}, {"choices": ["s"]}, {}, None]
    )
    def test_first_choice_returns_none_without_choices(self, payload: Any) -> None:
        assert engine._first_choice(payload) is None

    @pytest.mark.parametrize(
        "chunks",
        [
            pytest.param([{"choices": []}], id="empty-choices"),
            pytest.param([{"choices": [{}]}], id="choice-without-delta"),
            pytest.param([{"choices": [{"delta": {}}]}], id="delta-without-content"),
            pytest.param(
                [{"choices": [{"delta": {"content": None}}]}], id="content-is-none"
            ),
            pytest.param(
                [{"choices": [{"delta": {"content": 123}}]}], id="content-not-a-string"
            ),
        ],
    )
    def test_contentless_chunks_never_raise(self, chunks: list[Any]) -> None:
        eng = _engine_returning(iter(chunks))
        assert eng.generate(_MESSAGES, stream=True) == ""

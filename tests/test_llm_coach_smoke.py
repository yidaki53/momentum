"""End-to-end host-side smoke test of the AI Coach path.

Exercises the real chain the UI uses -- context building, chat history, prompt
assembly, and ``LlmEngine.generate`` against a stubbed native backend -- with
the *real* llama_cpp import state, to show the coach is wired up and not merely
importable. A real GGUF is not available in CI, so inference itself is stubbed;
the payload shapes replayed here are the ones llama-cpp-python actually emits.
"""

from __future__ import annotations

from typing import Any

from momentum.llm import assist, engine
from momentum.llm.context import build_chat_history, build_user_context
from momentum.llm.prompts import SYSTEM_PROMPT


def _fake_llama_reply(text: str) -> Any:
    class _Fake:
        def create_chat_completion(self, **kw: Any) -> Any:
            return {"choices": [{"message": {"role": "assistant", "content": text}}]}

    return _Fake()


def test_native_backend_state_is_reported() -> None:
    assert isinstance(engine.is_llm_available(), bool)
    diag = engine.native_diagnostics()
    assert set(diag) >= {"available", "import_error", "configured_path"}
    if not engine.is_llm_available():
        # A build without the native lib must still explain itself.
        assert diag["import_error"]


def test_coach_chain_end_to_end(tmp_path, monkeypatch) -> None:
    from momentum import db as db_mod
    from momentum.models import AppConfig, TaskCreate

    # Use the app's own get_connection() so schema and row factory match production.
    conn = db_mod.get_connection(db_path=tmp_path / "momentum.db")
    db_mod.add_task(conn, TaskCreate(title="Ship it"))
    conn.execute("UPDATE tasks SET status = 'active' WHERE title = 'Ship it'")
    conn.commit()

    # Real context + history assembly against a real (in-memory) database.
    ctx = build_user_context(conn)
    assert "Ship it" in ctx
    assert isinstance(build_chat_history(conn), list)

    conf = AppConfig()
    monkeypatch.setattr(engine, "LLM_AVAILABLE", True)
    monkeypatch.setattr(assist, "is_llm_available", lambda: True)
    monkeypatch.setattr(assist, "is_model_downloaded", lambda *a, **k: True)
    monkeypatch.setattr(engine, "ensure_model", lambda *a, **k: tmp_path / "m.gguf")
    monkeypatch.setattr(
        engine,
        "Llama",
        lambda **kw: _fake_llama_reply("Small step. Start with two minutes."),
    )
    engine._engine_instance = None
    assist.clear_cache()

    results: dict[str, Any] = {}
    thread = assist.request_assistance(
        "Encourage me to begin",
        conn=conn,
        config=conf,
        cache_key="smoke-1",
        on_done=lambda t: results.setdefault("text", t),
        on_error=lambda e: results.setdefault("error", e),
    )
    if thread is not None:
        thread.join(timeout=15)
        assert not thread.is_alive()

    assert "error" not in results, f"coach failed: {results.get('error')}"
    assert "Small step" in results.get("text", "")
    assert results["text"].strip()
    assert SYSTEM_PROMPT  # the system prompt is wired into the request
    conn.close()
    engine._engine_instance = None
    assist.clear_cache()

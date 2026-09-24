from pathlib import Path

from momentum import db
from momentum.llm import assist
from momentum.llm.context import build_user_context
from momentum.models import (
    AssessmentResultCreate,
    AssessmentType,
    FocusSessionCreate,
    TaskCreate,
)


def test_context_includes_history_and_activity(tmp_path: Path) -> None:
    conn = db.get_connection(tmp_path / "momentum.db")
    try:
        first = db.add_task(conn, TaskCreate(title="Start the report"))
        db.set_task_active(conn, first.id)
        db.complete_task(conn, first.id)
        db.add_task(conn, TaskCreate(title="Review the report"))
        db.log_focus_session(conn, FocusSessionCreate(duration_minutes=15))
        db.save_assessment(
            conn,
            AssessmentResultCreate(
                assessment_type=AssessmentType.BDEFS,
                score=20,
                max_score=80,
                domain_scores={"self-management": 5},
            ),
        )
        db.save_assessment(
            conn,
            AssessmentResultCreate(
                assessment_type=AssessmentType.BDEFS,
                score=16,
                max_score=80,
                domain_scores={"self-management": 4},
            ),
        )

        context = build_user_context(conn)
        assert "Recently completed tasks" in context
        assert "Review the report" in context
        assert "Recent focus sessions" in context
        assert "BDEFS history" in context
        assert "latest change down by 4 points" in context
    finally:
        conn.close()


def test_assistance_gate_respects_disabled_config(monkeypatch) -> None:
    from types import SimpleNamespace

    config = SimpleNamespace(llm_enabled=False, llm_model="tinyllama")
    monkeypatch.setattr(assist.cfg, "load_config", lambda: config)
    monkeypatch.setattr(assist, "is_llm_available", lambda: True)
    monkeypatch.setattr(assist, "is_model_downloaded", lambda _name: True)
    errors = []

    result = assist.request_assistance(
        "Give me a next step",
        config=config,
        on_error=errors.append,
    )

    assert result is None
    assert errors and isinstance(errors[0], RuntimeError)


def test_assistance_caches_generated_text(monkeypatch) -> None:
    from types import SimpleNamespace

    config = SimpleNamespace(llm_enabled=True, llm_model="tinyllama")
    monkeypatch.setattr(assist.cfg, "load_config", lambda: config)
    monkeypatch.setattr(assist, "is_llm_available", lambda: True)
    monkeypatch.setattr(assist, "is_model_downloaded", lambda _name: True)
    engine = type(
        "Engine", (), {"generate": lambda self, *args, **kwargs: "One small step."}
    )()
    monkeypatch.setattr(assist, "get_engine", lambda _name: engine)
    assist.clear_cache()
    values = []
    first = assist.request_assistance(
        "Give me a next step",
        config=config,
        cache_key="test-cache",
        on_done=values.append,
    )
    assert first is not None
    first.join(timeout=2)
    second = assist.request_assistance(
        "Give me a next step",
        config=config,
        cache_key="test-cache",
        on_done=values.append,
    )
    assert second is None
    assert values == ["One small step.", "One small step."]
    assist.clear_cache()

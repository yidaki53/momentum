"""Exercise the real Kivy coach with fake inference/network services."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("kivy")

from kivy.lang import Builder
from kivy.uix.button import Button

from mobile import main


@pytest.fixture
def coach(monkeypatch):
    main.MomentumApp()  # Provides the `app` global the KV rules bind against.
    Builder.load_string(main.KV)
    screen = main.CoachScreen()
    screen._disclaimer_shown = True
    monkeypatch.setattr(
        main.cfg, "load_config", lambda: SimpleNamespace(llm_model="tinyllama")
    )
    monkeypatch.setattr(screen, "_load_history", lambda funcs: None)
    return screen


def funcs(available=False, downloaded=False):
    return {
        "is_llm_available": lambda: available,
        "is_model_downloaded": lambda name: downloaded,
        "model_size_mb": lambda name: 720,
        "import_error": "ImportError: missing dependency",
        "SHORT_DISCLAIMER": "Not medical advice",
        "DISCLAIMER": "Not medical advice",
    }


def test_missing_engine_allows_drafting_and_explicit_download(coach, monkeypatch):
    monkeypatch.setattr(main, "_get_llm_funcs", lambda: funcs())
    coach.on_enter()
    assert not coach.ready
    assert not coach.ids.coach_input.disabled
    assert coach.ids.coach_input.height >= main.dp(104)
    assert any(
        isinstance(w, Button) and w.text == "Download model" for w in coach.walk()
    )
    assert any("missing dependency" in getattr(w, "text", "") for w in coach.walk())


@pytest.mark.parametrize("downloaded", [False, True])
def test_ready_requires_engine_and_model(coach, monkeypatch, downloaded):
    monkeypatch.setattr(main, "_get_llm_funcs", lambda: funcs(True, downloaded))
    coach.on_enter()
    assert coach.ready is downloaded
    assert not coach.ids.coach_input.disabled
    send = next(w for w in coach.walk() if isinstance(w, Button) and w.text == "Send")
    assert send.disabled is not downloaded
    coach.busy = True
    assert send.disabled


@pytest.mark.parametrize("fail", [False, True])
def test_download_callbacks_run_on_ui_queue_and_allow_retry(coach, monkeypatch, fail):
    pending = []
    notices = []
    f = funcs(True)

    def download(name, progress_callback):
        progress_callback(512, 1024)
        if fail:
            raise OSError("offline")

    f["ensure_model"] = download
    monkeypatch.setattr(
        main.Clock, "schedule_once", lambda cb, *args: pending.append(cb)
    )
    monkeypatch.setattr(
        main.threading, "Thread", lambda target, **kwargs: SimpleNamespace(start=target)
    )
    monkeypatch.setattr(main, "_show_info_popup", lambda *args: notices.append(args))
    monkeypatch.setattr(main, "_show_error_popup", lambda *args: notices.append(args))
    monkeypatch.setattr(coach, "on_enter", lambda: None)
    coach._start_model_download("tinyllama", f)
    assert coach.busy
    # Worker callbacks queue progress and completion; do not touch Kivy directly.
    for cb in list(pending):
        cb(0)
    assert not coach.busy
    assert notices[-1][0] == ("Download failed" if fail else "Download complete")

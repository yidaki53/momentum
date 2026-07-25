"""Non-Kivy guardrail tests for mobile home UI behavior contracts."""

from __future__ import annotations

from pathlib import Path


def _mobile_main_source() -> str:
    root = Path(__file__).resolve().parent.parent
    return (root / "mobile" / "main.py").read_text(encoding="utf-8")


def test_section_markers_are_ascii_only() -> None:
    src = _mobile_main_source()
    assert "▼" not in src
    assert "▶" not in src
    assert "↔" not in src
    assert "('+ '" in src or "'+ '" in src
    assert "('- '" in src or "'- '" in src


def test_collapsible_section_bodies_are_height_driven() -> None:
    src = _mobile_main_source()
    assert "height: self.minimum_height if root.tasks_expanded else dp(0)" in src
    assert "height: self.minimum_height if root.timer_expanded else dp(0)" in src
    assert (
        "height: self.minimum_height if root.journal_expanded and root.act_controls_visible else dp(0)"
        in src
    )
    assert "height: dp(180) if root.tasks_expanded else dp(0)" in src
    assert "height: dp(44) if root.tasks_expanded else dp(0)" in src
    assert "height: dp(50) if root.timer_expanded else dp(0)" in src
    assert "height: dp(8) if root.timer_expanded else dp(0)" in src
    assert "height: dp(48) if root.timer_expanded else dp(0)" in src
    assert (
        "height: dp(40) if root.journal_expanded and root.act_controls_visible else dp(0)"
        in src
    )
    assert "disabled: not root.tasks_expanded" in src
    assert "disabled: not root.timer_expanded" in src
    assert "disabled: not (root.journal_expanded and root.act_controls_visible)" in src


def test_collapsed_section_children_are_removed_from_touch_flow() -> None:
    src = _mobile_main_source()
    assert "opacity: 1 if root.tasks_expanded else 0" in src
    assert "opacity: 1 if root.timer_expanded else 0" in src
    assert "opacity: 1 if root.act_controls_visible else 0" in src
    assert "disabled: not (root.journal_expanded and root.act_controls_visible)" in src


def test_encouragement_is_outside_collapsible_sections() -> None:
    src = _mobile_main_source()
    assert "text: root.nudge_text" in src
    assert "text: 'New encouragement'" not in src
    assert "def refresh_nudge(self):" not in src
    assert src.index("text: root.nudge_text") > src.index(
        "on_release: root.open_act_history()"
    )
    assert src.index("text: root.nudge_text") < src.index("Toolbar:")


def test_home_screen_exposes_accordion_toggle_handlers() -> None:
    src = _mobile_main_source()
    assert "def toggle_tasks_section(self) -> None:" in src
    assert "def toggle_timer_section(self) -> None:" in src
    assert "def toggle_journal_section(self) -> None:" in src
    assert 'self._toggle_section("tasks")' in src
    assert 'self._toggle_section("timer")' in src
    assert 'self._toggle_section("journal")' in src


def test_act_section_is_labeled_and_gated_by_profile_threshold() -> None:
    src = _mobile_main_source()
    assert "'ACT - ' + root.journal_summary" in src
    assert "height: dp(42) if root.act_controls_visible else dp(0)" in src
    assert "if not self.act_controls_visible:" in src
    assert "Acceptance and Commitment Therapy" in src
    assert 'title="ACT Momentum Reset"' in src


def test_update_check_and_timer_cycle_copy_are_mobile_safe() -> None:
    src = _mobile_main_source()
    assert "TODO: Implement actual update checking" not in src
    assert "Auto focus/break" in src
    assert 'text="ⓘ"' not in src
    assert 'text="Info"' in src
    assert 'text="Save reset"' in src


def test_ai_coach_screen_is_wired_with_graceful_degradation() -> None:
    src = _mobile_main_source()
    # The coach screen class, lazy LLM loader, and Home entry point exist.
    assert "class CoachScreen(Screen):" in src
    assert "def _get_llm_funcs()" in src
    assert "def open_coach(self) -> None:" in src
    assert "on_release: root.open_coach()" in src
    assert "text: 'AI Coach'" in src
    # CoachScreen is registered in the screen manager.
    assert 'CoachScreen(name="coach")' in src
    # Graceful degradation: the availability probe gates the screen, and a
    # clear message is shown when the native backend is absent.
    assert "is_llm_available" in src
    assert "_COACH_UNAVAILABLE_MSG" in src
    assert "AI Coach inference is not available on this build" in src
    assert "def _show_unavailable" in src
    # Send is gated on availability (no inference attempt without the backend).
    assert 'if funcs is None or not funcs["is_llm_available"]():' in src
    # The model is opt-in (download prompt), not bundled.
    assert "def _offer_model_download" in src
    assert "Download model" in src
    # Chat persistence reuses the shared DB layer.
    assert "db.add_llm_chat_message" in src
    assert "db.list_llm_chat_messages" in src
    assert "db.delete_all_llm_chat_messages" in src


def test_settings_controls_use_black_and_white_checkboxes() -> None:
    src = _mobile_main_source()
    # Checkbox widget is imported and a tick-row helper exists.
    assert "from kivy.uix.checkbox import CheckBox" in src
    assert "def _make_check_row(" in src
    # Theme and timer cycle are radio-style checkbox groups (tick marks),
    # not accent-coloured ToggleButtons.
    assert 'group="theme_mode"' in src
    assert 'group="timer_cycle_mode"' in src
    assert 'state="down" if current.theme_mode.value' not in src
    assert 'state="down" if current.timer_cycle_mode.value' not in src
    # Accessibility options are independent checkboxes stacked vertically.
    assert "_access_cbs" in src
    assert 'state="down" if current.accessibility_large_text' not in src
    # "Check at startup" is a checkbox bound via active, not a ToggleButton.
    assert "check_startup_cb.bind(active=" in src
    assert 'state="down" if current.check_updates_at_startup' not in src
    # The Auto focus/break label is preserved on the timer-cycle checkbox.
    assert '"Auto focus/break"' in src


def test_stroop_uses_multiple_choice_buttons() -> None:
    src = _mobile_main_source()
    assert "Tap the INK COLOUR of the text, not the word." in src
    assert "id: stroop_input" not in src
    assert "on_release: root.answer_option(self.text)" in src


def test_self_update_wiring_is_present() -> None:
    src = _mobile_main_source()
    assert "def _trigger_apk_download" in src
    assert "def _is_play_installed" in src
    assert "def _android_activity" in src
    assert 'text="Update now"' in src
    # DownloadManager provides the content URI so no FileProvider is needed.
    assert "DownloadManager" in src
    assert "application/vnd.android.package-archive" in src
    # Play-installed builds must skip self-update.
    assert 'text="Open download page"' in src


def test_buildozer_spec_grants_install_packages() -> None:
    spec = (
        Path(__file__).resolve().parent.parent / "mobile" / "buildozer.spec"
    ).read_text(encoding="utf-8")
    assert "REQUEST_INSTALL_PACKAGES" in spec


def test_buildozer_spec_enables_backup_and_llama_recipe() -> None:
    root = Path(__file__).resolve().parent.parent
    spec = (root / "mobile" / "buildozer.spec").read_text(encoding="utf-8")
    # Auto Backup is explicitly enabled (update-safe persistence guarantee).
    assert "android.allow_backup = True" in spec
    # The llama-cpp-python recipe is wired in as a best-effort requirement with
    # a local recipes directory; CI falls back to building without it.
    assert "llama-cpp-python" in spec
    assert "p4a.local_recipes = p4a-recipes" in spec
    # The recipe file itself exists and declares the p4a recipe.
    recipe = root / "mobile" / "p4a-recipes" / "llama_cpp_python" / "__init__.py"
    assert recipe.exists()
    recipe_src = recipe.read_text(encoding="utf-8")
    assert "class LlamaCppPythonRecipe" in recipe_src
    assert "recipe = LlamaCppPythonRecipe()" in recipe_src
    # CPU-only build (no GPU backends) and the graceful-degradation note.
    assert "-DGGML_CUDA=OFF" in recipe_src
    assert "gracefully" in recipe_src.lower() or "graceful" in recipe_src.lower()


def test_ci_has_llama_recipe_fallback() -> None:
    ci = (
        Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")
    # Both the APK and AAB build steps strip the llama recipe on a cmake/llama
    # error and rebuild, so the APK always ships (graceful degradation).
    assert "llama-cpp-python" in ci
    assert "sed -i 's|,llama-cpp-python||' buildozer.spec" in ci
    assert "sed -i '/^p4a.local_recipes/d' buildozer.spec" in ci
    assert "CMake Error" in ci
    assert "is_llm_available" in ci or "graceful" in ci.lower()

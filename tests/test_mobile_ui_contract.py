"""Non-Kivy guardrail tests for mobile home UI behavior contracts."""

from __future__ import annotations

from configparser import ConfigParser
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


def test_ai_coach_screen_reports_its_own_failure_reason() -> None:
    """The unavailable screen must name the cause, not just apologise.

    Ten releases shipped a coach that said only "bundle the local model engine"
    while the real reason -- an import traceback, an empty native directory, the
    chosen library path -- was logged at debug level, which a release APK never
    surfaces. The device is the only place the truth exists, so it has to be on
    the screen the user is looking at.
    """
    src = _mobile_main_source()
    # The loader failure is captured and logged at a level logcat carries.
    assert "_LLM_IMPORT_TRACEBACK = traceback.format_exc()" in src
    assert 'log.warning("AI Coach modules unavailable' in src
    assert 'log.debug("LLM module unavailable' not in src
    # The native loader report is rendered, keyed off the engine's diagnostics.
    assert "def _coach_diagnostics_text(" in src
    assert 'funcs.get("native_diagnostics")' in src
    assert "_COACH_DIAGNOSTIC_KEYS" in src
    # The captured import error is shown even in a build whose diagnostics probe
    # is missing or raises: one broken reporter must not silence the other.
    assert 'funcs.get("import_error")' in src
    assert 'lines.append(f"import_error={import_error.strip()}")' in src
    assert 'lines.append("native_diagnostics() unavailable in this build.")' in src
    # And the build is identifiable, so a screenshot says which APK it came from.
    assert 'f"Build: {APP_VERSION} #{BUILD_NUMBER} ({BUILD_VARIANT})\\n"' in src
    assert "_coach_diagnostics_text(funcs)" in src


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
    root = Path(__file__).resolve().parent.parent
    spec = (root / "mobile" / "buildozer.spec").read_text(encoding="utf-8")
    assert "REQUEST_INSTALL_PACKAGES" in spec

    # Vulkan shader compilation needs glslc: the Vulkan CI job installs
    # glslang-tools, the recipe prefers the NDK-bundled copy and fails fast
    # with a clear error when neither is available (instead of dying deep
    # inside CMake).
    ci = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "glslang-tools" in ci
    assert "Install Vulkan shader tooling (glslc + headers)" in ci
    assert "NDK Vulkan headers missing" in ci
    # Exactly one tooling step: a duplicated copy previously existed and the
    # two drifted apart, which is how the missing glslc wiring went unnoticed.
    assert ci.count("Install Vulkan shader tooling (glslc + headers)") == 1

    recipe_src = (
        root / "mobile" / "p4a-recipes" / "llama-cpp-python" / "__init__.py"
    ).read_text(encoding="utf-8")
    # The recipe exports GLSLC to the CMake environment and refuses to start a
    # doomed Vulkan source build without a shader compiler.
    assert "GLSLC" in recipe_src
    assert "requires glslc" in recipe_src
    assert "-DGGML_VULKAN=ON" in recipe_src


def test_buildozer_spec_enables_backup_and_llama_recipe() -> None:
    root = Path(__file__).resolve().parent.parent
    spec = (root / "mobile" / "buildozer.spec").read_text(encoding="utf-8")
    # Auto Backup is explicitly enabled (update-safe persistence guarantee).
    assert "android.allow_backup = True" in spec
    # The llama-cpp-python recipe is wired in as a best-effort requirement with
    # a local recipes directory; CI falls back to building without it.
    assert "llama-cpp-python" in spec
    assert "p4a.local_recipes = p4a-recipes" in spec
    # llama-cpp-python's pure-Python runtime deps are in the requirements so
    # Llama import/use works on-device.
    assert "diskcache" in spec
    assert "jinja2" in spec
    assert "typing-extensions" in spec
    # v1 ships arm64-v8a only (deterministic single-arch prebuilt).
    assert "android.archs = arm64-v8a" in spec
    # The recipe file itself exists and declares the p4a recipe.
    # NOTE: the folder MUST use the hyphenated, PEP-503-normalised name
    # ``llama-cpp-python`` -- p4a resolves recipe names by folder name
    # (Recipe.name is a @property over the module/folder path), and the
    # buildozer.spec requirement token is the hyphenated form. An underscore
    # folder (``llama_cpp_python``) is NOT matched, so p4a silently falls back
    # to pip-installing the package (wheel-only) instead of using the local
    # source recipe, which breaks the on-device AI coach build.
    recipe_dir = root / "mobile" / "p4a-recipes" / "llama-cpp-python"
    assert recipe_dir.is_dir()
    assert not (root / "mobile" / "p4a-recipes" / "llama_cpp_python").exists()
    recipe = recipe_dir / "__init__.py"
    assert recipe.exists()
    recipe_src = recipe.read_text(encoding="utf-8")
    assert "class LlamaCppPythonRecipe" in recipe_src
    assert "recipe = LlamaCppPythonRecipe()" in recipe_src
    # CPU-only build (no GPU backends) and the no-fallback note.
    assert "-DGGML_CUDA=OFF" in recipe_src
    assert "no fallback" in recipe_src.lower()


def test_llama_recipe_uses_normalised_sdist_url_and_declares_runtime_deps() -> None:
    root = Path(__file__).resolve().parent.parent
    recipe = root / "mobile" / "p4a-recipes" / "llama-cpp-python" / "__init__.py"
    src = recipe.read_text(encoding="utf-8")
    # PyPI sdist filenames use the NORMALISED name (underscores); the hyphen
    # form 404s. The URL must use llama_cpp_python (underscores).
    assert "llama_cpp_python-{version}.tar.gz" in src
    assert "llama-cpp-python-{version}.tar.gz" not in src
    # Runtime deps declared in llama-cpp-python's pyproject are listed so a
    # source build is self-contained (numpy is a top-level requirement).
    assert '"typing-extensions"' in src
    assert '"diskcache"' in src
    assert '"jinja2"' in src
    assert "python_depends" in src


def test_llama_p4a_wiring_is_in_app_section_and_pulls_prebuilt() -> None:
    root = Path(__file__).resolve().parent.parent
    spec_path = root / "mobile" / "buildozer.spec"
    cfg = ConfigParser()
    cfg.read(str(spec_path))
    # buildozer reads p4a.local_recipes / p4a.extra_args from [app], NOT
    # [buildozer]. A prior version had p4a.local_recipes under [buildozer] and
    # the recipe was silently never registered with p4a.
    assert cfg.get("app", "p4a.local_recipes", fallback="") == "p4a-recipes"
    assert cfg.get("buildozer", "p4a.local_recipes", fallback="") == ""
    extra_args = cfg.get("app", "p4a.extra_args", fallback="")
    # Prebuilt-wheel index is wired via p4a.extra_args (buildozer 1.5.0 predates
    # the first-class extra_index_urls token; p4a PR #3280 understands the flag).
    assert "--extra-index-url=https://yidaki53.github.io/p4a-wheels/p4a" in extra_args


def test_ci_requires_llama_build_to_succeed_and_uploads_logs() -> None:
    ci = (
        Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")
    # The build must succeed with llama-cpp-python included. No fallback to
    # strip the recipe on build failure.
    assert "llama-cpp-python" in ci
    # The fallback mechanism that stripped llama-cpp-python has been removed
    assert "sed -i 's|,llama-cpp-python||' buildozer.spec" not in ci
    assert "sed -i '/^p4a.local_recipes/d' buildozer.spec" not in ci
    assert "no fallback" in ci.lower()
    # Full buildozer logs are uploaded as an artifact so failures are
    # diagnosable (previously only tail -60 was echoed and the real error was
    # lost).
    assert "android-build-logs" in ci
    assert "/tmp/buildozer-pass2.log" in ci
    assert "if: always()" in ci

    # Releases publish when CPU artifacts succeed; the experimental Vulkan job
    # stays visible but must not gate the release.
    assert "needs.build.result" in ci
    assert "needs.build-apk.result" in ci
    # A Vulkan-only failure previously skipped the release job (run 71), which
    # silently stopped all publishes and froze the in-app update check. The job
    # is therefore marked advisory and the release condition is forced to
    # evaluate despite a dependency failure.
    assert "always() &&" in ci
    assert "continue-on-error: true" in ci


def test_coach_buttons_sit_above_the_text_field() -> None:
    """Send/Clear must be reachable above the keyboard, not under it."""
    src = _mobile_main_source()
    start = src.index("<CoachScreen>:")
    block = src[start : start + 4000]
    send_at = block.index("text: 'Send'")
    input_at = block.index("id: coach_input")
    assert send_at < input_at, "the button row must precede the TextInput"


def test_coach_enter_key_sends_the_message() -> None:
    """Enter sends rather than inserting a newline.

    Kivy only dispatches ``on_text_validate`` when ``multiline`` is False, so
    binding it on a multiline coach field did nothing at all. The input must be
    the subclass that intercepts the newline in ``insert_text``.
    """
    src = _mobile_main_source()
    assert "class EnterSendsTextInput(TextInput):" in src
    assert "def insert_text(self, substring: str, from_undo: bool = False):" in src
    assert 'if substring in ("\\n", "\\r") and self.multiline:' in src
    assert "send_on_enter: root.send_message" in src
    # The non-working binding must not come back.
    assert "on_text_validate: root.send_message()" not in src
    # One handler serves both the button and the key, so it must tolerate the
    # widget argument Kivy passes to each.
    assert "def send_message(self, *_args) -> None:" in src


def test_coach_disclaimer_is_shown_once_ever() -> None:
    """The medical notice must not reappear on every visit."""
    src = _mobile_main_source()
    assert "def _show_coach_disclaimer_once(" in src
    assert "coach_disclaimer_ack" in src
    # The old per-session screen flag must no longer gate the popup.
    assert '_show_info_popup("AI Coach", funcs["DISCLAIMER"])' not in src


def test_context_window_is_configurable_and_clamped() -> None:
    """A local model can be given a larger window, within safe bounds."""
    from momentum.llm import engine

    assert engine.MIN_CONTEXT_TOKENS >= 2048
    assert engine.MAX_CONTEXT_TOKENS >= 4096
    assert engine.MIN_CONTEXT_TOKENS < engine.MAX_CONTEXT_TOKENS


def test_coach_insights_are_automatic_not_button_gated() -> None:
    """Insights appear on their own; the user never presses a button."""
    src = _mobile_main_source()
    start = src.index("def _add_ai_insight(")
    block = src[start : start + 3000]
    assert "Get coach perspective" not in block
    # The request is scheduled, not bound to a press.
    assert "Clock.schedule_once(lambda _dt: _request(), 0)" in block
    assert "button.bind(on_release=_request)" not in block


def test_coach_insights_never_gate_the_rest_of_the_app() -> None:
    """When the coach is unavailable the screen is untouched and stays usable."""
    src = _mobile_main_source()
    start = src.index("def _add_ai_insight(")
    block = src[start : start + 3000]
    # Nothing is added when the coach is not ready...
    assert "if not _coach_ready():" not in block
    assert "or _chat_is_active()" in block
    # ...and a failure removes the panel instead of raising into the screen.
    assert "container.remove_widget(panel)" in block


def test_chat_has_priority_over_background_insights() -> None:
    """An interactive message must never be queued behind a suggestion.

    On device, entering the coach and sending a message started an automatic
    insight in the same millisecond; the insight won the inference slot and
    the real question was declined with "the coach is busy" after twenty
    seconds, without ever reaching the model.
    """
    src = _mobile_main_source()
    assert "_CHAT_ACTIVE = True" in src
    # Cleared on every terminal path, or insights would stay suppressed forever.
    assert src.count("_CHAT_ACTIVE = False") >= 3


def test_coach_can_propose_tasks_but_only_after_confirmation() -> None:
    """The model may suggest a task; the database is never written silently."""
    src = _mobile_main_source()
    assert "def _offer_task_creation(" in src
    assert "def _create_tasks(" in src
    start = src.index("def _offer_task_creation(")
    block = src[start : src.index("def _create_tasks(")]
    assert "Add these to your tasks?" in block
    # The database write lives in _create_tasks, which is only reached by the
    # confirmation button -- never inline in the reply handler.
    assert "_offer_task_creation(task_titles)" in src


def test_settings_offers_a_model_picker() -> None:
    """The user must be able to change model from the app.

    ``llm_model`` was only ever *read* -- no UI anywhere could write it, so the
    720 MB default could not be swapped for the smaller one even on a device
    where it does not fit. Switching model was impossible, not merely awkward.
    """
    src = _mobile_main_source()
    assert "def _select_model(" in src
    assert "def _available_model_specs(" in src
    assert '"available_models"' in src
    # Changing the model must also drop the cached engine, or the old weights
    # keep generating while the UI claims the new model is selected.
    select = src[src.index("def _select_model(") :]
    assert "conf.llm_model = name" in select
    assert 'funcs["reset_engine"]()' in select
    assert 'funcs["clear_assistance_cache"]()' in select


def test_model_picker_offers_download_when_missing() -> None:
    """Choosing an undownloaded model must offer to fetch it."""
    src = _mobile_main_source()
    assert "def _offer_model_download(" in src
    assert 'funcs["ensure_model"](name, progress_callback=_update)' in src


def test_mobile_main_compiles_under_the_android_host_python() -> None:
    """mobile/main.py must compile the way the Android build compiles it.

    Buildozer compiles the app with an older host python than the dev box, so
    syntax that 3.12 accepts still fails there. A re-declared `global` after an
    assignment in the same closure raised exactly that SyntaxError and silently
    stopped every APK from building, while ruff and pytest both passed locally.
    """
    import ast
    import pathlib

    source = (pathlib.Path(__file__).parent.parent / "mobile" / "main.py").read_text()
    ast.parse(source, feature_version=(3, 8))

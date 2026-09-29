# Runtime Diagnostics Playbook

## Symptom: minutes on the Android splash screen
1. Distinguish Android's `am start -W` / `ActivityTaskManager: Displayed` splash-window time from Kivy's `Start application main loop` and the visible home screen. Do not claim a two-second app launch from the former alone.
2. On a new install or update, inspect `pythonutil:V` for `Extracting private assets` and `Extracting ...libpybundle assets` before attributing a delay to AI. python-for-android extracts `private.tar` and `libpybundle.so` before Python starts; adopted storage (`/mnt/expand/...`) can make this conspicuous.
3. Compare the first post-install launch with a second launch without clearing app data. Verify the native extraction progress UI while unpacking. App-only updates should skip an unchanged Python bundle; a changed bundle should unpack again.
4. `run-as` can reject app-private paths on adopted storage with `/mnt has wrong owner`; use process-filtered logcat and a device screenshot when that happens. Limit logcat output to the current launch so extraction spam does not evict the meaningful lines.

## Symptom: AI Coach does not work on the phone
1. Diagnose native loading on the device, not from APK contents alone: check `[LLM-IMPORT] llama_cpp:import:ok`, `native_diagnostics`, and failures in logcat. Confirm a GGUF model exists and actually generates text before declaring inference functional.
2. Android CPython reports `sys.platform == "android"`; llama-cpp-python's loader may reject it as `Unsupported platform`. Keep Android loader configuration and logger initialization before the guarded import; an import-order `NameError` can hide behind a later successful debug retry.
3. Do not run a debug-only LLM probe or optional nudge at startup when AI is disabled. Keep slow native import off the Kivy UI thread, and marshal SQLite-backed context work and UI updates onto their owning thread.

## Symptom: black screen after splash
1. Validate startup logs first; identify last successful screen lifecycle callback.
2. Check for eager imports that can fail on mobile runtime constraints.
3. Wrap top-level UI callbacks and schedule user-safe error popups for recoverable failures.
4. Re-test with reduced visual load and default settings to isolate rendering vs logic faults.

## Symptom: settings interaction crashes
1. Reproduce each settings branch independently:
  - appearance,
  - timer cycle mode,
  - accessibility.
2. Ensure config setters validate enum values and raise clear errors.
3. Ensure UI handlers catch exceptions and keep app responsive.
4. Refresh dependent UI state only after persistence succeeds.

## Symptom: ACT/adaptive UI not matching assessments
1. Confirm latest assessment rows exist for BIS/BAS, BDEFS, and Stroop.
2. Verify profile derivation from latest assessments before UI render.
3. Test at least three profile states:
  - reassurance-heavy,
  - breakdown/action-heavy,
  - balanced/default.
4. Ensure hidden adaptive controls preserve layout stability and do not throw on interaction paths.

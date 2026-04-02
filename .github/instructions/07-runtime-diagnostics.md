# Runtime Diagnostics Playbook

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

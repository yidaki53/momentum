# Local Android builds

Run `poetry install --with dev`, install Android SDK/NDK and JDK 17+, then
use `make mobile-apk` (debug APK) or `make mobile-aab` (signed release AAB;
requires signing credentials). `python3 mobile/scripts/build_android.py --check`
validates the environment without compiling.

The launcher works without shell activation: it selects the project `.venv`,
sets `VIRTUAL_ENV` and `PATH`, removes `PYTHONHOME`, chooses a JDK 17+ (overriding
an incompatible Java 8 `JAVA_HOME`), and exposes the committed pure-Python
MarkupSafe wheel using a file URI (safe for paths containing spaces).
It runs from `mobile/` and restores the spec after an AAB build, including failures.
No shell startup files or system Java alternatives are changed.

If p4a's generated helper venv has a mixed/broken pip installation, stop all
builds first and remove **only** the generated `build/venv` beneath the active
Buildozer storage directory. Do not delete the project `.venv` or the SDK.
For the default CPU build this generated directory is
`/tmp/buildozer-build/android/platform/build-arm64-v8a/build/venv`.
The next build recreates it. The pip-upgrade fallback patch is not a repair
for an already-corrupt pip; do not keep retrying the same failed environment.

After building, run `python3 mobile/scripts/verify_coach_apk.py <apk>`.
This checks package contents, not Android imports or inference. A USB device
must appear as `device` in `adb devices -l` (enable USB debugging and authorize
the computer) before runtime validation. A debug-signed APK may not update a
release-signed installation; do not uninstall the app to work around this
without backing up data and getting explicit approval.

The local CPU debug APK was built and passed the package verifier. The Vulkan
variant and release signing still need their own build/runtime validation.

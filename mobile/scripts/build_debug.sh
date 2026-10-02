#!/usr/bin/env bash
set -euo pipefail
# Helper: build and install a debuggable APK for local debugging
# Usage: ./mobile/scripts/build_debug.sh
cd "$(dirname "$0")/.."
REPO_ROOT="$(cd .. && pwd)"
BUILD_ENV="${BUILD_ENV:-$REPO_ROOT/.venv-build}"
BUILDOZER="$BUILD_ENV/bin/buildozer"
export PATH="$BUILD_ENV/bin:$PATH"
export PIP_INDEX_URL="https://pypi.org/simple"
unset PIP_EXTRA_INDEX_URL
export PIP_FIND_LINKS="$(poetry -C "$REPO_ROOT" run python -c 'import sys; from pathlib import Path; print(Path(sys.argv[1]).resolve().as_uri())' "$PWD/p4a-wheels")"

for java_home in /usr/lib/jvm/java-17-openjdk-amd64 /usr/lib/jvm/java-21-openjdk-amd64; do
  if [[ -x "$java_home/bin/java" ]]; then
    export JAVA_HOME="$java_home"
    export PATH="$JAVA_HOME/bin:$PATH"
    break
  fi
done

if [[ ! -x "$BUILDOZER" ]]; then
  echo "buildozer not found in PATH. Install buildozer in a venv first." >&2
  exit 2
fi

if ! "$BUILD_ENV/bin/python" -c "from distutils.version import LooseVersion" >/dev/null 2>&1; then
  echo "Installing the Python 3.12 distutils compatibility shim..."
  "$BUILD_ENV/bin/python" -m pip install --upgrade setuptools wheel
fi

if ! "$BUILD_ENV/bin/python" -c "import Cython" >/dev/null 2>&1; then
  echo "Installing Cython..."
  "$BUILD_ENV/bin/python" -m pip install --upgrade cython
fi

echo "Building debug APK (this may take a while)..."
"$BUILDOZER" android debug
APK="$PWD/bin/momentum-0.5.1-arm64-v8a-debug.apk"
poetry -C "$REPO_ROOT" run python "$PWD/scripts/verify_coach_apk.py" "$APK"
echo "Installing APK on connected device..."
ADB="$(command -v adb || true)"
if [[ -z "$ADB" ]]; then
  ADB="${ANDROIDSDK:-$HOME/.buildozer/android/platform/android-sdk}/platform-tools/adb"
fi
if ! "$ADB" install -r -d "$APK"; then
  echo "Existing APK has a different signing key; replacing it for debug testing."
  "$ADB" uninstall dev.momentum.momentum >/dev/null
  "$ADB" install "$APK"
fi
"$ADB" shell monkey -p dev.momentum.momentum 1 >/dev/null

echo "Done. Run: adb logcat -v threadtime PythonActivity:V System.err:V *:S > ~/momentum_debug_logcat.txt & and then open the Coach screen." 

#!/usr/bin/env bash
# Build a *pure-Python* MarkupSafe wheel for p4a's pure-python resolution stage.
#
# Why: p4a's pure-python dependency filter (pythonforandroid/build.py,
# get_filtered_pure_python_packages) runs pip with --only-binary=:all: and
# --platform=android_26_*. MarkupSafe's PyPI wheels are CPython/C-ABI platform
# wheels, so none matches --platform=android_26_*, and pip cannot fall back to
# the sdist under --only-binary. Because Jinja2 hard-depends on MarkupSafe at
# every published version, the resolver dies with ResolutionImpossible for
# jinja2 and the whole "Installing pure Python modules" stage aborts (seen in
# /tmp/momentum-debug-coach2.log: "Cannot install jinja2==... ResolutionImpossible").
#
# MarkupSafe's own setup.py ships a documented fallback: if the C _speedups
# extension fails to compile it rebuilds without it and produces a
# py3-none-any wheel. This script forces that fallback deterministically by
# putting a program that always fails on CC, so the C build can never succeed
# regardless of the host toolchain -- the wheel is reproducible everywhere.
#
# The generated wheel is committed at mobile/p4a-wheels/ so local builds and
# CI can point pip at it via PIP_FIND_LINKS (p4a's resolution call copies
# os.environ, and pip honors PIP_FIND_LINKS). Jinja2's speedups are not needed
# on-device for llama-cpp-python's tiny templating workload.
#
# Usage:
#   mobile/scripts/build-markupsafe-pure-wheel.sh            # build into mobile/p4a-wheels
#   mobile/scripts/build-markupsafe-pure-wheel.sh --verify   # verify committed wheel only (fast)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$REPO_ROOT/mobile/p4a-wheels"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

MARKUPSAFE_VERSION="${MARKUPSAFE_VERSION:-3.0.3}"
EXPECTED_WHEEL="markupsafe-${MARKUPSAFE_VERSION}-py3-none-any.whl"

verify() {
    # The committed wheel must be a pure-Python wheel with no compiled
    # extension, and must carry the MarkupSafe 3.0.3 metadata.
    local wheel="$OUT_DIR/$EXPECTED_WHEEL"
    [ -f "$wheel" ] || { echo "MISSING: $wheel" >&2; return 1; }

    python3 - "$wheel" <<'PY'
import sys, zipfile
wheel = sys.argv[1]
with zipfile.ZipFile(wheel) as z:
    names = z.namelist()
    so_files = [n for n in names if n.endswith((".so", ".pyd", ".pyd", ".dylib"))]
    assert not so_files, f"pure wheel must not contain binaries: {so_files}"
    wheel_meta = z.read("markupsafe-3.0.3.dist-info/WHEEL").decode()
    assert "Tag: py3-none-any" in wheel_meta, wheel_meta
    meta = z.read("markupsafe-3.0.3.dist-info/METADATA").decode()
    line = next(l for l in meta.splitlines() if l.startswith("Version:"))
    assert line.strip() == "Version: 3.0.3", line
    assert "markupsafe/__init__.py" in names
print(f"OK: {wheel} is a valid pure-Python wheel (py3-none-any, no binaries)")
PY
}

# Runtime smoke test in an isolated venv: MarkupSafe from the wheel + Jinja2
# from PyPI must install and render together.
verify_runtime() {
    local venv="$WORK_DIR/runtime-venv"
    python3 -m venv "$venv"
    "$venv/bin/pip" -q install "$OUT_DIR/$EXPECTED_WHEEL" "jinja2==3.1.6"
    "$venv/bin/python" - <<'PY'
import jinja2, markupsafe
out = jinja2.Template("hi {{ x|upper }}").render(x="coach")
assert out == "hi COACH", out
print(f"OK: jinja2 {jinja2.__version__} + markupsafe {markupsafe.__version__} render: {out!r}")
PY
}

if [ "${1:-}" = "--verify" ]; then
    verify
    exit 0
fi

mkdir -p "$OUT_DIR"
cd "$WORK_DIR"

# Everything runs inside an isolated venv: system pip may be PEP 668
# externally-managed (Ubuntu 23.04+), and we want the build reproducible
# regardless of host packages.
python3 -m venv "$WORK_DIR/venv"
venv="$WORK_DIR/venv"
"$venv/bin/pip" -q install --upgrade pip build wheel

# 1) Fetch the sdist (no build isolation surprises; --no-binary keeps it source).
"$venv/bin/pip" download --no-binary=:all: --no-deps \
    "markupsafe==${MARKUPSAFE_VERSION}" -d "$WORK_DIR/sdist" \
    >"$WORK_DIR/download.log" 2>&1 || {
    tail -n 20 "$WORK_DIR/download.log" >&2
    echo "ERROR: could not download markupsafe sdist" >&2
    exit 1
}

sd="$(ls "$WORK_DIR"/sdist/*.tar.gz)"
tar -xzf "$sd" -C "$WORK_DIR"
src_dir="$WORK_DIR/MarkupSafe-${MARKUPSAFE_VERSION}"
[ -d "$src_dir" ] || src_dir="$WORK_DIR/markupsafe-${MARKUPSAFE_VERSION}"

# 2) Build with a guaranteed-to-fail CC so setup.py takes its documented
#    pure-Python fallback (see the "Plain-Python build succeeded" banner).
cd "$src_dir"
CC=/bin/false "$venv/bin/python" -m build --wheel --no-isolation -o "$WORK_DIR/out" \
    >"$WORK_DIR/build.log" 2>&1 || {
    tail -n 40 "$WORK_DIR/build.log" >&2
    echo "ERROR: wheel build failed" >&2
    exit 1
}
grep -q "Plain-Python build succeeded" "$WORK_DIR/build.log" || {
    echo "ERROR: expected 'Plain-Python build succeeded' fallback not taken" >&2
    tail -n 20 "$WORK_DIR/build.log" >&2
    exit 1
}

built="$(ls "$WORK_DIR"/out/*.whl)"
base="$(basename "$built")"
if [ "$base" != "$EXPECTED_WHEEL" ]; then
    echo "ERROR: built '$base', expected '$EXPECTED_WHEEL'" >&2
    exit 1
fi

cp "$built" "$OUT_DIR/"
verify
verify_runtime

echo
echo "Committed-pure-wheel refreshed: $OUT_DIR/$EXPECTED_WHEEL"
echo "Point p4a's pip at it with: PIP_FIND_LINKS=$OUT_DIR"

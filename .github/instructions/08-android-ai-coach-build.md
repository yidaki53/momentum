# Android AI-Coach (llama-cpp-python) Native Build

Local gate before pushing: `buildozer android debug` from `mobile/`
(CI owns the signed release lane; never attempt release signing locally).

## p4a recipe resolution
- `Recipe.get_recipe()` lowercases names but does NOT normalize `-` vs `_`.
- Folder `mobile/p4a-recipes/<name>/` MUST match the requirement token in
  `buildozer.spec` exactly (modulo case). A mismatch raises
  "Recipe does not exist", and p4a then silently demotes the requirement to a
  pip `python_modules` install with `--only-binary`, which fails when no
  Android wheel exists for the package.
- Contract test (`tests/test_mobile_ui_contract.py`) pins the recipe dir name;
  keep the spec token, folder name, and test in sync.

## scikit-build-core config-setting key
- Upstream p4a `PyProjectRecipe.build_arch` passes
  `--config-setting builddir=<dir>`, but scikit-build-core >= 0.9 rejects it
  (`Unrecognized options in config-settings: builddir -> did you mean:
  build-dir, build?`).
- The local recipe overrides `build_arch` and passes `build-dir`. Keep the
  deviation documented in the recipe docstring and re-check on p4a upgrades.

## CMAKE_ARGS merging
- Merge p4a's base `CMAKE_ARGS` with recipe flags using SPACES, never `;`.
  A semicolon join collapses every `-D` flag into one BOOL cache value
  (`GGML_NATIVE:BOOL=OFF;-DGGML_OPENMP=OFF;...`), so flags never take effect
  and host-only flags (`-march=native`) leak into the NDK clang compile
  (`clang: error: the clang compiler does not support '-march=native'`).
- Success markers in the build log: `-- Setting GGML_NATIVE_DEFAULT to OFF`;
  compile lines carry `--target=aarch64-none-linux-android...` with no
  `-march=native` / `-mcpu=native`; OpenMP/CUDA/Metal/OpenCL OFF.

## Local build environment (Linux)
- Activate the repo venv before invoking buildozer (`VIRTUAL_ENV=.../.venv`,
  venv `bin/` first on PATH). Without activation buildozer runs
  `pip install -q --user ...` for p4a deps, which a venv interpreter refuses
  (`Can not perform a '--user' install`).
- Reuse the cache: the spec already sets
  `[buildozer] build_dir=/tmp/buildozer-build`; export `ANDROID_NDK_HOME` /
  `ANDROID_NDK` / `ANDROID_HOME` at the cached NDK r25b + SDK so nothing
  re-downloads. Heavy recipes then log "already built, skipping".
- Select JDK 17 explicitly (`/usr/lib/jvm/java-17-openjdk-amd64`); a naive
  newest-glob can pick temurin-8, which gradle/d8/apksigner reject (need 11+).
- Ensure several GB free (`df`) before launching; the llama.cpp CMake compile
  needs headroom beyond the ~4 GB cache.

## Long-build shell hygiene
- Launch long builds detached (`setsid`, stdout/stderr to a log file under
  `/tmp`) and return immediately; never run them in the foreground.
- Monitor via file reads of the log, not shell `tail`/`grep` pipelines that
  block on the still-running build process. Clean up stale background jobs
  (buildozer, gradle daemons, ninja) before relaunching.
- Buildozer prints "about to run X" markers sequentially, then runs compile
  steps whose output lands later: log order is NOT execution order.

## Verifying the APK
- Expect `mobile/bin/momentum-*-arm64-v8a-debug.apk` (~46 MB).
- App code (incl. `momentum/llm/`) ships in `assets/private.tar`.
- `site-packages` ride inside `lib/arm64-v8a/libpybundle.so`, staged from
  `dists/momentum/_python_bundle__arm64-v8a`. Confirm `llama_cpp` plus the
  native `libggml-*.so` are present: "APK exists" alone does not prove the
  AI coach is bundled.

## CI relationship and pre-push hygiene
- CI derives `android.numeric_version` from the run number
  (`100000000 + GITHUB_RUN_NUMBER`); never bump it by hand.
- A local green `debug` build is the gate before pushing for the CI artifact
  build; push only a clean, tested tree.
- Split renames from behavior fixes (review-friendly commits); delete `/tmp`
  scratch (build logs, diag scripts, extracted-APK dirs) and stray files
  before pushing; run the fast pre-commit gates (ruff, ruff-format, focused
  pytest) per commit.

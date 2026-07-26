[app]

# App metadata
title = Momentum
package.name = momentum
package.domain = dev.momentum
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,md
version = 0.4.0

# Note: The AI Coach chat UI ships in every APK and degrades gracefully: when
# the native llama-cpp-python backend is absent (the default -- see Stage 2 notes
# in WARP.md / the p4a-recipes directory), the coach screen shows an informative
# "inference not available on this build" message. The model is downloaded only
# when the user opts in from the coach screen, keeping the installed APK small.
# momentum/llm/engine.py guards the llama_cpp import so it is import-safe without
# the native lib.

# Dependencies
# numpy 2.x requires C++17 which breaks on NDK r25b. Pin to 1.26.4.
# numpy git tags are v-prefixed (v1.26.4), so the pin uses 'v1.26.4' to make
# p4a run 'git checkout v1.26.4' directly. CI also patches the cloned p4a
# numpy recipe as a fallback (see .github/workflows/ci.yml, Build APK step).
# No prebuilt numpy 1.26.4 wheel exists, so numpy builds from source.
#
# pydantic is intentionally NOT included: pydantic-core is a Rust crate with
# no PyPI Android wheel, and p4a's maturin source build fails (ANDROID_API_LEVEL
# not seen in the isolated build env), and the prebuilt-wheel install is rejected
# by p4a's pure-Python pip (no --platform flag). momentum/models.py uses stdlib
# dataclasses instead, so pydantic is not needed on Android.
#
# llama-cpp-python: the production path is a prebuilt android_26 arm64-v8a wheel
# hosted in the yidaki53/p4a-wheels index and pulled via --extra-index-url (see
# p4a.extra_args below; p4a PR #3280 prebuilt-wheel support). A local p4a recipe
# (p4a-recipes/) is the in-recipe source fallback used when no prebuilt wheel
# matches; if that source build also fails, CI's build-apk job strips the recipe
# and rebuilds without it, and the AI Coach UI degrades gracefully (no on-device
# inference, but the chat screen and an informative message are present). The
# model downloads on first use only when the user opts in, so the APK stays
# small either way.
# diskcache/jinja2/typing-extensions are llama-cpp-python runtime deps (numpy is
# already listed); they are pure-Python and must be in the APK or Llama import/use
# fails on-device.
requirements = python3,kivy,pillow,matplotlib,numpy==v1.26.4,certifi,llama-cpp-python,diskcache,jinja2,typing-extensions

# Include the core momentum package (via symlink) and data files
source.include_patterns = main.py,momentum/*.py,momentum/**/*.py,momentum/**/**/*.py,ENCOURAGEMENTS.md,SCIENCE.md,README.md,IMAGES.md

# App icon and presplash
icon.filename = icon.png
presplash.filename = presplash.png

# Android settings
# REQUEST_INSTALL_PACKAGES: required since Android 8 (API 26) for the
# self-update flow -- the in-app "Update now" enqueues the APK via
# DownloadManager and the completion notification opens the package installer.
android.permissions = INTERNET,VIBRATE,WAKE_LOCK,REQUEST_INSTALL_PACKAGES
android.api = 34
android.minapi = 26
android.ndk = 25b
android.accept_sdk_license = True
android.numeric_version = 100000003
android.release_artifact = apk
android.debug_artifact = apk
# v1 ships arm64-v8a only: every API 26+ device is arm64, and a single prebuilt
# android_26_arm64_v8a wheel then deterministically satisfies p4a's per-arch
# prebuilt lookup. Building armeabi-v7a too would require an android_26_arm wheel
# in the index, and a missing armeabi-v7a prebuilt would make p4a fall back to a
# source build for that arch, which (on failure) trips CI's strip-and-rebuild
# fallback and removes llama from BOTH arches. Add android_26_arm later to restore
# 32-bit support.
android.archs = arm64-v8a
# allowBackup=true enables Android Auto Backup: all app-private data (the DB,
# config, and downloaded models under getFilesDir()/data) is backed up to the
# user's Google account and transferred to a new device by default. This pairs
# with Android's built-in preservation of app-private storage across app
# UPDATES (which is independent of this flag) so user data is kept both between
# updates and across device transfers. No backup-rules XML is needed because we
# want to preserve everything (rules files are only required to EXCLUDE data).
android.allow_backup = True

# App appearance
orientation = portrait
fullscreen = 0

# Build
log_level = 2

# python-for-android wiring -- all p4a.* tokens MUST live under [app]; buildozer
# reads them from [app] (targets/android.py getdefault('app', ...)), NOT [buildozer].
# A previous version had p4a.local_recipes under [buildozer], so the llama recipe was
# silently never registered with p4a (it was only bundled as app source data) and p4a
# fell back to pip-installing llama-cpp-python from PyPI, which failed.
# Local recipes directory for the llama-cpp-python source recipe (see p4a-recipes/).
# CI strips this line in the fallback pass when the recipe fails to build.
p4a.local_recipes = p4a-recipes
# Pull a prebuilt android_26 arm64-v8a wheel for llama-cpp-python from our
# self-hosted PEP 503 index (yidaki53/p4a-wheels, served via GitHub Pages at
# https://yidaki53.github.io/p4a-wheels/p4a). p4a PR #3280 added prebuilt-wheel
# support (--extra-index-url/--platform/--only-binary); buildozer 1.5.0 predates
# the first-class spec tokens (extra_index_urls/p4a.save_wheel_dir, buildozer
# commit 79ad6d5), so we forward the flag via p4a.extra_args (read from [app]).
# PyProjectRecipe.check_prebuilt runs first and installs the wheel via pip
# --platform if found; otherwise it falls back to the source recipe, then CI's
# strip-and-rebuild fallback ships a no-llama APK (graceful degradation).
p4a.extra_args = --extra-index-url=https://yidaki53.github.io/p4a-wheels/p4a

# iOS (future)
ios.kivy_ios_url = https://github.com/kivy/kivy-ios
ios.kivy_ios_branch = master

[buildozer]
log_level = 2
warn_on_root = 1
build_dir = /tmp/buildozer-build
# buildozer clones python-for-android fresh from GitHub by default. p4a.source only
# takes effect when it points at a git repo or tarball URL -- a site-packages path is
# ignored and buildozer falls back to cloning. CI relies on that fresh clone and patches
# the cloned numpy recipe in the Build APK step. Set p4a.source only if you want a
# specific local p4a git checkout for local builds.

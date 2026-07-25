[app]

# App metadata
title = Momentum
package.name = momentum
package.domain = dev.momentum
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,md
version = 0.4.0

# Note: AI Coach / LLM features are opt-in on mobile because llm/engine.py
# tries to import llama-cpp-python, which is not available via buildozer by default.
# Users who manually enable it will need to install the runtime dependency separately.

# Dependencies
# numpy 2.x requires C++17 which breaks on NDK r25b. Pin to 1.26.4.
# numpy git tags are v-prefixed (v1.26.4), so the pin uses 'v1.26.4' to make
# p4a run 'git checkout v1.26.4' directly. CI also patches the cloned p4a
# numpy recipe as a fallback (see .github/workflows/ci.yml, Build APK step).
# No prebuilt numpy 1.26.4 wheel exists, so numpy builds from source.
#
# pydantic + pydantic-core: pydantic-core is a Rust crate with no PyPI Android
# wheel, and p4a's maturin source build fails (ANDROID_API_LEVEL not seen in
# the isolated build env). Instead we pull a prebuilt Android wheel from the
# community p4a-wheels index (p4a.extra_args below). That index hosts
# pydantic_core-2.41.4 (cp314, android_24_*), and pydantic 2.12.3 requires
# pydantic-core==2.41.4, so the pair is pinned to match the prebuilt.
# android.minapi is 24 to match the android_24_* prebuilt platform tags.
requirements = python3,kivy,pydantic==2.12.3,pydantic-core==2.41.4,pillow,matplotlib,numpy==v1.26.4,certifi

# Prebuilt Android wheels (p4a PR #3280). The community index hosts cp314 /
# android_24_* wheels built with NDK r25b for p4a's target Python 3.14.
# p4a checks each recipe for a prebuilt first and falls back to source build.
# pydantic-core uses the prebuilt (bypasses maturin); numpy falls back to
# source (no 1.26.4 prebuilt). If this index is down, pydantic-core falls
# back to source build and the APK build fails -- pin a mirror if needed.
p4a.extra_args = --extra-index-url https://anshdadwal.is-a.dev/p4a-wheels/p4a/

# Include the core momentum package (via symlink) and data files
source.include_patterns = main.py,momentum/*.py,momentum/**/*.py,momentum/**/**/*.py,ENCOURAGEMENTS.md,SCIENCE.md,README.md,IMAGES.md

# App icon and presplash
icon.filename = icon.png
presplash.filename = presplash.png

# Android settings
android.permissions = INTERNET,VIBRATE,WAKE_LOCK
android.api = 34
android.minapi = 24
android.ndk = 25b
android.accept_sdk_license = True
android.numeric_version = 100000003
android.release_artifact = apk
android.debug_artifact = apk

# App appearance
orientation = portrait
fullscreen = 0

# Build
log_level = 2

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

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
# Pin numpy to 1.26.4 for Android NDK compatibility (numpy 2.x C++17 breaks on r25b)
# Use single = for buildozer requirements syntax
requirements = python3,kivy,pydantic,pydantic-core,pillow,matplotlib,numpy==1.26.4,certifi

# Include the core momentum package (via symlink) and data files
source.include_patterns = main.py,momentum/*.py,momentum/**/*.py,momentum/**/**/*.py,ENCOURAGEMENTS.md,SCIENCE.md,README.md,IMAGES.md

# App icon and presplash
icon.filename = icon.png
presplash.filename = presplash.png

# Android settings
android.permissions = INTERNET,VIBRATE,WAKE_LOCK
android.api = 34
android.minapi = 26
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

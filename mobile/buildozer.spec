[app]

# App metadata
title = Momentum
package.name = momentum
package.domain = dev.momentum
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 0.1.0

# Dependencies
requirements = python3,kivy,pydantic,pydantic-core

# Include the core momentum package
source.include_patterns = main.py,../momentum/*.py

# Android settings
android.permissions = VIBRATE,WAKE_LOCK
android.api = 33
android.minapi = 26
android.ndk = 25b
android.accept_sdk_license = True

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

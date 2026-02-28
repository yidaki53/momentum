[app]

# App metadata
title = Momentum
package.name = momentum
package.domain = dev.momentum
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,md
version = 0.1.0

# Dependencies
requirements = python3,kivy,pydantic,pydantic-core,pillow,matplotlib,numpy

# Include the core momentum package (via symlink) and data files
source.include_patterns = main.py,momentum/*.py,ENCOURAGEMENTS.md,SCIENCE.md,README.md,IMAGES.md

# App icon and presplash
icon.filename = icon.png
presplash.filename = presplash.png

# Android settings
android.permissions = INTERNET,VIBRATE,WAKE_LOCK
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
build_dir = /media/robin/persistence/buildozer-build

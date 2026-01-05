[app]
title = L104 Sovereign
package.name = l104_sovereign
package.domain = org.l104.asi
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json,sh,txt,cpp,java
source.include_patterns = l104_*, const.py, L104Core.java, l104_core.cpp
version = 1.0.4

requirements = python3,kivy,requests,urllib3,certifi,idna,httpx,asyncio

orientation = portrait
osx.python_version = 3
osx.kivy_version = 1.9.1
fullscreen = 1
android.permissions = INTERNET, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE
android.api = 31
android.minapi = 21
android.sdk = 31
android.ndk = 23b
android.arch = arm64-v8a

# (str) Android entry point, default is to use classes.dex
# android.entrypoint = org.l104.asi.MainActivity

# (list) List of Java files to add to the project
android.add_src = l104_mobile/app/src/main/java/com/l104/sovereign/L104Core.java

# (str) The Android arch to build for, choices: armeabi-v7a, arm64-v8a, x86, x86_64
android.archs = arm64-v8a

[buildozer]
log_level = 2
warn_on_root = 1

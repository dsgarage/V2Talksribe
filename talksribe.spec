# -*- mode: python ; coding: utf-8 -*-
import os
import sys

block_cipher = None

src_dir = os.path.join(SPECPATH, 'src')

a = Analysis(
    [os.path.join(src_dir, 'menubar.py')],
    pathex=[src_dir],
    binaries=[],
    datas=[
        (os.path.join(src_dir, 'mic_icon.png'), '.'),
        (os.path.join(src_dir, 'audio_worker.py'), '.'),
    ],
    hiddenimports=[
        'rumps',
        'sounddevice',
        'numpy',
        'torch',
        'torchaudio',
        'faster_whisper',
        'ctranslate2',
        'huggingface_hub',
        'tokenizers',
        'librosa',
        'soundfile',
        'speechbrain',
        'sklearn',
        'sklearn.cluster',
        'audio_worker',
        '_sounddevice_data',
        'multiprocessing.resource_tracker',
        'multiprocessing.spawn',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'IPython', 'jupyter'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Talksribe',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Talksribe',
)

app = BUNDLE(
    coll,
    name='Talksribe.app',
    icon=None,
    bundle_identifier='com.dsgarage.talksribe',
    info_plist={
        'CFBundleName': 'Talksribe',
        'CFBundleDisplayName': 'Talksribe',
        'CFBundleVersion': '1.1.0',
        'CFBundleShortVersionString': '1.1.0',
        'LSUIElement': True,
        'NSMicrophoneUsageDescription': 'Talksribe はリアルタイム文字起こしのためにマイクを使用します。',
        'LSMinimumSystemVersion': '14.0',
    },
)

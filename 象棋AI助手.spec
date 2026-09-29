# -*- mode: python ; coding: utf-8 -*-
import os


# Pikafish 引擎本体不进仓库（GPL-3.0），但本地构建时要带进产物。
# 仓库里没有 engine/ 时自动跳过，不会因为缺目录而构建失败。
_extra_datas = [('engine', 'engine')] if os.path.isdir('engine') else []

a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=[],
    datas=[('ad_close.png', '.')] + _extra_datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='象棋AI助手',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='象棋AI助手',
)

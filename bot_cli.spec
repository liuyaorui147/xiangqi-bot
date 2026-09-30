# -*- mode: python ; coding: utf-8 -*-
"""后端 bot 的控制台版打包配置（供外部界面/脚本以子进程方式调用）。

和 象棋AI助手.spec 的区别：
  * 入口是 main.py（纯命令行），而不是图形界面 gui.py
  * console=True —— 必须。windowed 程序没有 stdout，调用方读不到任何日志
  * 名为 象棋Bot.exe，输出目录带 bot/ 前缀，被上层当作后端调用
"""
import os


# 棋子模板 + 开始界面「10分钟场」按钮模板 + 引擎本体
_extra_datas = [d for d in (('engine', 'engine'), ('templates', 'templates'))
                if os.path.isdir(d[0])]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('ad_close.png', '.')] + _extra_datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter'],          # 后端不用的 GUI 库，能省一点体积
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='象棋Bot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,                  # 关键：保留 stdout，界面靠它收日志
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
    name='象棋Bot',
)

"""运行日志：把 stdout 同时落盘，事后能凭日志还原现场。

为什么需要它
------------
界面上的日志区只有几百行滚动缓冲，程序停在半路、用户关掉窗口、或者日志
被后续输出刷掉之后就彻底没有现场了。排查"一直思考不落子"这类问题时只能
靠截图猜。现在每次启动写一份 logs/run_YYYYMMDD_HHMMSS.log，内容包括：
    - 运行头：时间、命令行参数、关键配置、App 包名、标定来源、模板数
    - 每一步：步号、识别出的局面、引擎评价、胜率、阶段、点击结果
    - 每次失败/拒绝/停止的原因
出了问题打开对应文件往下翻即可，"第几步、什么局面、引擎说了什么、为什么会停"
都在里面。

用法
----
    import logger
    logger.install()        # main() 开头调一次，之后所有 print 都进文件
    logger.header(...)      # 写运行头
    logger.path()           # 当前日志文件路径

设计要点
--------
* 文件里每行带时间戳，界面上不带——界面的日志区本来就窄，再加前缀更难读。
  做法是给原 stdout 转发原始内容、给文件写加工后的版本。
* 单句柄 + 锁，跨线程写不会串行。worker 线程和 GUI 主线程都在用 stdout。
* 自动轮转：只保留最近 KEEP 份，避免日志自己把磁盘吃满。
* 静默超过 TS_EVERY 秒后插入时间分隔（例如等对手几十秒），方便跨时间段定位。
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time

LOG_DIR = "logs"
KEEP = 15             # 保留最近几份日志
TS_EVERY = 120.0      # 静默超过这么久就插一行时间，便于分段定位

# 行首已带时间戳的输出（main.py 的"轮到我方""局面变化"等），别再补一层
_STAMP_RE = re.compile(r"^\[\d{2}:\d{2}:\d{2}\]")

_lock = threading.Lock()
_fh = None
_path = None
orig_stdout = None
_next_stamp = [0.0]


class _Tee:
    """把写入分给文件和原 stdout：文件加工时间戳，界面保持原样。"""

    def __init__(self, orig):
        self.orig = orig
        # 还没以换行收尾的残段。存下来的原因：主循环等对手时用 print(".", end="")
        # 打点，一行行收会变成日志里几百个孤零零的点；等到换行（或下一行内容的
        # 开头）再落盘，这些点才能拼成一行 "..."。
        self._tail = ""

    def write(self, s):
        if not s:
            return
        try:
            self.orig.write(s)      # 界面照常显示原始内容
        except Exception:
            pass
        chunk = self._tail + s
        parts = chunk.split("\n")
        # 最后一段：chunk 以换行收尾时是空串（没有残段），否则就是未写完的行
        complete = [p for p in parts[:-1] if p]
        self._tail = "" if s.endswith("\n") else parts[-1]
        if not complete:
            return
        now = time.time()
        stamp = time.strftime("%H:%M:%S")
        with _lock:
            if _fh is None:
                return
            if now >= _next_stamp[0]:
                _fh.write(f"\n----- {time.strftime('%Y-%m-%d %H:%M:%S')} -----\n")
            _next_stamp[0] = now + TS_EVERY
            # main.py 自己的几处提示已经带了 [HH:MM:SS]（如"[12:00:00] 轮到我方"），
            # 这里再补一个就成了 "[12:00:00] [12:00:00]"，跳过已有前缀的行。
            lines = [seg if _STAMP_RE.match(seg) else f"[{stamp}] {seg}"
                     for seg in complete]
            _fh.write("\n".join(lines) + "\n")
            _fh.flush()

    def flush(self):
        try:
            self.orig.flush()
        except Exception:
            pass
        with _lock:
            if _fh is not None:
                try:
                    _fh.flush()
                except Exception:
                    pass

    def isatty(self):
        return False


def _rotate():
    """删掉超出 KEEP 的旧日志（按名字排序，越靠后越新）。"""
    try:
        names = sorted(n for n in os.listdir(LOG_DIR)
                       if n.startswith("run_") and n.endswith(".log"))
        for n in names[:-KEEP]:
            try:
                os.remove(os.path.join(LOG_DIR, n))
            except OSError:
                pass
    except OSError:
        pass


def install(log_dir=LOG_DIR):
    """接管 stdout，开始记录。可重复调用，第二次起只对首次生效。"""
    global _fh, _path, orig_stdout
    with _lock:
        if _fh is not None:
            return _path
        try:
            os.makedirs(log_dir, exist_ok=True)
            _path = os.path.join(log_dir, time.strftime("run_%Y%m%d_%H%M%S.log"))
            _fh = open(_path, "a", encoding="utf-8", errors="replace")
        except OSError as e:     # 只读目录/权限问题不该让主程序崩
            print(f"[logger] 无法打开日志文件: {e}")
            return None
        orig_stdout = sys.stdout
        sys.stdout = _Tee(orig_stdout)
        _rotate()
        return _path


def uninstall():
    """恢复 stdout（正常收尾时调用；不调用atexit 也会兜底 flush）。"""
    global _fh, orig_stdout
    with _lock:
        if _fh is None:
            return
        try:
            _fh.flush()
            _fh.close()
        except Exception:
            pass
        _fh = None
        if orig_stdout is not None:
            sys.stdout = orig_stdout
            orig_stdout = None


def path():
    return _path


def header(title, lines):
    """写一段带标题的键值块，用于运行头。不经过 print，免得界面重复。"""
    write_raw("")
    write_raw(f"===== {title} =====")
    for line in lines:
        write_raw(line)


def write_raw(text):
    """直接往日志写一行（不进界面）。"""
    with _lock:
        if _fh is None:
            return
        try:
            _fh.write(f"[{time.strftime('%H:%M:%S')}] {text}\n")
            _fh.flush()
        except Exception:
            pass

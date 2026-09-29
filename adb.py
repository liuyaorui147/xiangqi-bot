"""adb 后端：截屏与点击都走 Android 层，绕开 Windows 输入注入被拦的问题。

坐标体系：screencap 出来的图像像素 == input tap 使用的物理像素，天然一致，
不存在 DPI / 窗口边框 / 遮挡问题。
"""
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np


class AdbError(RuntimeError):
    """adb 调用失败（找不到 adb / 超时 / 设备掉线等）。

    以前这些错误安静地藏在返回码里，调用方拿到空输出继续跑，
    设备掉线表现为"一直识别不到棋盘"而不是"adb 挂了"。
    """


_BAD_EXT = (".py", ".pyw", ".pyc", ".txt", ".md")


def _which_adb():
    """在 PATH 里找 adb。

    Windows 的 PATHEXT 含 .PY，且 shutil.which 会搜当前目录，所以直接
    shutil.which("adb") 会把本项目自己的 adb.py 当成可执行文件返回——
    必须按扩展名和所在目录排除掉。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("adb", "adb.exe"):
        p = shutil.which(name)
        if not p:
            continue
        p = os.path.abspath(p)
        if os.path.dirname(p) == here:                     # 命中自己这个脚本
            continue
        if os.path.splitext(p)[1].lower() in _BAD_EXT:     # 命中某个 .py
            continue
        return p
    return None


def _find_adb():
    """定位 adb 可执行文件，按优先级：
    环境变量 XIANGQI_ADB/ADB_PATH -> PATH -> ANDROID_HOME/SDK_ROOT -> 常见安装位置。

    以前硬编码 C:\\Users\\Administrator\\...\\adb.exe，换台机器直接跑不了。
    """
    for var in ("XIANGQI_ADB", "ADB_PATH"):
        p = os.environ.get(var)
        if p and os.path.exists(p):
            return os.path.abspath(p)

    found = _which_adb()
    if found:
        return found

    cands = []
    for var in ("ANDROID_HOME", "ANDROID_SDK_ROOT", "ANDROID_SDK_HOME"):
        root = os.environ.get(var)
        if root:
            cands += [os.path.join(root, "platform-tools", n) for n in ("adb.exe", "adb")]
    home = os.path.expanduser("~")
    cands += [
        os.path.join(home, "AppData", "Local", "Android", "Sdk", "platform-tools", "adb.exe"),
        os.path.join(home, "Android", "Sdk", "platform-tools", "adb.exe"),
        os.path.join("/usr", "bin", "adb"),
        os.path.join("/usr", "local", "bin", "adb"),
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


ADB = _find_adb()

# 最近一次 adb 失败的原因，供调用方/界面显示
LAST_ERROR = [None]

# adb / 引擎都是控制台程序，不加这个标志每调用一次就会闪一个 cmd 窗口
NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

# adb 断连的典型提示，用来把"设备掉线"和"命令本身失败"区分开
_OFFLINE_HINTS = ("no devices/emulators found", "device offline", "device not found",
                  "closed", "cannot connect")


def _diag(rc, out, err):
    """把一次 adb 调用的失败信息整理成人话。"""
    msg = (err or b"").decode("utf-8", "replace").strip()
    low = msg.lower()
    if any(h in low for h in _OFFLINE_HINTS):
        return f"adb 设备离线或未连接: {msg or '(无输出)'}"
    if rc != 0:
        return f"adb 返回 {rc}: {msg or (out or b'').decode('utf-8','replace').strip()[:200]}"
    return None


def _run(args, timeout=15, serial=None, check=False):
    """执行 adb 命令。check=True 时失败直接抛 AdbError，不再让错误静默传播。"""
    if ADB is None:
        raise AdbError("找不到 adb：请安装 Android platform-tools，"
                       "或设置 XIANGQI_ADB 环境变量指向 adb 可执行文件")
    cmd = [ADB]
    if serial:
        cmd += ["-s", serial]
    cmd += args
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout,
                           creationflags=NO_WINDOW)
    except subprocess.TimeoutExpired:
        LAST_ERROR[0] = f"adb {' '.join(args)} 超时 {timeout}s"
        if check:
            raise AdbError(LAST_ERROR[0])
        return 124, b"", LAST_ERROR[0].encode()
    except FileNotFoundError:
        LAST_ERROR[0] = f"adb 可执行文件不存在: {ADB}"
        if check:
            raise AdbError(LAST_ERROR[0])
        return 127, b"", LAST_ERROR[0].encode()
    except OSError as e:
        LAST_ERROR[0] = f"adb 启动失败: {e}"
        if check:
            raise AdbError(LAST_ERROR[0])
        return 1, b"", str(e).encode()

    LAST_ERROR[0] = _diag(p.returncode, p.stdout, p.stderr)
    if check and p.returncode != 0:
        raise AdbError(LAST_ERROR[0] or f"adb {' '.join(args)} 返回 {p.returncode}")
    return p.returncode, p.stdout, p.stderr


def adb_path():
    """返回当前使用的 adb 路径（找不到时为 None），便于诊断。"""
    return ADB


def devices(timeout=10):
    """返回 [(序列号, 状态)]，只保留 state=device 的项由调用方判断。

    adb 本身失败（未安装/超时）时返回空列表，并把原因写进 LAST_ERROR。
    """
    try:
        rc, out, err = _run(["devices", "-l"], timeout=timeout, check=True)
    except AdbError as e:
        LAST_ERROR[0] = str(e)
        print(f"[adb] 获取设备列表失败: {e}")
        return []
    res = []
    for line in out.decode("utf-8", "replace").splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            res.append((parts[0], parts[1]))
    return res


def first_device(timeout=10):
    ds = devices(timeout)
    for s, st in ds:
        if st == "device":
            return s
    return None


def connect(host="127.0.0.1", port=5555, timeout=12):
    rc, out, err = _run(["connect", f"{host}:{port}"], timeout=timeout)
    msg = (out.decode("utf-8", "replace") + err.decode("utf-8", "replace")).strip()
    return rc == 0 or "connected" in msg or "already" in msg, msg


def screencap(serial=None, timeout=15):
    """返回 BGR numpy 图像，或 None。

    失败时把原因写进 LAST_ERROR——以前只检查返回数据长度，
    adb 掉线和"PNG 被截断"在调用方看来是同一种"返回 None"，无从分辨。
    """
    if ADB is None:
        LAST_ERROR[0] = "找不到 adb（未安装或未设置 XIANGQI_ADB）"
        return None
    try:
        rc, buf, err = _run(["exec-out", "screencap", "-p"], timeout=timeout, serial=serial)
    except AdbError as e:
        LAST_ERROR[0] = str(e)
        return None
    if rc != 0:
        LAST_ERROR[0] = LAST_ERROR[0] or f"screencap 失败 rc={rc}"
        return None
    if not buf or len(buf) < 1024:
        LAST_ERROR[0] = (f"screencap 返回数据过小（{len(buf) if buf else 0} 字节），"
                         "exec-out 可能被截断")
        return None
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        LAST_ERROR[0] = "screencap 图像解码失败（PNG 数据不完整）"
    return img


def tap(x, y, serial=None, timeout=10):
    return _run(["shell", "input", "tap", str(int(x)), str(int(y))],
                timeout=timeout, serial=serial)[0] == 0


def swipe(x0, y0, x1, y1, dur=150, serial=None, timeout=10):
    return _run(["shell", "input", "swipe", str(int(x0)), str(int(y0)),
                 str(int(x1)), str(int(y1)), str(int(dur))],
                timeout=timeout, serial=serial)[0] == 0


def key(keycode, serial=None, timeout=10):
    """发送按键事件，如 4=BACK。"""
    return _run(["shell", "input", "keyevent", str(int(keycode))],
                timeout=timeout, serial=serial)[0] == 0


def wm_size(serial=None):
    rc, out, _ = _run(["shell", "wm", "size"], serial=serial)
    s = out.decode("utf-8", "replace").strip()
    if ":" in s:
        try:
            a, b = s.split(":")[1].strip().split("x")
            return int(a), int(b)
        except Exception:
            pass
    return None


def top_activity(serial=None):
    rc, out, _ = _run(["shell", "dumpsys", "activity", "activities"], timeout=20, serial=serial)
    s = out.decode("utf-8", "replace")
    for line in s.splitlines():
        if "topResumedActivity" in line or "mResumedActivity" in line:
            return line.strip()[:120]
    return None


if __name__ == "__main__":
    print("adb 路径:", adb_path() or "未找到（请安装 platform-tools 或设 XIANGQI_ADB）")
    print("全部设备:", devices())
    s = first_device()
    print("可用设备:", s)
    if s:
        img = screencap(s)
        print("截屏:", None if img is None else f"{img.shape[1]}x{img.shape[0]}")
        cv2.imwrite("shots/adb_shot.png", img) if img is not None else None
        print("分辨率:", wm_size(s))
        print("前台 Activity:", top_activity(s))

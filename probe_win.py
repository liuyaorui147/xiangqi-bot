"""用 Windows API 枚举可见窗口 + 探测模拟器 adb 端口。"""
import ctypes
import subprocess
import re
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

user32.EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]
user32.EnumWindows.restype = wintypes.BOOL
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]


def enum_windows(visible_only=True):
    win = []

    @WNDENUMPROC
    def _cb(hwnd, lparam):
        if visible_only and not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        cls = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, cls, 256)
        r = wintypes.RECT()
        ok = user32.GetWindowRect(hwnd, ctypes.byref(r))
        rect = (r.left, r.top, r.right - r.left, r.bottom - r.top) if ok else None
        win.append({"hwnd": hwnd, "title": buf.value, "class": cls.value, "rect": rect})
        return True

    user32.EnumWindows(_cb, 0)
    return win


def get_exe(pid):
    try:
        p = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, encoding="gbk", errors="ignore",
        )
        for line in p.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) >= 2:
                return parts[0].strip('"')
    except Exception:
        pass
    return "?"


if __name__ == "__main__":
    print("=" * 92)
    print("可见窗口列表")
    print("=" * 92)
    pids = {}
    for w in enum_windows():
        if not w["title"].strip():
            continue
        print(f"hwnd={w['hwnd']:<10} [{w['class']:<26}] {w['rect']}  {w['title'][:60]}")

    print()
    print("=" * 92)
    print("监听中的 TCP 端口（找模拟器 adb）")
    print("=" * 92)
    out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True,
                         encoding="gbk", errors="ignore").stdout
    seen = {}
    for line in out.splitlines():
        if "LISTENING" in line:
            f = line.split()
            addr, pid = f[1], f[-1]
            m = re.match(r"[\d.]+:(\d+)$", addr)
            if not m:
                continue
            port = int(m.group(1))
            if pid not in pids:
                pids[pid] = get_exe(pid)
            seen.setdefault(port, (pid, pids[pid]))
    for port in sorted(seen):
        pid, name = seen[port]
        flag = "  <== 疑似 adb" if port in (5555, 5554, 5037, 62001, 7555, 21503, 6555, 26944) else ""
        print(f"  {port:<8} pid={pid:<8} {name}{flag}")

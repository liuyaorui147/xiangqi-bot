"""MuMu 模拟器共享内存截图（external_renderer_ipc）。

走网易官方 SDK（MuMu 安装目录自带的 external_renderer_ipc.dll），
直接读模拟器帧缓冲，**不走 adb**。实测 900x1600 单帧约 5.5ms，
是 adb exec-out screencap（约 640ms）的百分之一量级。

调用方只管 open()/screencap()/close()：
  - open() 找不到 MuMu / 连接失败返回 False，调用方回退 adb 即可
  - screencap() 连续失败会自动停用，之后恒返回 None（调用方走 adb）
  - 点击不走这里（仍由 adb tap 承担），本模块只管"看"

DLL 与安装目录搜索顺序：
  1. 环境变量 MUMU_INSTALL_PATH / MUMU_INSTALL
  2. bot_config.json 里的 "mumu_path"（open 时传进来）
  3. 常见安装位置（C/D/E 盘 Program Files）

实例与 DLL 版本必须匹配：vms/MuMuPlayer-15.0-0 的实例要用
nx_device/15.0 的 DLL；12.0 布局则用 shell/sdk 或 nx_device/12.0。
"""
import glob
import os
import threading
import time

import ctypes
import cv2
import numpy as np

# nemu_capture_display 的返回码：0/1 成功，>1 失败
_OK = (0, 1)
_MAX_FAILS = 5          # 连续失败多少次后停用（调用方回退 adb）
_DLL_RELPATHS = (
    r"nx_device\15.0\shell\sdk\external_renderer_ipc.dll",
    r"nx_device\12.0\shell\sdk\external_renderer_ipc.dll",
    r"shell\sdk\external_renderer_ipc.dll",
    r"nx_main\sdk\external_renderer_ipc.dll",
)
_COMMON_INSTALLS = (
    r"C:\Program Files\Netease\MuMu",
    r"D:\Program Files\Netease\MuMu",
)

_LOCK = threading.Lock()


def _load_dll(path):
    dll = ctypes.WinDLL(path)
    dll.nemu_connect.restype = ctypes.c_int
    # 注意：官方要宽字符串（c_wchar_p）。传 bytes 会静默返回 0。
    dll.nemu_connect.argtypes = [ctypes.c_wchar_p, ctypes.c_int]
    dll.nemu_disconnect.argtypes = [ctypes.c_int]
    dll.nemu_capture_display.restype = ctypes.c_int
    dll.nemu_capture_display.argtypes = [
        ctypes.c_int, ctypes.c_uint, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_ubyte),
    ]
    return dll


def find_install(config_path=None):
    """返回 MuMu 安装根目录（含 vms/），找不到返回 None。"""
    for var in ("MUMU_INSTALL_PATH", "MUMU_INSTALL"):
        p = os.environ.get(var)
        if p and os.path.isdir(os.path.join(p, "vms")):
            return p
    if config_path and os.path.isdir(os.path.join(config_path, "vms")):
        return config_path
    cands = list(_COMMON_INSTALLS)
    for drive in "CDEFGH":
        cands += glob.glob(f"{drive}:\\Program Files\\Netease\\MuMuPlayer*")
        cands += glob.glob(f"{drive}:\\Program Files (x86)\\Netease\\MuMuPlayer*")
    for p in cands:
        if os.path.isdir(os.path.join(p, "vms")):
            return p
    return None


def _instances(install):
    """从 vms/ 枚举实例 [(index, (主,次))]，如 [(0, (15, 0))]。"""
    out = []
    for d in sorted(glob.glob(os.path.join(install, "vms", "MuMuPlayer-*"))):
        name = os.path.basename(d)          # MuMuPlayer-15.0-0
        try:
            _, ver, idx = name.rsplit("-", 2)
            out.append((int(idx), tuple(int(x) for x in ver.split(".")[:2])))
        except ValueError:
            continue
    return out


class MuMuCap:
    def __init__(self, config_path=None, verbose=False):
        self.install = find_install(config_path)
        self.handle = 0
        self.w = self.h = 0
        self.buf = None
        self.dll = None
        self.fails = 0
        self.verbose = verbose
        self._has_input = False
        self._why = "" if self.install else "未找到 MuMu 安装目录（可设 MUMU_INSTALL_PATH）"

    @property
    def available(self):
        return bool(self.handle) and self.fails < _MAX_FAILS

    @property
    def last_error(self):
        return self._why

    def open(self):
        """连接模拟器主屏。成功返回 True。"""
        if self.available:
            return True
        if not self.install:
            return False
        for idx, ver in _instances(self.install):
            dll = self._dll_for(ver)
            if dll is None:
                continue
            try:
                h = dll.nemu_connect(self.install, idx)
            except OSError:
                continue
            if not h:
                continue
            w, hh = ctypes.c_int(), ctypes.c_int()
            rc = dll.nemu_capture_display(h, 0, 0, ctypes.byref(w), ctypes.byref(hh), None)
            # 宽和高都要校验：只查宽度的话，高度为 0 会让缓冲区长度变成 0
            # （w*0*4），后续 reshape 出一个空数组，cv2 任何操作都会抛
            # "!_src.empty()" —— 模拟器刚启动/渲染未就绪时 SDK 就会返回 0x0。
            if rc not in _OK or not w.value or not hh.value:
                dll.nemu_disconnect(h)
                continue
            self.handle = h
            self.w, self.h = w.value, hh.value
            self.buf = (ctypes.c_ubyte * (self.w * self.h * 4))()
            self.dll = dll
            self.fails = 0
            # 输入接口（与截图同一条 IPC 通道，~1ms/次 vs adb tap ~80ms）
            try:
                dll.nemu_input_event_touch_down.restype = ctypes.c_int
                dll.nemu_input_event_touch_down.argtypes = [ctypes.c_int] * 4
                dll.nemu_input_event_touch_up.restype = ctypes.c_int
                dll.nemu_input_event_touch_up.argtypes = [ctypes.c_int] * 2
                self._has_input = True
            except AttributeError:
                self._has_input = False
            if self.verbose:
                print(f"[mumu] 已连接实例 {idx}（{self.w}x{self.h}，共享内存截图）")
            return True
        self._why = "所有实例连接失败（模拟器没在运行？）"
        return False

    def _dll_for(self, ver):
        want = rf"nx_device\{ver[0]}.{ver[1]}" + r"\shell\sdk\external_renderer_ipc.dll"
        ordered = [want] + [r for r in _DLL_RELPATHS if r != want]
        for rel in ordered:
            p = os.path.join(self.install, rel)
            if os.path.exists(p):
                try:
                    return _load_dll(p)
                except OSError as e:
                    self._why = f"DLL 加载失败: {e}"
        return None

    def screencap(self):
        """返回 BGR numpy 图（与 adb screencap 同构），失败返回 None。"""
        if not self.available and not self.open():
            return None
        with _LOCK:
            w, hh = ctypes.c_int(), ctypes.c_int()
            rc = self.dll.nemu_capture_display(
                self.handle, 0, len(self.buf),
                ctypes.byref(w), ctypes.byref(hh), self.buf)
            if rc not in _OK:
                self.fails += 1
                if self.fails >= _MAX_FAILS:
                    self._why = f"连续 {self.fails} 次 capture 失败(rc={rc})，已停用"
                    self.close()
                return None
            # 分辨率变了（旋转/换设备）：旧缓冲区会错位，重连换新缓冲
            if (w.value, hh.value) != (self.w, self.h):
                self.close()
                if not self.open():
                    return None
            # rc 报成功但尺寸为 0 会让缓冲区为空 —— 直接丢弃这一帧，
            # 否则空图传出去，调用方 cv2 一碰就崩（!_src.empty()）。
            if not w.value or not hh.value:
                self.fails += 1
                if self.fails >= _MAX_FAILS:
                    self._why = f"capture 返回空分辨率 {w.value}x{hh.value}，已停用"
                    self.close()
                return None
        self.fails = 0
        if not self.w or not self.h or self.w * self.h * 4 != len(self.buf):
            self._why = f"缓冲区尺寸不匹配 {self.w}x{self.h} vs {len(self.buf)}"
            self.close()
            return None
        arr = np.frombuffer(self.buf, dtype=np.uint8).reshape(self.h, self.w, 4)
        img = cv2.flip(cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR), 0)
        return img if img.size else None

    def tap(self, x, y, hold=0.02):
        """SDK 触摸点击（设备坐标系，与 adb tap 一致）。成功返回 True。

        hold 是按下到抬起的时长。实测 2ms 就能让游戏收到（选中反馈完整），
        原来写死 50ms 是照搬 adb 时代的保守值，纯浪费 —— 每步两次点击
        就是 100ms。这里取 20ms，对 2ms 的实测下限留 10 倍余量。
        失败计数与截图共用：连续失败达到上限会整体停用并回退 adb。
        """
        if not self.available and not self.open():
            return False
        if not self._has_input:
            self._why = "SDK 无输入接口"
            return False
        with _LOCK:
            try:
                rc1 = self.dll.nemu_input_event_touch_down(
                    self.handle, 0, int(x), int(y))
                time.sleep(hold)
                rc2 = self.dll.nemu_input_event_touch_up(self.handle, 0)
            except Exception as e:
                self.fails += 1
                self._why = f"tap 异常: {e}"
                return False
        if rc1 != 0 or rc2 != 0:
            self.fails += 1
            self._why = f"tap rc=({rc1},{rc2})"
            if self.fails >= _MAX_FAILS:
                self.close()
            return False
        self.fails = 0
        return True

    def close(self):
        if self.handle:
            try:
                self.dll.nemu_disconnect(self.handle)
            except Exception:
                pass
        self.handle = 0
        self.buf = None


if __name__ == "__main__":
    cap = MuMuCap(verbose=True)
    print("安装目录:", cap.install)
    print("open ->", cap.open())
    import time
    ts = []
    img = None
    for _ in range(30):
        t0 = time.perf_counter()
        img = cap.screencap()
        ts.append((time.perf_counter() - t0) * 1000)
    print("screencap x30: 平均 %.1fms" % (sum(ts) / len(ts)))
    print("图像:", None if img is None else img.shape)
    cap.close()

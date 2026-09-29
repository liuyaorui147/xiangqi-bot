"""从天天象棋窗口截屏。优先 PrintWindow / BitBlt，失败则回退全屏裁剪。"""
import ctypes
import sys
from contextlib import contextmanager
from ctypes import wintypes

import cv2
import numpy as np

try:
    ctypes.WinDLL("shcore").SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.WinDLL("user32").SetProcessDPIAware()
    except Exception:
        pass

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
user32.EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]
user32.EnumWindows.restype = wintypes.BOOL
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetClassNameW.restype = ctypes.c_int
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
user32.PrintWindow.restype = wintypes.BOOL
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.ShowWindow.restype = wintypes.BOOL
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.SetForegroundWindow.restype = wintypes.BOOL
user32.IsIconic.argtypes = [wintypes.HWND]
user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, wintypes.UINT]
user32.SetWindowPos.restype = wintypes.BOOL

HWND_TOP = 0
HWND_TOPMOST = -1
HWND_NOTOPMOST = -2
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040

SW_RESTORE = 9
PW_RENDERFULLCONTENT = 0x00000002
SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0


def find_window(keyword):
    """按标题关键字找窗口（含最小化的也匹配），返回 hwnd。优先选面积最大的匹配项。"""
    found = []

    @WNDENUMPROC
    def _cb(hwnd, lparam):
        n = user32.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        title = buf.value
        if keyword in title:
            r = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(r))
            w, h = r.right - r.left, r.bottom - r.top
            found.append((hwnd, title, (r.left, r.top, w, h)))
        return True

    user32.EnumWindows(_cb, 0)
    if not found:
        return None
    found.sort(key=lambda x: x[2][2] * x[2][3], reverse=True)
    return found[0]


def enum_windows(visible_only=True):
    """枚举窗口，返回 [{"hwnd","title","class","rect"}]。

    visible_only=False 时连最小化/被藏到 (-32000,-32000) 的窗口也一并返回——
    模拟器常把窗口藏起来，只看可见窗口会找不到目标。
    """
    out = []

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
        out.append({"hwnd": hwnd, "title": buf.value, "class": cls.value,
                    "rect": (r.left, r.top, r.right - r.left, r.bottom - r.top) if ok else None})
        return True

    user32.EnumWindows(_cb, 0)
    return out


def find_window_any(keywords=(), classes=(), visible_only=False):
    """按**多个**标题关键字或窗口类名找窗口，返回面积最大的匹配项。

    只认死一个标题关键字太脆：游戏改个标题就永远找不到。类名比标题稳定，
    两个条件任一命中即可。默认 visible_only=False，因为模拟器窗口常被最小化。
    """
    kws = [k for k in (keywords or ()) if k]
    cls_want = [c.lower() for c in (classes or ()) if c]
    if not kws and not cls_want:
        return None
    best, best_area = None, -1
    for w in enum_windows(visible_only=visible_only):
        title = w["title"] or ""
        wcls = (w["class"] or "").lower()
        if not (any(k in title for k in kws) or any(c in wcls for c in cls_want)):
            continue
        rect = w["rect"] or (0, 0, 0, 0)
        area = rect[2] * rect[3]
        if area > best_area:
            best_area, best = area, (w["hwnd"], title, rect)
    return best


def restore(hwnd, foreground=True):
    """多手段恢复窗口：解除隐藏 + 解除最小化 + 置前返回可见。"""
    k32 = ctypes.WinDLL("kernel32")
    for cmd in (1, 9, 5):  # SW_SHOWNORMAL, SW_RESTORE, SW_SHOW
        user32.ShowWindow(hwnd, cmd)
        k32.Sleep(250)
        r = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(r))
        if r.left > -1000 and (r.right - r.left) > 100:
            break
    if foreground:
        try:
            user32.SetForegroundWindow(hwnd)
        except Exception:
            pass
        k32.Sleep(400)
    r = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    return r.left > -1000 and (r.right - r.left) > 100


@contextmanager
def _screen_dc(hwnd):
    """GetDC/ReleaseDC 配对。hwnd=0 表示全屏 DC。

    GDI 句柄必须成对释放：每 2 秒截一次图，漏一个句柄跑一晚就耗尽系统
    GDI 资源（Windows 截屏程序的经典死法）。异常路径尤其容易漏。
    """
    pdc = user32.GetDC(hwnd)
    try:
        yield pdc
    finally:
        if pdc:
            user32.ReleaseDC(hwnd, pdc)


@contextmanager
def _compat_bitmap(pdc, w, h):
    """CreateCompatibleDC + CreateCompatibleBitmap + SelectObject，并保证逆序释放。

    注意必须先把旧位图 SelectObject 回去再 DeleteObject，否则位图仍被 DC
    选中，DeleteObject 实际不释放、句柄照样泄漏。
    """
    hdc = bmp = old = None
    try:
        hdc = gdi32.CreateCompatibleDC(pdc)
        if not hdc:
            yield None, None
            return
        bmp = gdi32.CreateCompatibleBitmap(pdc, w, h)
        if not bmp:
            yield hdc, None
            return
        old = gdi32.SelectObject(hdc, bmp)
        yield hdc, bmp
    finally:
        if hdc and old:
            gdi32.SelectObject(hdc, old)     # 先换回旧位图
        if bmp:
            gdi32.DeleteObject(bmp)
        if hdc:
            gdi32.DeleteDC(hdc)


def capture_screen():
    """截取整块虚拟屏幕（多显示器合并区域）。返回 BGR 图与原点偏移。"""
    SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN = 76, 77
    SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 78, 79
    x = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    y = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    w = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
    h = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
    with _screen_dc(0) as pdc:
        if not pdc:
            return None, (x, y)
        with _compat_bitmap(pdc, w, h) as (hdc, bmp):
            if not bmp:
                return None, (x, y)
            gdi32.BitBlt(hdc, 0, 0, w, h, pdc, x, y, SRCCOPY)
            img = _hdc_to_bgr(hdc, bmp, w, h)
    return img, (x, y)


def _hdc_to_bgr(hdc, bmp, w, h):
    """把 HBITMAP 内容转成 numpy BGR 数组。"""
    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", wintypes.DWORD), ("biWidth", ctypes.c_int),
            ("biHeight", ctypes.c_int), ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", ctypes.c_long),
            ("biYPelsPerMeter", ctypes.c_long), ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD),
        ]

    bi = BITMAPINFOHEADER()
    bi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bi.biWidth = w
    bi.biHeight = -h
    bi.biPlanes = 1
    bi.biBitCount = 32
    bi.biCompression = 0
    bi.biSizeImage = 0

    buf = (ctypes.c_char * (w * h * 4))()
    gdi32.GetDIBits(hdc, bmp, 0, h, buf, ctypes.byref(bi), DIB_RGB_COLORS)
    img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def _grab(hwnd, w, h, use_screen_dc):
    """抓取窗口客户区（use_screen_dc=False）或屏幕上该区域（True）。失败返回 None。"""
    parent = None if use_screen_dc else hwnd
    with _screen_dc(parent) as pdc:
        if not pdc:
            return None
        with _compat_bitmap(pdc, w, h) as (hdc, bmp):
            if not bmp:
                return None
            if use_screen_dc:
                # 用客户区原点而不是窗口原点，避免把标题栏/边框裁进来
                pt = wintypes.POINT(0, 0)
                user32.ClientToScreen(hwnd, ctypes.byref(pt))
                gdi32.BitBlt(hdc, 0, 0, w, h, pdc, pt.x, pt.y, SRCCOPY)
            return _hdc_to_bgr(hdc, bmp, w, h)


def raise_no_activate(hwnd, topmost=True):
    """把窗口抬到顶层且不抢键盘焦点。

    HWND_TOP 对前台窗口无效（前台保护带），必须用 TOPMOST 悬浮层才能
    压过其他进程的前台窗口。topmost=True 时保持置顶（适合 bot 运行期）。
    """
    after = HWND_TOPMOST if topmost else HWND_TOP
    ok = user32.SetWindowPos(hwnd, after, 0, 0, 0, 0,
                             SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW)
    if not ok and topmost:
        user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0,
                            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)
        ok = user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                                 SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)
    return bool(ok)


def capture_window(hwnd, raise_if_occluded=True):
    """截取窗口客户区。依次尝试 PrintWindow -> 窗口DC BitBlt -> 全屏裁剪。

    ScreenCrop 抓的是屏幕上该区域的最顶层内容，因此截之前先把目标窗口抬起，
    避免抓到压在上面的其他窗口（不抢焦点）。
    """
    cr = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(cr))
    w, h = cr.right - cr.left, cr.bottom - cr.top
    if w < 10 or h < 10:
        return None

    if raise_if_occluded:
        raise_no_activate(hwnd)
        ctypes.WinDLL("kernel32").Sleep(120)

    # 方案 1：PrintWindow 带 RENDERFULLCONTENT，能抓 OpenGL/DirectX 窗口
    # 资源一律交给 with 释放——以前是手工逐行释放，_hdc_to_bgr 一旦抛异常
    # 就会整组泄漏（天天象棋走的就是"PrintWindow 返回全黑"的降级路径）。
    ok, img = False, None
    try:
        with _screen_dc(hwnd) as pdc:
            if not pdc:
                raise OSError("GetDC 失败")
            with _compat_bitmap(pdc, w, h) as (hdc, bmp):
                if not bmp:
                    raise OSError("CreateCompatibleBitmap 失败")
                ok = bool(user32.PrintWindow(hwnd, hdc, PW_RENDERFULLCONTENT))
                img = _hdc_to_bgr(hdc, bmp, w, h)
        if ok and img is not None and float(img.mean()) > 8.0:
            return img, "PrintWindow"
    except Exception:
        pass

    # 方案 2：窗口自身 DC
    try:
        img = _grab(hwnd, w, h, False)
        if img is not None and float(img.mean()) > 8.0:
            return img, "WindowDC"
    except Exception:
        pass

    # 方案 3：全屏带偏移裁剪（兜底）
    img = _grab(hwnd, w, h, True)
    return img, "ScreenCrop"


if __name__ == "__main__":
    kw = sys.argv[1] if len(sys.argv) > 1 else "天天象棋"
    out = sys.argv[2] if len(sys.argv) > 2 else "shots/grab.png"
    if len(sys.argv) > 1 and sys.argv[1] == "screen":
        img, (ox, oy) = capture_screen()
        cv2.imwrite(out, img)
        print(f"全屏 {img.shape[1]}x{img.shape[0]} 原点=({ox},{oy}) 亮度={img.mean():.1f} -> {out}")
        sys.exit(0)

    hit = find_window(kw)
    if not hit:
        print(f"未找到标题含 {kw!r} 的窗口")
        sys.exit(1)
    hwnd, title, rect = hit
    print(f"找到: hwnd={hwnd} rect={rect} title={title!r}")
    for parent_kw in ("腾讯应用宝", "应用宝"):
        ph = find_window(parent_kw)
        if ph and ph[0] != hwnd:
            restore(ph[0], foreground=False)
            print(f"恢复宿主窗口: {ph[1]!r} -> {'ok' if True else ''}")
            break
    ok = restore(hwnd)
    print(f"游戏窗口恢复: {ok}")
    res = capture_window(hwnd)
    if not res or res[0] is None:      # 三条路径全失败时 img 为 None
        print("截屏失败")
        sys.exit(2)
    img, how = res
    cv2.imwrite(out, img)
    print(f"方式={how}  尺寸={img.shape[1]}x{img.shape[0]}  均值亮度={img.mean():.1f}  已保存 {out}")

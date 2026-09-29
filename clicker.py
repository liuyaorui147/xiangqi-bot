"""向游戏窗口发送鼠标点击（不截屏、不抢焦点之外的操作）。"""
import ctypes
from ctypes import wintypes

import capture as cap

user32 = cap.user32

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000

SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN = 76, 77
SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 78, 79


# MOUSEINPUT.dwExtraInfo 的类型是 ULONG_PTR。ctypes.wintypes 里没定义这个类型，
# 故按指针宽度自适应：64 位 = c_ulonglong，32 位 = c_ulong。
# 原来写成 POINTER(c_ulong)——64 位下两者都是 8 字节所以能跑（歪打正着），
# 但 32 位 Python 上结构体大小会从 28 变成 32，SendInput 解析出的字段全是错的。
ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("mi", _MOUSEINPUT)]


def client_to_screen(hwnd, x, y):
    pt = cap.wintypes.POINT(int(x), int(y))
    user32.ClientToScreen(hwnd, cap.ctypes.byref(pt))
    return pt.x, pt.y


def click_screen(sx, sy, steps=3):
    """在屏幕绝对坐标处单击（含移动，便于游戏识别为连续轨迹）。"""
    vx, vy = user32.GetSystemMetrics(SM_XVIRTUALSCREEN), user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    vw, vh = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN), user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)

    def norm(x, y):
        return int((x - vx) * 65536 / max(vw - 1, 1)), int((y - vy) * 65536 / max(vh - 1, 1))

    cur = cap.wintypes.POINT()
    user32.GetCursorPos(cap.ctypes.byref(cur))
    x0, y0 = cur.x, cur.y
    for k in range(1, steps + 1):
        mx, my = norm(x0 + (sx - x0) * k / steps, y0 + (sy - y0) * k / steps)
        inp = _INPUT(INPUT_MOUSE, _MOUSEINPUT(mx, my, 0,
                                              MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE
                                              | MOUSEEVENTF_VIRTUALDESK, 0, 0))
        user32.SendInput(1, cap.ctypes.byref(inp), ctypes.sizeof(_INPUT))

    mx, my = norm(sx, sy)
    for flag in (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP):
        inp = _INPUT(INPUT_MOUSE, _MOUSEINPUT(mx, my, 0,
                                              flag | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
                                              0, 0))
        user32.SendInput(1, cap.ctypes.byref(inp), ctypes.sizeof(_INPUT))


def click_cell(hwnd, loc, col, row):
    """点击棋盘格点 (col,row)。loc 来自 board_locator.locate_board。"""
    x, y = loc["points"][(col, row)]
    sx, sy = client_to_screen(hwnd, x, y)
    click_screen(sx, sy)
    return sx, sy


def uci_to_cells(move):
    """'b2e2' -> ((1,7),(4,7))。棋盘 rank 0 是红方底线 => row = 9 - rank。"""
    def part(s):
        col = ord(s[0]) - ord("a")
        rank = int(s[1])
        return col, 9 - rank
    return part(move[:2]), part(move[2:4])

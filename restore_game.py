"""恢复模拟器宿主窗口与象棋游戏窗口，并报告实际坐标。

以前这个脚本在**顶层直接执行副作用**——import 它就会去动别人的窗口，
既不能复用也没法测试。现在全部包成函数，只有直接运行才执行。

窗口匹配也不再只认死一个标题关键字（游戏改个标题就永远找不到）：
现在支持多个候选关键字 + 窗口类名，且默认连最小化/被藏起来的窗口一起找。
"""
import argparse
import sys
import time

import capture as cap

# 标题候选关键字，任一命中即可
HOST_TITLES = ("腾讯应用宝", "应用宝", "MuMu", "雷电", "LDPlayer", "Nox", "夜神", "BlueStacks")
GAME_TITLES = ("天天象棋", "JJ象棋", "象棋")

# 窗口类名比标题稳定：应用宝宿主固定是 Qt5152QWindowIcon
HOST_CLASSES = ("Qt5152QWindowIcon",)
GAME_CLASSES = ()


def restore_host(titles=HOST_TITLES, classes=HOST_CLASSES, settle=1.0):
    """恢复模拟器宿主窗口。返回 (hwnd, title, rect)，未找到返回 None。"""
    hit = cap.find_window_any(titles, classes)
    if not hit:
        return None
    hwnd, title, rect = hit
    cap.user32.ShowWindow(hwnd, 1)      # SW_SHOWNORMAL
    time.sleep(settle)
    cap.user32.ShowWindow(hwnd, 9)      # SW_RESTORE
    time.sleep(settle)
    return hwnd, title, rect


def restore_game(titles=GAME_TITLES, classes=GAME_CLASSES, settle=1.5):
    """恢复象棋窗口。返回 (hwnd, title, rect, ok)，未找到返回 None。"""
    hit = cap.find_window_any(titles, classes)
    if not hit:
        return None
    hwnd, title, rect = hit
    ok = cap.restore(hwnd)
    time.sleep(settle)
    r = cap.wintypes.RECT()
    cap.user32.GetWindowRect(hwnd, cap.ctypes.byref(r))
    return hwnd, title, (r.left, r.top, r.right - r.left, r.bottom - r.top), ok


def list_windows(skip_empty=True):
    """列出窗口，便于确认新模拟器的标题/类名该填什么。"""
    for w in cap.enum_windows():
        if skip_empty and not (w["title"] or "").strip():
            continue
        yield w


def main(argv=None):
    ap = argparse.ArgumentParser(description="恢复模拟器/象棋窗口并报告坐标")
    ap.add_argument("--host", nargs="*", default=list(HOST_TITLES),
                    help="宿主窗口标题候选关键字")
    ap.add_argument("--game", nargs="*", default=list(GAME_TITLES),
                    help="游戏窗口标题候选关键字")
    ap.add_argument("--list", action="store_true", help="只列出窗口，不做恢复")
    args = ap.parse_args(argv)

    if args.list:
        print("=== 当前所有窗口 ===")
        for w in list_windows():
            print(f"  [{w['class'][:22]:<22}] {w['rect']}  {w['title'][:40]}")
        return 0

    host = restore_host(args.host)
    if host:
        print(f"宿主: hwnd={host[0]} title={host[1]!r} rect={host[2]}")
    else:
        print(f"未找到宿主窗口（候选 {args.host}）")
    time.sleep(1.5)

    game = restore_game(args.game)
    if game:
        hwnd, title, rect, ok = game
        print(f"游戏: hwnd={hwnd} title={title!r} rect={rect} ok={ok}")
        return 0

    print(f"未找到游戏窗口（候选 {args.game}）。当前窗口：")
    for w in list_windows():
        print(f"  [{w['class'][:22]:<22}] {w['rect']}  {w['title'][:40]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

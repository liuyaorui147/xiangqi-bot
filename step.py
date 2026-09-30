"""单步驱动：截图 -> 识别 -> 引擎 -> 落子。一步一调，结果直接打到 stdout。

给人工（或上层代理）驱动对局用，不跑自动状态机。相比 main.py auto：
每一步的局面、评估、胜率都当场可见，识别错了能立刻发现，不用去翻日志。

用法：
  python step.py look                 只看当前局面 + 引擎评估 + 胜率，不落子
  python step.py move                 看局面并落子（默认执红）
  python step.py move --black         按执黑走（棋盘翻转时用）
  python step.py wait [秒]            轮询等对手落子，局面一变就返回
  python step.py tap X Y              直接点屏幕坐标（用来点「再来一局」等）
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

import advice
import board_locator as bl
from engine import Engine
from recognize import Recognizer, board_diagram, to_fen, validate_board

DEV = "emulator-5554"
CALIB = None
if os.path.exists("board_calib.json"):
    CALIB = json.load(open("board_calib.json"))

_REC = None
_ENG = None


def rec():
    global _REC
    if _REC is None:
        _REC = Recognizer()
    return _REC


def eng():
    global _ENG
    if _ENG is None:
        _ENG = Engine(exe=os.path.join("engine", "pikafish.exe"),
                      nnue_path=None, threads=None, hash_mb=256)
    return _ENG


def grab():
    r = subprocess.run(["adb", "-s", DEV, "exec-out", "screencap", "-p"],
                       capture_output=True, timeout=30)
    if not r.stdout:
        return None
    return cv2.imdecode(np.frombuffer(r.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)


def tap(x, y):
    subprocess.run(["adb", "-s", DEV, "shell", "input", "tap",
                    str(int(x)), str(int(y))], capture_output=True, timeout=20)


def read(img, my_side="red"):
    """识别棋盘。返回 (board, loc) 或 (None, None)。

    朝向处理：board 已按"红在下"归一化（r=0 黑方、r=9 红方）。屏幕格点
    points[(c,r)] 的 (0,0) 是棋盘左上角，与 r=0（黑方在上）一致，所以
    执红时可直接用；执黑（App 翻转棋盘）时要上下左右都翻。
    """
    if img is None:
        return None, None
    loc = (bl.locate_with_calib(img, CALIB) or bl.locate_affine(img)
           or bl.locate_board(img))
    if loc is None:
        return None, None
    board = rec().recognize(img, loc)
    cells = {k: (v[0] if isinstance(v, tuple) else v)
             for k, v in board.items()
             if (v[0] if isinstance(v, tuple) else v)}
    # 红帅应在 r>=7；不在就说明屏幕上是"红在上"，需要翻转
    flip = False
    kr = [r for (c, r), p in cells.items() if p == "K"]
    if kr and min(kr) < 5:
        flip = True
    return {"cells": cells, "flip": flip}, loc


def cell_to_screen(loc, cell, flip):
    (c, r) = cell
    if flip:
        c, r = 8 - c, 9 - r
    return loc["points"][(c, r)]


def uci_to_cells(u):
    """'h0g2' -> ((c,r),(c,r))，按 board 坐标（r=0 黑方底线）。"""
    f, t = u[:2], u[2:4]
    def one(s):
        return (ord(s[0]) - ord('a'), int(s[1]))
    return one(f), one(t)


def show(board_d, loc, img=None):
    cells = board_d["cells"]
    n = len(cells)
    print(f"棋子 {n} 个   朝向{'已翻转' if board_d['flip'] else '正常'}")
    print(board_diagram({k: (v, 1.0) for k, v in cells.items()}))
    ok, why = validate_board({k: (v, 1.0) for k, v in cells.items()})
    print("校验:", "通过" if ok else f"失败 {why}")
    return ok


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    cmd = sys.argv[1]
    if cmd == "tap":
        tap(float(sys.argv[2]), float(sys.argv[3]))
        print(f"已点击 ({sys.argv[2]},{sys.argv[3]})")
        return 0

    my_side = "black" if "--black" in sys.argv else "red"
    img = grab()
    if img is None:
        print("截图失败")
        return 2
    board_d, loc = read(img, my_side)
    if board_d is None:
        print("未定位到棋盘")
        return 3
    ok = show(board_d, loc, img)
    if cmd == "look" or not ok:
        if not ok:
            print("局面不合法，不落子")
            return 4

    fen_body = to_fen({k: (v, 1.0) for k, v in board_d["cells"].items()})
    side = "w" if my_side == "red" else "b"
    fen = f"{fen_body} {side} - - 0 1"
    print("FEN:", fen)

    if cmd == "wait":
        limit = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
        t0 = time.time()
        base = fen_body
        print(f"等对手落子（最多 {limit:.0f}s）…", flush=True)
        while time.time() - t0 < limit:
            time.sleep(1.0)
            img2 = grab()
            if img2 is None:
                continue
            b2, _ = read(img2, my_side)
            if b2 is None or not b2["cells"]:
                continue
            f2 = to_fen({k: (v, 1.0) for k, v in b2["cells"].items()})
            if f2 != base:
                print(f"\n对手已落子（{time.time() - t0:.0f}s）")
                show(b2, None)
                return 0
        print("等待超时，局面未变")
        return 5

    if cmd != "move":
        print(__doc__)
        return 1

    # 引擎：先按选着时间搜（快），再按评估时间搜（准），两条都报出来
    E = eng()
    bm, res = E.analyse(fen, movetime=350)
    info = res["info"]
    wp_shallow = advice.win_percent(info, flip=False, board=board_d["cells"],
                                    my_side=my_side)
    print(f"\n选着 {bm}  {info.get('depth','?')}层  评估 {advice.eval_text(info)}"
          f"  胜率(350ms) {'—' if wp_shallow is None else f'{wp_shallow*100:.1f}%'}")

    bm2, res2 = E.analyse(fen, movetime=1500)
    i2 = res2["info"]
    ok2, why2 = advice.eval_credible(board_d["cells"], i2, my_side)
    wp = advice.win_percent(i2, flip=False, board=board_d["cells"],
                            my_side=my_side)
    print(f"深搜 {bm2}  {i2.get('depth','?')}层  评估 {advice.eval_text(i2)}"
          f"  胜率(1.5s) {'不可信：' + why2 if not ok2 else ('—' if wp is None else f'{wp*100:.1f}%')}")

    use = bm2 or bm
    if not use:
        print("引擎没给出着法")
        return 6
    src, dst = uci_to_cells(use)
    p = board_d["cells"].get(src)
    if not p:
        print(f"起点 {src} 没有棋子，放弃")
        return 7
    x1, y1 = cell_to_screen(loc, src, board_d["flip"])
    x2, y2 = cell_to_screen(loc, dst, board_d["flip"])
    print(f"落子 {p} {use}  ->  ({int(x1)},{int(y1)}) ({int(x2)},{int(y2)})")
    tap(x1, y1)
    time.sleep(0.12)
    tap(x2, y2)
    time.sleep(0.6)
    img3 = grab()
    b3, _ = read(img3, my_side) if img3 is not None else (None, None)
    if b3 is not None:
        f3 = to_fen({k: (v, 1.0) for k, v in b3["cells"].items()})
        print("落子后:", "已生效" if f3 != fen_body else "!! 局面没变，可能未生效")
        show(b3, None)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""对局画面监视器：持续截图 + 自动判状态，结果写 logs/monitor.log。

用途：bot 跑起来后用它观察"现在到底在哪个页面"。人可以直接翻 shots/monitor/
看截图，不方便看图时读 monitor.log 里的文本结论即可。

每轮输出一行，包含：
  - 页面判定：结算页 / 对局中 / 加载中 / 其他
  - 画面亮度（弹窗会让画面变暗）
  - 识别到的棋子数
用法：python monitor.py [间隔秒数]
"""
import os
import sys
import time

import cv2

import adb as adbmod
import main as bot
import mumu_cap as mumucap


def grab():
    img = None
    try:
        m = mumucap.MuMuCap()
        if m.open():
            img = m.screencap()
    except Exception:
        img = None
    if img is None:
        try:
            import subprocess
            r = subprocess.run(["adb", "-s", "emulator-5554", "exec-out",
                                "screencap", "-p"], capture_output=True)
            if r.stdout:
                import numpy as np
                img = cv2.imdecode(np.frombuffer(r.stdout, dtype=np.uint8),
                                   cv2.IMREAD_COLOR)
        except Exception:
            img = None
    return img


_REC = None


def _rec():
    """监视器自己建识别器：bot 的 REC 要等 main() 起来才有值，
    直接拿来用会得到 None。"""
    global _REC
    if _REC is None:
        from recognize import Recognizer
        _REC = Recognizer()
    return _REC


def judge(img):
    """返回 (页面判定, 备注)。"""
    if img is None:
        return "截图失败", ""
    bright = float(img.mean())
    note = f"亮度{bright:.0f}"
    try:
        btn = bot.green_button(img)
    except Exception:
        btn = None
    if btn is not None:
        return "结算页", f"{note} 绿色按钮面积{int(btn[0])} @{int(btn[1])},{int(btn[2])}"
    try:
        import board_locator as bl
        loc = (bl.locate_with_calib(img, bot.CALIB) or bl.locate_affine(img)
               or bl.locate_board(img))
        if loc is None:
            return "非对局页", note
        board, _flip = bot.orient_board(_rec().recognize(img, loc))
        n = sum(1 for v in board.values() if (v[0] if isinstance(v, tuple) else v))
        if n <= 2:
            return "加载/匹配中", f"{note} 棋子{n}"
        return "对局中", f"{note} 棋子{n}"
    except Exception as e:
        return "识别异常", f"{note} {type(e).__name__}: {e}"


def main():
    gap = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    d = os.path.join("shots", "monitor")
    os.makedirs(d, exist_ok=True)
    logp = os.path.join("logs", "monitor.log")
    print(f"监视器启动，每 {gap:g}s 一张 -> {d}/latest.png，结论写 {logp}")
    last = ""
    while True:
        try:
            img = grab()
            if img is not None:
                cv2.imwrite(os.path.join(d, "latest.png"), img)
                ts = time.strftime("%H%M%S")
                cv2.imwrite(os.path.join(d, f"m_{ts}.png"), img)
            page, note = judge(img)
            line = f"[{time.strftime('%H:%M:%S')}] {page:<10} {note}"
            # 状态变化或每 30 秒才写一行，避免刷屏
            if page != last or int(time.time()) % 30 < gap:
                print(line, flush=True)
                with open(logp, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            last = page
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[监视器] {type(e).__name__}: {e}", flush=True)
        time.sleep(gap)


if __name__ == "__main__":
    main()

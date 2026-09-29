"""把游戏窗口挪到不同位置，验证定位是否仍然正确。"""
import json
import time

import cv2
import numpy as np

import capture as cap
import board_locator as bl

user32 = cap.user32


def get_rect(hwnd):
    r = cap.wintypes.RECT()
    user32.GetWindowRect(hwnd, cap.ctypes.byref(r))
    return (r.left, r.top, r.right - r.left, r.bottom - r.top)


def measure(hwnd, tag):
    img, how = cap.capture_window(hwnd)
    if img is None:
        print(f"[{tag}] 截屏失败")
        return None
    cv2.imwrite(f"shots/move_{tag}.png", img)
    loc = bl.locate_board(img)
    if loc is None:
        print(f"[{tag}] 定位失败")
        return None
    # 偏差统计
    pts = np.array([loc["points"][(i, j)] for i in range(9) for j in range(10)])
    cs = loc["circles"][:, :2]
    dxs, dys, used = [], [], 0
    for x, y in cs:
        d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
        k = d.argmin()
        if d[k] < 0.35 * loc["cell_x"]:
            dxs.append(x - pts[k][0])
            dys.append(y - pts[k][1])
            used += 1
    print(f"[{tag}] 窗口{get_rect(hwnd)}  图{img.shape[1]}x{img.shape[0]}  方式={how}")
    print(f"       cell=({loc['cell_x']:.2f},{loc['cell_y']:.2f})  相位=({loc['ox']:.1f},{loc['oy']:.1f})  "
          f"圆={len(cs)}  命中={used}")
    if dxs:
        print(f"       偏移 dx={np.mean(dxs):+.2f} dy={np.mean(dys):+.2f}  std={np.std(dxs):.2f}/{np.std(dys):.2f}")
    return loc


hit = cap.find_window("天天象棋")
hwnd = hit[0]
orig = get_rect(hwnd)
print(f"原始窗口位置 {orig}")

l0 = measure(hwnd, "A_orig")

SWP_NOSIZE = 0x0001
for (nx, ny) in [(120, 60), (900, 120)]:
    user32.SetWindowPos(hwnd, 0, nx, ny, 0, 0, SWP_NOSIZE)
    time.sleep(1.2)
    measure(hwnd, f"B_{nx}_{ny}")

# 恢复原位
user32.SetWindowPos(hwnd, 0, orig[0], orig[1], 0, 0, SWP_NOSIZE)
time.sleep(1.0)
print(f"\n已恢复原位 -> {get_rect(hwnd)}")

"""从已知局面（初始排布）收割棋子模板 + 统计红/黑/空格颜色基准。

用法: python harvest.py <截图> [--apply]
不加 --apply 只预览拼图；加 --apply 才写入 templates/ 并更新 color_ref.json。
"""
import json
import os
import sys

import cv2
import numpy as np

from board_locator import locate_affine, locate_board

# 初始局面：row0 在棋盘上方（黑方），row9 在下方（红方）
INITIAL = {
    0: ["r", "n", "b", "a", "k", "a", "b", "n", "r"],
    2: [None, "c", None, None, None, None, None, "c", None],
    3: ["p", None, "p", None, "p", None, "p", None, "p"],
    6: ["P", None, "P", None, "P", None, "P", None, "P"],
    7: [None, "C", None, None, None, None, None, "C", None],
    9: ["R", "N", "B", "A", "K", "A", "B", "N", "R"],
}
ROLE_CN = {"r": "车", "n": "马", "b": "象", "a": "士", "k": "将", "c": "炮", "p": "卒",
           "R": "车", "N": "马", "B": "相", "A": "仕", "K": "帅", "C": "炮", "P": "兵"}


def cell_crop(img, x, y, cell, scale=0.52):
    """裁出以 (x,y) 为中心的格子区域。"""
    r = int(cell * scale)
    h, w = img.shape[:2]
    x0, y0 = int(x - r), int(y - r)
    x1, y1 = int(x + r), int(y + r)
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    return img[y0:y1, x0:x1].copy()


def color_features(crop):
    """红/黑判别特征：红色像素占比、深色像素占比（HSV）。"""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = (((h < 8) | (h > 170)) & (s > 90) & (v > 60)).mean()
    dark = ((s < 120) & (v < 90)).mean()
    return float(red), float(dark)


def auto_thresholds(img, loc):
    """自动定阈值与朝向，让识别不依赖具体皮肤。

    返回 (ink_thr, red_top, 每格 ink, 每格 (rr,dr))。
    ink = max(红占比, 深占比)，用大津法在有子/无子之间自动切分。
    """
    cell = (loc["cell_x"] + loc["cell_y"]) / 2
    inks, stats = [], {}
    for j in range(10):
        for i in range(9):
            x, y = loc["points"][(i, j)]
            crop = cell_crop(img, x, y, cell)
            if crop is None:
                stats[(i, j)] = (0.0, 0.0)
                inks.append(0.0)
                continue
            rr, dr = color_features(crop)
            stats[(i, j)] = (rr, dr)
            inks.append(max(rr, dr))
    # 最大间隙聚类：在排序后的 ink 序列里找最大跳变处切开。
    # 比大津法稳——大津法会被"App 在棋盘上画的小红点标记"(ink≈0.02，而真棋子≈0.10)
    # 这类小聚类带偏，把阈值卡在误判那一档上。
    vals = sorted(float(v) for v in inks)
    gap, cut = 0.0, len(vals) - 1
    for k in range(len(vals) - 1):
        d = vals[k + 1] - vals[k]
        if d > gap:
            gap, cut = d, k
    if len(vals) < 2 or gap < 0.01:
        ink_thr = 0.02                      # 退化情况（全是空格）用保底值
    else:
        ink_thr = (vals[cut] + vals[cut + 1]) / 2.0

    top_red = sum(1 for j in range(5) for i in range(9)
                  if max(stats[(i, j)]) >= ink_thr and stats[(i, j)][0] > stats[(i, j)][1])
    bot_red = sum(1 for j in range(5, 10) for i in range(9)
                  if max(stats[(i, j)]) >= ink_thr and stats[(i, j)][0] > stats[(i, j)][1])
    return ink_thr, top_red > bot_red, inks, stats


def role_at(rank, col):
    """按 rank(0=红方底线) 与列给出棋子角色字母（大小写由阵营另定）。"""
    if rank in (0, 9):
        return ["R", "N", "B", "A", "K", "A", "B", "N", "R"][col]
    if rank in (2, 7):
        return "C" if col in (1, 7) else None
    if rank in (3, 6):
        return "P" if col % 2 == 0 else None
    return None


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "shots/step1_start.png"
    apply = "--apply" in sys.argv
    img = cv2.imread(src)
    if img is None or getattr(img, "size", 0) == 0:
        # 不检查的话后面 cvtColor 才崩，报的是 "!_src.empty()"，离真正原因
        # （截图没写成功 / 路径不对）远得多。
        print(f"读不到截图: {src}（文件不存在或为空）")
        sys.exit(2)
    loc = locate_affine(img) or locate_board(img)
    if loc is None:
        print("定位失败，无法收割")
        sys.exit(2)
    cell = (loc["cell_x"] + loc["cell_y"]) / 2
    ink_thr, red_top, inks, stats = auto_thresholds(img, loc)
    print(f"cell={cell:.2f}  共 90 格")
    print(f"自动阈值 ink>={ink_thr:.3f}   棋盘朝向: 红方在{'上' if red_top else '下'}")

    os.makedirs("templates", exist_ok=True)
    tiles, feats = [], {"red": [], "black": [], "empty": []}
    n_saved = 0
    for j in range(10):
        for i in range(9):
            x, y = loc["points"][(i, j)]
            crop = cell_crop(img, x, y, cell)
            if crop is None:
                continue
            rr, dr = stats[(i, j)]
            has = max(rr, dr) >= ink_thr
            # rank: 0 = 红方底线
            rank = j if red_top else 9 - j
            role = role_at(rank, i) if has else None
            side = "red" if (rr > dr) else "black"
            if not has or role is None:
                # 有墨迹但不在初始局面该有的位置：多半是界面标记，当空格处理
                if has:
                    feats.setdefault("noise", []).append((rr, dr))
                feats["empty"].append((rr, dr))
                tiles.append((f"e{i}_{j}", crop))
                continue
            piece = role.upper() if side == "red" else role.lower()
            feats[side].append((rr, dr))
            tag = f"{piece}{i}_{j}"
            tiles.append((tag, crop))
            if piece and apply:
                cv2.imwrite(f"templates/t_{piece}_{i}_{j}.png", crop)
                n_saved += 1

    # 拼预览大图
    cols = 9
    size = tiles[0][1].shape[0]
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * size, cols * size, 3), 255, np.uint8)
    for k, (tag, c) in enumerate(tiles):
        r, cc = divmod(k, cols)
        sheet[r * size:(r + 1) * size, cc * size:(cc + 1) * size] = c
    os.makedirs("shots", exist_ok=True)
    cv2.imwrite("shots/harvest_sheet.png", sheet)
    print(f"拼图 -> shots/harvest_sheet.png  共 {len(tiles)} 格")

    ref = {"ink_thr": ink_thr, "red_top": bool(red_top)}
    for k, v in feats.items():
        a = np.asarray(v, float)
        if len(a) == 0:
            continue
        ref[k] = {"n": len(v), "red_mean": float(a[:, 0].mean()), "dark_mean": float(a[:, 1].mean()),
                  "red_std": float(a[:, 0].std()), "dark_std": float(a[:, 1].std())}
        print(f"  {k:<5} n={len(v):<2} red={ref[k]['red_mean']:.3f}±{ref[k]['red_std']:.3f} "
              f"dark={ref[k]['dark_mean']:.3f}±{ref[k]['dark_std']:.3f}")
    if apply:
        json.dump(ref, open("color_ref.json", "w"), ensure_ascii=False, indent=1)
        calib = {"cell_x": loc["cell_x"], "cell_y": loc["cell_y"],
                 "ox": loc["ox"], "oy": loc["oy"],
                 "ink_thr": ink_thr, "red_top": bool(red_top)}
        if loc.get("affine"):
            calib.update({"affine": True, "U": loc["U"], "V": loc["V"], "O": loc["O"]})
            print("  标定类型: 仿射（可吸收透视/旋转）")
        json.dump(calib, open("board_calib.json", "w"), indent=1)
        print(f"已写入 templates/ {n_saved} 张 + color_ref.json + board_calib.json")


if __name__ == "__main__":
    main()

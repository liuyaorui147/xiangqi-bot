"""棋盘网格自动校正：从棋子圆心反推 9x10 交叉点网格。

核心原则（实测教训）：
1. 格距与相位必须联合搜索，只搜相位会在初值偏差 >12% 时整体错位。
2. 打分必须用覆盖率 n/(span+1)，否则 30px 亚倍数网格会打掉 60px 真值。
3. 拟合完必须校验宽高比：8 格宽 / 9 格高 => 0.889，偏差大即判定失败。
"""
import sys

import cv2
import numpy as np


def fit_grid_1d(vals, guess, expected_span, cell_range=(0.55, 1.45), tol=0.25):
    """在一维坐标里找 (格距, 相位) 使尽量多的特征点落在棋盘格位上。

    expected_span 是该轴的格距数（x 方向 8，y 方向 9），作为硬约束：
    - 候选网格的跨度超过它直接判无效（掐掉 2 倍频/1.5 倍频假解）
    - 覆盖率按 n/(expected_span+1) 计，空格位不扣分
    返回 (cell, origin_at_k0, 内点数, 覆盖率) 或 None。
    """
    vals = np.asarray(vals, float)
    if vals.size < 4:
        return None
    best = None
    lo, hi = cell_range
    for cell in np.arange(guess * lo, guess * hi, 0.5):
        for o in vals:
            k = np.round((vals - o) / cell)
            m = np.abs(vals - (o + k * cell)) < tol * cell
            n = int(m.sum())
            if n < 3:
                continue
            km = k[m]
            span = int(km.max() - km.min())
            if span > expected_span:
                continue  # 网格过粗，必有检测点被挤出棋盘范围
            cov = n / (expected_span + 1)
            if best is None or cov > best[0] + 1e-9 or (
                abs(cov - best[0]) < 1e-9 and n > best[1]
            ):
                best = (cov, n, float(cell), k.copy(), m.copy())
    if best is None:
        return None
    _, _, _, k, m = best
    ks, vs = k[m], vals[m]
    a = np.vstack([ks, np.ones(ks.size)]).T.astype(float)
    (cell2, intercept), *_ = np.linalg.lstsq(a, vs, rcond=None)
    return float(cell2), float(intercept), int(m.sum()), float(best[0])


def detect_piece_circles(img):
    """两阶段 Hough：先粗测棋子半径，再用紧半径区间精测。"""
    if img is None or getattr(img, "size", 0) == 0:
        return None          # 空图直接放弃，否则 cvtColor 抛 !_src.empty()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    h, w = gray.shape

    coarse = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=18,
        param1=110, param2=30, minRadius=10, maxRadius=int(min(w, h) * 0.08),
    )
    if coarse is None:
        return None, None
    r_med = float(np.median(coarse[0][:, 2]))
    cell_guess = 2.5 * r_med

    fine = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(10, cell_guess * 0.55),
        param1=110, param2=34,
        minRadius=int(cell_guess * 0.28), maxRadius=int(cell_guess * 0.65),
    )
    circles = None if fine is None else fine[0]
    return circles, cell_guess


def locate_board(img, verbose=False):
    """返回 dict(cell_x, cell_y, ox, oy, points[(i,j)->(x,y)], n_x, n_y) 或 None。

    ox/oy 是第 0 列 / 第 0 行的坐标（k=0 相位），不保证对应棋盘第 0 格。
    """
    circles, cell_guess = detect_piece_circles(img)
    if circles is None or len(circles) < 8:
        return None
    if cell_guess is None:
        cell_guess = img.shape[1] / 9.3

    xs = circles[:, 0]
    ys = circles[:, 1]
    # x 方向 9 列全有子，先拟合它，再用 cell_x 反向约束 y（棋盘格是正方形）
    fx = fit_grid_1d(xs, cell_guess, expected_span=8)
    if fx is None:
        return None
    cell_x, ix0, nx, covx = fx
    fy = fit_grid_1d(ys, cell_x, expected_span=9, cell_range=(0.85, 1.18))
    if fy is None:
        return None
    cell_y, iy0, ny, covy = fy

    # 宽高比自检：8 格宽 / 9 格高 => 0.889
    ratio = (cell_x * 8) / (cell_y * 9)
    if verbose:
        print(f"  格距 x={cell_x:.2f} (内点{nx}, 覆盖{covx:.2f})  "
              f"y={cell_y:.2f} (内点{ny}, 覆盖{covy:.2f})  宽高比={ratio:.3f}")
    if not (0.83 < ratio < 0.95):
        return None
    if abs(cell_x - cell_y) / max(cell_x, cell_y) > 0.12:
        return None

    loc = {
        "cell_x": cell_x, "cell_y": cell_y, "ox": ix0, "oy": iy0,
        "ratio": ratio, "circles": circles,
        "inliers_x": nx, "inliers_y": ny,
    }
    anchor_board(loc)
    return loc


def draw_overlay(img, loc, out_path):
    vis = img.copy()
    for (i, j), (x, y) in loc["points"].items():
        p = (int(round(x)), int(round(y)))
        cv2.circle(vis, p, 3, (0, 200, 0), -1)
        cv2.circle(vis, p, int(loc["cell_x"] * 0.42), (255, 120, 0), 1)
    for c in loc["circles"]:
        cv2.circle(vis, (int(c[0]), int(c[1])), int(c[2]), (0, 0, 255), 1)
    cv2.imwrite(out_path, vis)
    return out_path


def locate_with_calib(img, calib, verbose=False, detect_circles=None):
    """用冻结的标定直接生成网格。

    模拟器画面里棋盘像素位置恒定，比每帧重新拟合可靠——mid-game 棋子
    稀疏、选中红圈、阴影都会产生伪圆，把"圆命中率"当校验会被它们骗到
    （实测 25/44 被误判失效）。因此这里只做边界检查，真正的守门是上层
    的 validate_board（王在九宫、子力编制），它能自动发现棋盘翻转等异常。

    detect_circles: 是否跑整图霍夫圆检测（实测中位 125ms，占本函数耗时 98%）。
      默认 None = 只在 verbose 时做（诊断用，正常路径不跑）。
      True/False 可显式指定。circles 结果仅用于 verbose 打印的"圆命中"统计
      和 draw_overlay 可视化，运行时识别链路不使用它——所以正常路径跑它是纯浪费。
    """
    import json
    import os
    if calib is None and os.path.exists("board_calib.json"):
        calib = json.load(open("board_calib.json"))
    if not calib:
        return None
    affine = calib.get("affine") and "U" in calib
    cell_x, cell_y = calib["cell_x"], calib["cell_y"]
    ox, oy = calib["ox"], calib["oy"]
    h, w = img.shape[:2]
    if ox + 8 * cell_x >= w or oy + 9 * cell_y >= h:
        return None                      # 分辨率变了，标定失效

    if detect_circles is None:
        detect_circles = bool(verbose)
    circles = detect_piece_circles(img)[0] if detect_circles else None
    empty = np.zeros((0, 3))

    if affine:
        U = np.asarray(calib["U"], float)
        V = np.asarray(calib["V"], float)
        O = np.asarray(calib["O"], float)
        pts = {(i, j): tuple(O + i * U + j * V) for i in range(9) for j in range(10)}
        return {
            "cell_x": cell_x, "cell_y": cell_y, "ox": float(O[0]), "oy": float(O[1]),
            "points": pts, "ratio": cell_x / max(1e-6, cell_y),
            "circles": circles if circles is not None else empty,
            "inliers_x": 0, "inliers_y": 0, "from_calib": True, "affine": True,
        }

    hits = total = 0
    if circles is not None and len(circles):
        pts = np.array([(ox + i * cell_x, oy + j * cell_y)
                        for i in range(9) for j in range(10)])
        cs = circles[:, :2]
        d = np.hypot(pts[:, None, 0] - cs[None, :, 0], pts[:, None, 1] - cs[None, :, 1])
        near = d.min(axis=0)
        total = len(cs)
        hits = int((near < 0.22 * cell_x).sum())
    if verbose:
        print(f"  标定网格; 圆{total} 命中{hits}（仅供参考，不做否决）")
    return {
        "cell_x": cell_x, "cell_y": cell_y, "ox": ox, "oy": oy,
        "points": {(i, j): (ox + i * cell_x, oy + j * cell_y)
                   for i in range(9) for j in range(10)},
        "ratio": (cell_x * 8) / (cell_y * 9),
        "circles": circles if circles is not None else empty,
        "inliers_x": 0, "inliers_y": 0, "from_calib": True,
    }


def fit_affine_grid(circles, cell_x, cell_y, ox, oy, rounds=((0.50, 3), (0.30, 3), (0.15, 3))):
    """ICP 式仿射网格拟合：points ≈ O + i*U + j*V。

    轴对齐模型吸收不了 3D 渲染棋盘的轻微透视/旋转（实测 JJ象棋 x 坐标
    随列漂移 +15~21px），仿射模型可以。从 1D 拟合的近似网格出发，
    逐轮收缩容差迭代：分配格位 -> 最小二乘重估 O/U/V。
    返回 (O, U, V, 内点数, rms) 或 None。
    """
    pts = circles[:, :2].astype(float)
    O = np.array([ox, oy], float)
    U = np.array([cell_x, 0.0])
    V = np.array([0.0, cell_y])
    cell = max(cell_x, cell_y)
    inliers, rms = [], None
    for tol_frac, iters in rounds:
        tol = tol_frac * cell
        for _ in range(iters):
            A = np.array([[U[0], V[0]], [U[1], V[1]]])
            if abs(np.linalg.det(A)) < 1e-6:
                return None
            idxs, res = [], []
            for x, y in pts:
                try:
                    ij = np.linalg.solve(A, np.array([x, y]) - O)
                except np.linalg.LinAlgError:
                    continue
                i, j = int(round(ij[0])), int(round(ij[1]))
                d = np.hypot(*(O + i * U + j * V - (x, y)))
                if -3 <= i <= 11 and -3 <= j <= 12 and d < tol:
                    idxs.append((i, j, x, y))
            if len(idxs) < 20:
                return None
            M = np.array([[1.0, i, j] for i, j, _, _ in idxs])
            tx = np.array([x for _, _, x, _ in idxs])
            ty = np.array([y for _, _, _, y in idxs])
            (Ox, Ux, Vx), *_ = np.linalg.lstsq(M, tx, rcond=None)
            (Oy, Uy, Vy), *_ = np.linalg.lstsq(M, ty, rcond=None)
            O = np.array([Ox, Oy])
            U, V = np.array([Ux, Uy]), np.array([Vx, Vy])
        # 重新按最终参数分配并统计
        A = np.array([[U[0], V[0]], [U[1], V[1]]])
        idxs, res = [], []
        for x, y in pts:
            ij = np.linalg.solve(A, np.array([x, y]) - O)
            i, j = int(round(ij[0])), int(round(ij[1]))
            d = np.hypot(*(O + i * U + j * V - (x, y)))
            if -3 <= i <= 11 and -3 <= j <= 12 and d < tol:
                idxs.append((i, j, x, y))
                res.append(d)
        inliers = idxs
        rms = float(np.sqrt(np.mean(np.square(res)))) if res else 99.0
    if len(inliers) < 24 or rms > 0.12 * cell:
        return None
    return O, U, V, len(inliers), rms



def estimate_from_backranks(circles, guess, verbose=False):
    """用"棋子最多的一行"做初估——开局两排底线各有 9 子，是最强结构特征。

    1D 拟合会被头像/菜单图标/河界装饰带偏（实测 JJ象棋 被拉到 cell=119、
    原点落在屏幕底部），而从最密行入手杂圆骗不了它。
    返回 (ox, oy, cell_x, cell_y)。
    """
    xs, ys = circles[:, 0].astype(float), circles[:, 1].astype(float)
    order = np.argsort(ys)
    clusters, cur = [], [ys[order[0]]]
    for k in order[1:]:
        if ys[k] - cur[-1] <= 0.35 * guess:
            cur.append(ys[k])
        else:
            clusters.append((float(np.mean(cur)), len(cur)))
            cur = [ys[k]]
    clusters.append((float(np.mean(cur)), len(cur)))

    # 候选行按成员数降序；必须同时满足"相距约 9 格"，否则可能是菜单图标等
    # 同样密集的装饰行（实测 JJ象棋 底部菜单图标有 6 个，差点被当成底线）
    cand = sorted([c for c in clusters if c[1] >= 6], key=lambda c: -c[1])
    if len(cand) < 2:
        return None
    best = None
    for a in range(len(cand)):
        for b in range(a + 1, len(cand)):
            y0, y1 = min(cand[a][0], cand[b][0]), max(cand[a][0], cand[b][0])
            span = (y1 - y0) / 9.0
            if 0.75 * guess <= span <= 1.3 * guess:
                score = cand[a][1] + cand[b][1]
                if best is None or score > best[0]:
                    best = (score, y0, y1)
    if best is None:
        return None
    _, y_top, y_bot = best
    cell_y = (y_bot - y_top) / 9.0

    # 最密的一行用来定 x 网格
    dense = max(clusters, key=lambda c: c[1])
    row = np.sort(xs[np.abs(ys - dense[0]) < 0.35 * guess])
    if len(row) < 6:
        return None
    dif = np.diff(row)
    dif = dif[dif > 0.5 * guess]
    if len(dif) == 0:
        return None
    cell_x = float(np.median(dif))
    k = np.round((row - row.min()) / cell_x)
    ox = float(np.mean(row - k * cell_x))
    if verbose:
        print(f"  最密行 y={dense[0]:.0f}({dense[1]}子) 底线y={y_top:.0f}/{y_bot:.0f} "
              f"cell=({cell_x:.1f},{cell_y:.1f})")
    return ox, y_top, cell_x, cell_y


def locate_affine(img, verbose=False):
    """半径过滤 + 1D 初估 + 仿射精修的通用定位。任意 App/分辨率/轻微透视均可。"""
    circles, cell_guess = detect_piece_circles(img)
    if circles is None or len(circles) < 12:
        return None
    r_med = float(np.median(circles[:, 2]))
    keep = circles[(circles[:, 2] > 0.72 * r_med) & (circles[:, 2] < 1.35 * r_med)]
    if len(keep) < 12:
        keep = circles
    if verbose:
        print(f"  圆 {len(circles)} -> 半径过滤 {len(keep)}")

    guess = cell_guess if cell_guess else img.shape[1] / 9.3
    est = estimate_from_backranks(keep, guess, verbose=verbose)
    if est is None:
        # 退化时才回退到 1D 拟合
        fx = fit_grid_1d(keep[:, 0], guess, expected_span=8)
        fy = fit_grid_1d(keep[:, 1], guess, expected_span=9)
        if fx is None or fy is None:
            return None
        cx0, ix0, _, _ = fx
        cy0, iy0, _, _ = fy
        est = (ix0, iy0, cx0, cy0)
    ix0, iy0, cx0, cy0 = est
    est_x0, est_y0 = ix0, iy0        # 第 0 列 / 第 0 行的已知坐标
    if abs(cx0 - cy0) / max(cx0, cy0) > 0.25:
        return None

    aff = fit_affine_grid(keep, cx0, cy0, ix0, iy0)
    if aff is None:
        return None
    O, U, V, nin, rms = aff
    # 相位锚定：用已知结构（最密行=第 0 行、该行最左子=第 0 列）直接算平移量。
    # 不能靠统计杂圆的索引范围来平移——头像/菜单图标会把整张格网带偏若干行，
    # 取点落到棋盘外深色背景上，后续识别会全崩（实测偏移 3 行）。
    j_shift = int(round((est_y0 - O[1]) / V[1])) if abs(V[1]) > 1e-6 else 0
    i_shift = int(round((est_x0 - O[0]) / U[0])) if abs(U[0]) > 1e-6 else 0
    O = O + i_shift * U + j_shift * V
    if verbose:
        print(f"  仿射: U=({U[0]:.1f},{U[1]:.1f}) V=({V[0]:.1f},{V[1]:.1f}) "
              f"内点{nin} rms={rms:.1f}px 锚定偏移=({i_shift},{j_shift})")
    points = {(i, j): tuple(O + i * U + j * V) for i in range(9) for j in range(10)}
    return {
        "cell_x": float(np.hypot(*U)), "cell_y": float(np.hypot(*V)),
        "ox": float(O[0]), "oy": float(O[1]), "points": points,
        "ratio": float(np.hypot(*U)) / max(1e-6, float(np.hypot(*V))),
        "circles": keep, "inliers_x": nin, "inliers_y": nin,
        "affine": True, "U": U.tolist(), "V": V.tolist(), "O": O.tolist(),
    }


def anchor_axis(vals, origin, cell, n_lines, search=12):
    """把拟合相位锚定到棋盘第 0 格。

    对每个整数平移 s，统计检测点落在网格线 origin+(i+s)*cell 上的数量，
    取匹配数最大者；平移不改变匹配结构，所以只有正确 s 能让所有线同时对上。
    返回第 0 格的绝对坐标。
    """
    vals = np.asarray(vals, float)
    best = None
    for s in range(-search, 1):
        lines = origin + (np.arange(n_lines) + s) * cell
        d = np.abs(vals[:, None] - lines[None, :])
        near = d.min(axis=1)
        hit = int((near < 0.3 * cell).sum())
        resid = float((near[near < 0.3 * cell] ** 2).sum())
        if best is None or hit > best[0] or (hit == best[0] and resid < best[1]):
            best = (hit, resid, s)
    return origin + best[2] * cell, best[0]


def anchor_board(loc):
    """在 locate 结果上锚定 x/y 相位，返回对齐后的 points（键 (col,row)）。"""
    xs = loc["circles"][:, 0]
    ys = loc["circles"][:, 1]
    ox, hx = anchor_axis(xs, loc["ox"], loc["cell_x"], 9)
    oy, hy = anchor_axis(ys, loc["oy"], loc["cell_y"], 10)
    loc["ox"], loc["oy"] = ox, oy
    loc["anchor_hits"] = (hx, hy)
    loc["points"] = {
        (i, j): (ox + i * loc["cell_x"], oy + j * loc["cell_y"])
        for i in range(9) for j in range(10)
    }
    return loc


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "shots/step1_start.png"
    out = sys.argv[2] if len(sys.argv) > 2 else "shots/overlay.png"
    img = cv2.imread(src)
    if img is None:
        print(f"读不到 {src}")
        sys.exit(1)
    print(f"图像 {img.shape[1]}x{img.shape[0]}")
    loc = locate_board(img, verbose=True)
    if loc is None:
        print("定位失败")
        sys.exit(2)
    print(f"cell=({loc['cell_x']:.2f},{loc['cell_y']:.2f})  "
          f"相位=({loc['ox']:.1f},{loc['oy']:.1f})  圆检测数={len(loc['circles'])}")
    draw_overlay(img, loc, out)
    print(f"叠加图 -> {out}")

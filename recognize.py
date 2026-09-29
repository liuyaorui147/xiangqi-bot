"""棋子识别：KNN 模板匹配 + 空/红/黑判定 -> FEN。

坐标约定（已用引擎着法 b2e2 反推验证）：
- 屏幕网格 (col 0..8 左->右, row 0..9 上->下)，红方在下方
- 棋盘 rank 0 = 红方底线 => rank = 9 - row
- FEN 行序与屏幕 row 顺序一致（第 0 行是黑方底线）
"""
import glob
import os

import cv2
import numpy as np

from harvest import color_features

PIECES = "RNBAKCP rnbakcp".replace(" ", "")


class Recognizer:
    def __init__(self, tpl_dir="templates", color_ref="color_ref.json"):
        import json
        self.files = sorted(glob.glob(os.path.join(tpl_dir, "t_*.png")))
        if not self.files:
            raise RuntimeError(f"{tpl_dir} 里没有模板，先跑 harvest.py")
        self.labels, self.gray = [], []
        bad = []
        ok_files = []
        for f in self.files:
            base = os.path.basename(f)[2:-4]  # t_<piece>...
            pc = base.split("_")[0]
            img = cv2.imread(f)
            # imread 对损坏/非图片文件返回 None，直接 cvtColor 会抛
            # "!_src.empty()"。跳过坏的并报出来，别静默丢精度。
            if img is None or getattr(img, "size", 0) == 0:
                bad.append(f)
                continue
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
            self.labels.append(pc)
            self.gray.append(g)
            ok_files.append(f)
        if bad:
            print(f"[识别] 警告：{len(bad)} 个模板读不出来已跳过 -> {bad}")
        if not self.labels:
            raise RuntimeError(f"{tpl_dir} 里的模板全部读取失败，重新跑 harvest.py")
        self.files = ok_files
        m = np.asarray(self.gray)
        self._gm = m - m.mean(axis=1, keepdims=True)
        self._gn = np.linalg.norm(self._gm, axis=1)
        self._labels = np.asarray(self.labels)
        self.ref = json.load(open(color_ref)) if os.path.exists(color_ref) else None
        # 自适应阈值：由 harvest 用大津法标定，换皮肤/换 App 自动生效
        self.ink_thr = float(self.ref.get("ink_thr", 0.02)) if self.ref else 0.02
        proto = cv2.imread(self.files[0], cv2.IMREAD_GRAYSCALE)
        if proto is None or proto.size == 0:
            raise RuntimeError(f"模板 {self.files[0]} 无法读取，重新跑 harvest.py")
        self.tpl_shape = proto.shape  # (h, w)，所有 crop 统一缩放到该尺寸
        self.tpl_cell = None          # 收割时的参考格距，用于按比例换算搜索范围

    def _match(self, crops):
        """crops: (n, n_pixels) 灰度展平。返回 (n,) 最佳分数 与 (n,) 标签。"""
        gm = crops - crops.mean(axis=1, keepdims=True)
        gn = np.linalg.norm(gm, axis=1)
        denom = np.maximum(self._gn[None, :] * gn[:, None], 1e-9)
        s = (self._gm @ gm.T) / denom.T          # (n_tpl, n_crop)
        idx = s.argmax(axis=0)
        return s.max(axis=0), self._labels[idx]

    @staticmethod
    def _ring_offsets(max_r):
        """按距离从近到远生成偏移：中心 -> 环1 -> 环2 ..."""
        yield 0, 0
        for r in range(1, max_r + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) == r:
                        yield dx, dy

    def prep(self, img, loc, scale=0.52):
        """预计算整图灰度/HSV 与格距相关参数，避免每格重复转换。"""
        cell = (loc["cell_x"] + loc["cell_y"]) / 2
        return {
            "img": img, "gray": cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            "hsv": cv2.cvtColor(img, cv2.COLOR_BGR2HSV),
            "r": int(cell * scale), "cell": cell,
            "th": self.tpl_shape[0], "tw": self.tpl_shape[1],
            "max_search": max(2, int(round(5 * cell / 51.0))),
            "h": img.shape[0], "w": img.shape[1],
        }

    def _crop_gray(self, p, cx, cy, dx, dy):
        r = p["r"]
        x0, y0 = int(cx - r) + dx, int(cy - r) + dy
        x1, y1 = x0 + 2 * r, y0 + 2 * r
        if x0 < 0 or y0 < 0 or x1 > p["w"] or y1 > p["h"]:
            return None
        return cv2.resize(p["gray"][y0:y1, x0:x1], (p["tw"], p["th"]),
                          interpolation=cv2.INTER_AREA).astype(np.float32).ravel()

    def _side(self, p, cx, cy):
        """从预计算 HSV 上取中心格，判断空/红/黑。"""
        r = p["r"]
        x0, y0 = int(cx - r), int(cy - r)
        if x0 < 0 or y0 < 0 or x0 + 2 * r > p["w"] or y0 + 2 * r > p["h"]:
            return None
        hsv = p["hsv"][y0:y0 + 2 * r, x0:x0 + 2 * r]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        rr = float((((h < 8) | (h > 170)) & (s > 90) & (v > 60)).mean())
        dr = float(((s < 120) & (v < 90)).mean())
        if max(rr, dr) < self.ink_thr:      # 空格：几乎没有墨迹
            return None
        return "red" if rr > dr else "black"

    def cell_piece(self, p, x, y):
        """识别单个格位，返回 (棋子字符|None, 分数)。环形搜索 + 达标提前退出。"""
        side = self._side(p, x, y)
        if side is None:
            return None, 0.0
        best_s, best_pc = -1.0, None
        for dx, dy in self._ring_offsets(p["max_search"]):
            g = self._crop_gray(p, x, y, dx, dy)
            if g is None:
                continue
            s, lab = self._match(g[None, :])
            if float(s[0]) > best_s:
                best_s, best_pc = float(s[0]), str(lab[0])
            if best_s >= 0.92:
                break
        pc = best_pc.upper() if side == "red" else best_pc.lower()
        return pc, best_s

    def recognize(self, img, loc, max_search=None, scale=0.52, accept=0.92):
        """对 90 个格点做识别，返回 {(col,row): (piece, 分数)} 或 None。"""
        p = self.prep(img, loc, scale)
        if max_search is not None:
            p["max_search"] = max_search
        out = {}
        for j in range(10):
            for i in range(9):
                x, y = loc["points"][(i, j)]
                pc, s = self.cell_piece(p, x, y)
                out[(i, j)] = (pc, s) if pc else None
        return out

    def recognize_cells(self, img, loc, cells, scale=0.52):
        """只识别指定的少数格位，用于落子前后的快速复核（毫秒级）。"""
        p = self.prep(img, loc, scale)
        res = {}
        for key in cells:
            x, y = loc["points"][key]
            pc, s = self.cell_piece(p, x, y)
            res[key] = (pc, s) if pc else None
        return res


def moved_fen(board, src, dst):
    """把 (src->dst) 这一步应用到 board 上，直接推出落子后的 FEN。

    用于免去一次全盘识别就能得到"我方走完后的期望局面"。
    """
    b = dict(board)
    v = b.get(src)
    p = v[0] if isinstance(v, tuple) else v
    b[dst] = (p, 1.0)
    b[src] = None
    return to_fen(b)


def validate_board(board):
    """识别结果的常识校验，挡住垃圾 FEN（引擎遇到非法局面会崩）。"""
    counts, kings = {}, {"K": None, "k": None}
    for (c, r), v in board.items():
        p = v[0] if isinstance(v, tuple) else v
        if not p:
            continue
        counts[p] = counts.get(p, 0) + 1
        if p in kings:
            kings[p] = (c, r)
    if counts.get("K", 0) != 1 or counts.get("k", 0) != 1:
        return False, f"王数量异常 K={counts.get('K',0)} k={counts.get('k',0)}"
    for side, rows in (("K", (7, 8, 9)), ("k", (0, 1, 2))):
        c, r = kings[side]
        if not (3 <= c <= 5 and r in rows):
            return False, f"{side} 王不在九宫 ({c},{r})"
    up = sum(v for k, v in counts.items() if k.isupper())
    lo = sum(v for k, v in counts.items() if k.islower())
    if up > 16 or lo > 16:
        return False, f"子力超编 红{up} 黑{lo}"
    # 各兵种上限。以前只查总数 <=16，"认错一个炮成兵"这种误读（红方变成
    # 6 兵）照样能过，喂给引擎就是非法局面——引擎对此静默不响应，要等
    # 超时才被发现，白白卡好几秒。逐兵种卡上限能挡掉绝大多数这类误读。
    for p, cap in (("P", 5), ("R", 2), ("N", 2), ("C", 2), ("A", 2), ("B", 2),
                   ("p", 5), ("r", 2), ("n", 2), ("c", 2), ("a", 2), ("b", 2)):
        if counts.get(p, 0) > cap:
            return False, f"{p} 数量 {counts[p]} 超过上限 {cap}"
    for (c, r), v in board.items():
        p = v[0] if isinstance(v, tuple) else v
        if p == "P" and r == 9:
            return False, "红兵出现在底线"
        if p == "p" and r == 0:
            return False, "黑卒出现在底线"
    # 这里不查"将帅照面"：虽然规则上非法，但实测 Pikafish 能接受并给出
    # 结果，加了只会误杀引擎本来能处理的局面——而漏识别中间一颗子就会
    # 造成假照面，误杀的代价是 bot 停在那一手不动。
    return True, "ok"


def to_fen(board):
    """board: {(col,row): piece(char|None)} -> 标准象棋 FEN（不含走子方后缀）。"""
    rows = []
    for r in range(10):
        line, empty = "", 0
        for c in range(9):
            v = board.get((c, r))
            p = v[0] if isinstance(v, tuple) else v
            if p:
                if empty:
                    line += str(empty)
                    empty = 0
                line += p
            else:
                empty += 1
        if empty:
            line += str(empty)
        rows.append(line)
    return "/".join(rows)


def board_diagram(board):
    lines = []
    for r in range(10):
        cells = []
        for c in range(9):
            v = board.get((c, r))
            p = v[0] if isinstance(v, tuple) else v
            cells.append(p if p else ".")
        lines.append(f"{9 - r}  " + " ".join(cells))
    lines.append("   " + " ".join(chr(ord('a') + i) for i in range(9)))
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    import board_locator as bl

    src = sys.argv[1] if len(sys.argv) > 1 else "shots/step1_start.png"
    img = cv2.imread(src)
    loc = bl.locate_board(img)
    rec = Recognizer()
    board = rec.recognize(img, loc)
    print(board_diagram(board))
    fen = to_fen(board)
    print(f"\nFEN: {fen}")

    expect = ("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/"
              "P1P1P1P1P/1C5C1/9/RNBAKABNR")
    hit = sum(1 for a, b in zip(fen.split(" ")[0], expect) if a == b)
    print(f"与标准初始局面逐字符吻合: {hit}/{len(expect)}")
    print("完全一致！" if fen.split(" ")[0] == expect else "有差异，需调参")

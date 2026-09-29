"""天天象棋自动对弈主程序。

用法:
  python main.py watch            持续识别并打印棋盘（不落子）
  python main.py once             分析当前局面并替红方走一步
  python main.py once --dry       只算出走法，不点击
  python main.py auto             完整循环自动对弈（红方）
"""
import glob
import os
import sys
import threading as _threading
import time

import cv2
import numpy as np

import adb as adbmod
import board_locator as bl
import capture as cap
import clicker as clk
import mumu_cap as mumucap
from engine import Engine
from recognize import Recognizer, board_diagram, moved_fen, to_fen, validate_board

WINDOW = "天天象棋"
MY_SIDE = "red"          # 用户执红
THINK_MS = 350           # 思考时间（要更强可适当调大）
POLL = 0.35              # 轮询间隔
BOOT_WAIT = 2.5          # 新局开局观察窗口（等对手先动，黑先时对方会先落子）
MATE_DEPTH = 2           # 将死检测的搜索深度（有无合法着法第 1 层就看清，2 留余量）
USE_PONDER = True        # 我方落子后让引擎后台预搜索对手应手（不降棋力，命中即省）

PONDER = {"move": None, "expect": None}   # 后台预搜索的对手应手 + 命中后应有的局面
PENDING = [None]                          # 命中后取回的 (bestmove, res, fen_body)
PONDER_STAT = [0, 0]                      # [命中数, 尝试数]，命中率决定实际收益


def clear_ponder():
    """状态重置时停掉后台搜索并丢弃待用结果。

    不清的话，下一局会把上一局残留的预搜索结果当成自己的着法。
    ENGINE 是延迟初始化的全局量，这里用 globals() 取以免 NameError。
    """
    PONDER["move"] = PONDER["expect"] = None
    PENDING[0] = None
    eng = globals().get("ENGINE")
    try:
        if eng is not None and eng.pondering:
            eng.ponder_miss()
    except Exception:
        pass
MOVE_SETTLE = 0.35       # 落子后的动画等待上限（轮询到落定即提前返回）
TAP_GAP = 0.10           # 起点与终点两次点击的间隔
MUMU_TAP = True          # MuMu SDK 触摸点击（比 adb tap 快 4 倍且更稳，失败自动回退）
WAIT_MAX = 90            # 等对方落子的耐心上限，超过则怀疑我方未生效
POPUP_BRIGHT_RATIO = 0.7  # 画面亮度低于基线 70% 视为有压暗弹窗
AUTO_NEXT = True         # 对局结束后自动点「再来一局」
INITIAL_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"

BACKEND = None           # 'adb' | 'win'
SER = None
HW = None
MUMU = None              # MuMu 共享内存截图（~10ms/帧）；None/不可用时 grab 走 adb
CALIB = None             # 冻结的棋盘标定（由 harvest.py 写入）
BOARD_FLIP = False       # 当前棋盘是否相对"红在下"翻转了 180°（执黑时 App 会翻棋盘）
# 这两个必须在模块级先定义：shutdown_engine() 会在 Engine 创建之前就被调用
# （main() 的 finally 对任何返回路径都生效），没有初值会直接 NameError。
REC = None
ENGINE = None

import json as _json
import os as _os
if _os.path.exists("board_calib.json"):
    CALIB = _json.load(open("board_calib.json"))

# 允许外部配置覆盖（GUI 会把设置写进 bot_config.json）
_CFG = {}
if _os.path.exists("bot_config.json"):
    try:
        _CFG = _json.load(open("bot_config.json"))
    except Exception:
        _CFG = {}
THINK_MS = int(_CFG.get("think_ms", THINK_MS))
POLL = float(_CFG.get("poll", POLL))
BOOT_WAIT = float(_CFG.get("boot_wait", BOOT_WAIT))
MATE_DEPTH = int(_CFG.get("mate_depth", MATE_DEPTH))
USE_PONDER = bool(_CFG.get("ponder", USE_PONDER))
MOVE_SETTLE = float(_CFG.get("move_settle", MOVE_SETTLE))
TAP_GAP = float(_CFG.get("tap_gap", TAP_GAP))
MUMU_TAP = bool(_CFG.get("mumu_tap", MUMU_TAP))
WAIT_MAX = int(_CFG.get("wait_max", WAIT_MAX))
AUTO_NEXT = bool(_CFG.get("auto_next", AUTO_NEXT))
ENGINE_THREADS = int(_CFG.get("threads", 0)) or None
ENGINE_HASH = int(_CFG.get("hash_mb", 256))
# 后台预搜索的时间上限（毫秒）。Pikafish 的 ponder 是无限搜索、movetime 对
# 它无效，不设上限就会在等待对手的几十秒里满线程吃 CPU。3s 覆盖对手落子的
# 高概率窗口，之后 CPU 自动归零。
# 执红 / 执黑。执黑时棋盘朝向由 orient_board() 自动摆正，这里只决定
# 走子方（w/b）和"哪些子是我方的"。命令行 --black 可临时覆盖。
MY_SIDE = str(_CFG.get("side", MY_SIDE)).lower()
PONDER_MAX_MS = int(_CFG.get("ponder_max_ms", 3000))
# 等对手落子时的采样间隔。画面这段时间本就静止，25Hz 采样纯属浪费：
# frame_changed 一次约 9ms CPU，25 次/秒就是两成单核。降到 ~8Hz，
# 发现落子最多晚 80ms，几乎无感。
POLL_IDLE = float(_CFG.get("poll_idle", 0.12))


def init_backend(prefer_adb=True):
    """优先用 adb（坐标一致、不受遮挡影响），否则退回 Windows 窗口模式。

    adb 在线时还会尝试 MuMu 共享内存截图（~10ms/帧，adb screencap 约 640ms）：
    截图走 MuMu、点击仍走 adb tap，MuMu 不可用则自动回退 adb 截图。
    """
    global BACKEND, SER, HW, MUMU
    s = adbmod.first_device() if prefer_adb else None
    if s:
        BACKEND, SER = "adb", s
        if prefer_adb:  # auto/once 等真实对局才初始化，纯 adb 工具不必
            MUMU = mumucap.MuMuCap()
            if MUMU.open():
                print(f"[截图] MuMu 共享内存 {MUMU.w}x{MUMU.h}（~10ms/帧），点击仍走 adb")
            else:
                print(f"[截图] MuMu SDK 不可用（{MUMU.last_error}），截图回退 adb")
                MUMU = None
        return f"adb:{s}"
    hit = cap.find_window(WINDOW)
    if hit:
        BACKEND, HW = "win", hit[0]
        return f"win:{hit[0]} {hit[2]}"
    return None


def img_ok(img):
    """图像是否可用。空数组（shape 含 0）和 None 都算不可用。

    只判 `img is None` 是不够的：截图后端失败时可能返回一个 size==0 的
    数组（比如分辨率被报成 0），它能通过 None 检查，但 cv2 任何操作都会
    抛 "(-215:Assertion failed) !_src.empty()"。统一在这里挡掉。
    """
    return img is not None and getattr(img, "size", 0) > 0 and img.ndim >= 2


def grab():
    """抓一帧。保证返回值要么是可用的图像，要么是 None —— 绝不返回空数组。"""
    global MUMU
    img = _grab_raw()
    if img is None or img_ok(img):
        return img
    print(f"[截图] 丢弃空帧 shape={getattr(img, 'shape', None)}（后端返回空图像）")
    return None


def _grab_raw():
    global MUMU
    if BACKEND == "adb":
        if MUMU is not None:
            img = MUMU.screencap()
            if img is not None:
                return img
            # 共享内存连续失败已停用，本轮彻底放弃，之后恒走 adb 截图
            print(f"[截图] MuMu 停用（{MUMU.last_error}），回退 adb")
            MUMU.close()
            MUMU = None
        return adbmod.screencap(SER)
    img, _ = cap.capture_window(HW)
    return img


def click_point(x, y):
    if BACKEND == "adb":
        # MuMu SDK 触摸优先：~1ms/次 vs adb tap ~80ms，坐标同源已验证一致。
        # 失败（含 SDK 整体停用）自动回退 adb。
        if MUMU is not None and MUMU_TAP:
            if MUMU.tap(x, y):
                return True
            print(f"[点击] MuMu tap 停用（{MUMU.last_error}），回退 adb")
        return adbmod.tap(x, y, SER)
    sx, sy = clk.client_to_screen(HW, x, y)
    return clk.click_screen(sx, sy)


_FP_PREV = [None]        # 上一帧的缩略图指纹
_FP_THR = 0.3            # 变化判定阈值（实测：静止噪声 <0.08，走子变化 >1.0）


def frame_changed(img):
    """画面是否与上一帧实质不同。约 1ms，用来挡掉昂贵的全盘识别。

    全盘识别 90 格要 38ms，而一局棋里绝大多数帧画面是静止的（等对手思考）。
    先算整图缩略图指纹：没变就复用上一帧的识别结果，帧成本降到 10ms。
    阈值取 0.3 —— 离噪声上限 0.08 有 4 倍余量，离走子信号 1.0 有 3 倍余量。
    """
    if not img_ok(img):
        return False          # 空图不算"有变化"，别让 cvtColor 抛 !_src.empty()
    small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (72, 128),
                       interpolation=cv2.INTER_AREA)
    prev = _FP_PREV[0]
    _FP_PREV[0] = small
    if prev is None:
        return True
    return float(cv2.absdiff(small, prev).mean()) > _FP_THR


def orient_board(board):
    """把识别结果统一成"红方在下"的内部表示，返回 (board, 是否翻转过)。

    执黑时 App 会把棋盘翻过来（红在上、黑在下），而 `to_fen` 假定 row0 是
    黑方底线、`validate_board` 又要求红帅落在 row7-9 —— 直接喂进去会同时踩到
    两个坑：得到上下颠倒的镜像局面（引擎走法全错），且必然校验失败（表现为
    一直打点不落子）。

    这里按红帅所在半区判断朝向，必要时做 180° 归一化。之后所有逻辑都只需
    面对"红在下"这一种表示，不必各自关心朝向。
    """
    global BOARD_FLIP
    krow = None
    for (_c, r), v in board.items():
        p = v[0] if isinstance(v, tuple) else v
        if p == "K":
            krow = r
            break
    if krow is None:
        return board, BOARD_FLIP     # 认不出红帅：沿用上一帧的判断
    # 红帅只在 row7-9，翻转后落到 row0-2；row5 是河界，帅不可能跨过
    flip = krow < 5
    if not flip:
        return board, False
    return {(8 - i, 9 - j): v for (i, j), v in board.items()}, True


def raw_cell(cell):
    """归一化格点 -> 屏幕原始格点。

    识别结果已统一成"红在下"，但像素坐标仍是 App 实际画的样子，取点前必须
    反变换回去。180° 旋转是自逆的，正反变换共用同一个映射。
    """
    i, j = cell
    return (8 - i, 9 - j) if BOARD_FLIP else cell


def snapshot(tries=4, verbose=False, reuse=None):
    """抓一张 -> 定位 -> 识别 + 合法性校验。返回 (img, loc, board)。

    定位优先用冻结标定（棋盘像素位置恒定，mid-game 重新拟合反而易漂移），
    标定校验失败才回退自动拟合。识别结果要过常识校验，防止垃圾 FEN 喂引擎。

    reuse=(img, loc, board) 传上一帧结果；画面未变时直接复用，省掉识别开销。

    返回的 board 已归一化成"红在下"，像素坐标仍要用 raw_cell() 反变换后取。
    """
    global BOARD_FLIP
    img = None
    last_reason = ""
    for k in range(tries):
        img = grab()
        if img is None:
            time.sleep(0.4)
            continue
        if reuse is not None and not frame_changed(img):
            return img, reuse[1], reuse[2]
        loc = (bl.locate_with_calib(img, CALIB, verbose=verbose)
               or bl.locate_affine(img)
               or bl.locate_board(img))
        if loc is None:
            last_reason = "定位失败"
            time.sleep(0.5)
            continue
        board = REC.recognize(img, loc)
        # 校验之前先摆正朝向：翻转棋盘下 validate_board 的方向假设必然不成立
        board, BOARD_FLIP = orient_board(board)
        ok, reason = validate_board(board)
        if ok:
            return img, loc, board
        last_reason = reason
        if verbose:
            print(f"  [校验失败] {reason}")
        time.sleep(0.6)
    if verbose and last_reason:
        print(f"  [snapshot] {tries} 次均失败: {last_reason}")
    return img, None, None


def fen_cells(fen):
    """FEN 主体展开成 90 个格位（行优先），None 表示空格。"""
    out = []
    for row in fen.split("/"):
        for ch in row:
            if ch.isdigit():
                out.extend([None] * int(ch))
            else:
                out.append(ch)
    return out


def fen_diff(a, b):
    """两个局面相差几个格子。一步合法棋必然恰好改变 2 格（含吃子）。"""
    ca, cb = fen_cells(a), fen_cells(b)
    if len(ca) != len(cb) or len(ca) != 90:
        return 99
    return sum(1 for x, y in zip(ca, cb) if x != y)


def cells_to_uci(cell):
    """(col,row) -> 'b2'。uci_to_cells 的逆运算，rank = 9 - row。"""
    col, row = cell
    return chr(ord("a") + col) + str(9 - row)


def move_from_fen_diff(prev, cur):
    """从两个 FEN 主体的差推出走了哪一步（uci）。推不出返回 None。

    与 is_single_move 判据同源，但要把"哪格是哪格"明确下来——ponder 必须
    知道对手具体走了哪一步才能判断是否命中。
    """
    if not prev or not cur:
        return None      # expect 可能还没建立，别拿去 split
    ca, cb = fen_cells(prev), fen_cells(cur)
    if len(ca) != 90 or len(cb) != 90:
        return None
    idx = [i for i in range(90) if ca[i] != cb[i]]
    if len(idx) != 2:
        return None
    i, j = idx
    for a, b in ((i, j), (j, i)):
        # 棋子从 b 移到 a：a 现在是那颗子，b 已不是
        if cb[a] is not None and cb[a] == ca[b] and cb[b] != ca[b]:
            return cells_to_uci((b % 9, b // 9)) + cells_to_uci((a % 9, a // 9))
    return None


def cells_to_fen_body(cells):
    """fen_cells 的逆运算：90 个格位拼回 FEN 主体。"""
    rows = []
    for r in range(10):
        row, empty = "", 0
        for c in range(9):
            p = cells[r * 9 + c]
            if p is None:
                empty += 1
            else:
                if empty:
                    row += str(empty)
                    empty = 0
                row += p
        if empty:
            row += str(empty)
        rows.append(row)
    return "/".join(rows)


def fen_apply_move(fen_body, uci):
    """对 FEN 主体应用一步棋，返回新主体。用于推算 ponder 命中后的局面。"""
    cells = fen_cells(fen_body)
    if len(cells) != 90:
        return None
    src, dst = clk.uci_to_cells(uci)
    si, di = src[1] * 9 + src[0], dst[1] * 9 + dst[0]
    piece = cells[si]
    if piece is None:
        return None
    cells[si] = None
    cells[di] = piece
    return cells_to_fen_body(cells)


def is_single_move(prev, cur):
    """严格判定 prev->cur 是否像"一步棋"：恰好 2 格变化，且能找到
    '某颗子从 j 移到 i' 的完整证据（i 现在正是那颗子，j 变了）。

    比单纯"差 2 格"严格得多，能挡住结算框上按错误网格读出的垃圾局面。
    """
    ca, cb = fen_cells(prev), fen_cells(cur)
    if len(ca) != 90 or len(cb) != 90:
        return False
    idx = [i for i in range(90) if ca[i] != cb[i]]
    if len(idx) != 2:
        return False
    i, j = idx
    for a, b in ((i, j), (j, i)):
        if cb[a] is not None and cb[a] == ca[b] and cb[b] != ca[b]:
            return True
    return False


def flatten(board):
    """把 {(col,row): piece} 压成可比较的字符串签名。"""
    from recognize import to_fen
    return to_fen(board)


def play_once(dry=False):
    img, loc, board = snapshot()
    if loc is None:
        print("识别失败：找不到棋盘")
        return None
    print(board_diagram(board))
    fen = to_fen(board) + f" {'w' if MY_SIDE == 'red' else 'b'} - - 0 1"
    print(f"FEN: {fen}")

    t = time.time()
    bm, res = ENGINE.analyse(fen, movetime=THINK_MS)
    info = res["info"]
    print(f"引擎建议: {bm}  ({info.get('depth', '?')}层 "
          f"{info.get('nodes', '?')}节点)  耗时 {time.time() - t:.2f}s")
    if dry or not bm:
        return bm

    src, dst = clk.uci_to_cells(bm)
    mine = board.get(src)
    mine = mine[0] if isinstance(mine, tuple) else mine
    if not mine or (mine.isupper() != (MY_SIDE == "red")):
        print(f"拒绝落子：{src} 处是 {mine!r}，不像我方的子")
        return None
    print(f"落子 {src} -> {dst}  ({mine})")
    # board 已归一化成"红在下"，取像素点前要还原成屏幕实际朝向
    click_point(*loc["points"][raw_cell(src)])
    time.sleep(0.18)
    click_point(*loc["points"][raw_cell(dst)])
    time.sleep(0.5)
    img2 = grab()
    if img2 is not None:
        loc2 = bl.locate_board(img2)
        if loc2 is not None:
            b2 = REC.recognize(img2, loc2)
            print("落子后局面：")
            print(board_diagram(b2))
            print("局面已变化" if to_fen(b2) != to_fen(board) else "!! 局面没变，点击可能未生效")
    return bm


NEXT_BTN = {   # 结算框「再来一局」按钮的屏幕比例位置，按包名区分
    "com.tencent.qqgame.xq": (0.707, 0.842),
    "cn.jj.chess": (0.489, 0.926),
}
DEFAULT_BTN = (0.5, 0.9)


def current_package():
    if BACKEND == "adb":
        t = adbmod.top_activity(SER) or ""
        if " u0 " in t:
            seg = t.split(" u0 ")[1].split(" ")[0]
            return seg.split("/")[0]
    return ""


def opponent_mated(fen_full):
    """fen_full 必须是带走子方的完整 FEN（如 '... b - - 0 1'）。

    将死的明确信号：score mate 0，或 bestmove (none)。
    注意：引擎异常（bm 为 None）不能算将死——曾因喂了缺走子方的
    残缺 FEN 而误判绝杀，刚走一步就宣布获胜。
    """
    # 用固定深度而非 movetime：有没有合法着法第 1 层就能看清，
    # 原来跑 movetime=250 是白白多花 250ms，把将死检测拖成每步的固定开销。
    bm, res = ENGINE.analyse(fen_full, depth=MATE_DEPTH)
    i = res["info"]
    if i.get("scoretype") == "mate" and str(i.get("score")) == "0":
        return True
    return bm == "(none)"


def green_button(img=None, min_area=500):
    """找结算框上的绿色按钮（JJ象棋「再来一局」为绿底白字）。

    返回 (cx, cy, 面积) 或 None。结算框消失后返回 None，可用来验证点击是否生效。
    """
    if img is None:
        img = grab()
    if img is None:
        return None
    h = img.shape[0]
    try:
        b, g, r = cv2.split(img.astype(np.int16))
        mask = ((g - np.maximum(b, r)) > 40).astype(np.uint8)
        low = mask[int(h * 0.6):]
        n, _, stats, cent = cv2.connectedComponentsWithStats(low, 8)
        best = None
        for i in range(1, n):
            if stats[i][4] > min_area:
                cand = (float(stats[i][4]), float(cent[i][0]), float(cent[i][1]) + h * 0.6)
                if best is None or cand[0] > best[0]:
                    best = cand
        return best
    except Exception:
        return None


def flipped_initial():
    """初始局面从黑方视角看的 FEN（万一新局轮黑先、App 翻转棋盘）。"""
    rows = INITIAL_FEN.split("/")[::-1]
    return "/".join(r.swapcase() for r in rows)


def find_close_x(img=None):
    """在截屏里找广告弹窗的 × 关闭钮（模板匹配，多尺度）。返回 (x,y) 或 None。"""
    img = img if img is not None else grab()
    if img is None:
        return None
    _app_dir = os.path.dirname(os.path.abspath(__file__))
    tpl_path = os.path.join(_app_dir, "ad_close.png")
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        cand = os.path.join(exe_dir, "ad_close.png")
        tpl_path = cand if os.path.exists(cand) else tpl_path
        meipass = getattr(sys, "_MEIPASS", None)
        if not os.path.exists(tpl_path) and meipass:
            tpl_path = os.path.join(meipass, "ad_close.png")
    if not os.path.exists(tpl_path):
        return None
    tpl = cv2.imread(tpl_path)
    if tpl is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tg = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
    best = None
    for sc in (0.8, 1.0, 1.25):
        t = cv2.resize(tg, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        if t.shape[0] >= gray.shape[0] or t.shape[1] >= gray.shape[1]:
            continue
        m = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(m)
        if best is None or mx > best[0]:
            best = (mx, loc, t.shape[1] // 2, t.shape[0] // 2)
    if best and best[0] >= 0.62:
        mx, loc, hw, hh = best
        return loc[0] + hw, loc[1] + hh
    return None


def dismiss_popup(max_try=3):
    """关掉挡路的弹窗/广告。策略：× 按钮模板匹配 -> 安卓返回键兜底。

    弹窗会压暗整个屏幕（实测亮度 56 vs 正常 104），亮度比基线低 30% 即
    认为弹窗还在，重试。
    """
    for k in range(max_try):
        img = grab()
        if img is None:
            time.sleep(1.0)
            continue
        bright = float(img.mean())
        if k > 0 and bright > POPUP_BRIGHT_RATIO * popup_baseline[0]:
            return True                      # 屏幕亮度恢复，弹窗已关
        pos = find_close_x(img)
        if pos:
            print(f"[弹窗] 检测到 × 关闭钮 {pos}，点击")
            click_point(*pos)
        else:
            print("[弹窗] 未找到 ×，按返回键")
            if BACKEND == "adb":
                adbmod.key(4, SER)           # KEYCODE_BACK
            else:
                print("  （Windows 后端无返回键，跳过）")
        time.sleep(1.4)
    img = grab()
    return img is None or float(img.mean()) > POPUP_BRIGHT_RATIO * popup_baseline[0]


popup_baseline = [110.0]   # 正常对局画面的亮度基线（滑动平均）


def click_next_game(attempts=3, gap=1.5):
    """点「再来一局」并强校验，失败快速重试。

    两个坑：
    1. 结算动画播完前点击无效 -> 不能只点一次就干等，要 1.5s 一次连点。
    2. JJ象棋 结算画面会把棋盘缩小上移，冻结标定读出来是垃圾，但偶尔
       碰巧能拼出"合法"局面 -> 不能用"读到合法局面"当成功依据，
       必须读到标准初始局面（再来一局必然从初始局面开始）。
    """
    ok_fens = {INITIAL_FEN, flipped_initial()}
    for k in range(attempts):
        # 绿色按钮找不到时，多半是广告/奖励弹窗压在上面，先关弹窗再找
        if k >= 1 and green_button() is None:
            print("[结算框] 没找到按钮，尝试关闭弹窗")
            dismiss_popup()
        elif k == 0:
            time.sleep(1.0)      # 给结算动画留点时间
        btn = green_button()
        img_t = grab()
        if btn:
            x, y = int(btn[1]), int(btn[2])
            print(f"[结算框] 绿色按钮定位 ({x},{y}) 面积{int(btn[0])}")
        else:
            w = h = None
            if img_t is not None:
                h, w = img_t.shape[:2]
            if w is None:
                return False
            rx, ry = NEXT_BTN.get(current_package(), DEFAULT_BTN)
            x, y = int(w * rx), int(h * ry)
            print(f"[结算框] 按配置点击 ({x},{y})")
        click_point(x, y)
        # 不再干等固定 1.5s：新局界面一出现就返回（截图只要 ~10ms，
        # 轮询比固定 sleep 快，动画播完即可进入下一局）
        deadline = time.time() + gap + 1.5
        started = False
        while time.time() < deadline:
            if sleep_check(0.3): return False
            if green_button() is None:      # 结算按钮已消失 -> 不在结算页了
                _, _, b = snapshot(tries=1)
                if b is not None and to_fen(b) in ok_fens:
                    started = True
                    break
        if started:
            print("[结算框] 新对局已开始（初始局面确认）")
            return True
        if k == 0:
            print("[结算框] 结算动画未播完或点击未生效，快速重试...")
    print("[结算框] 多次尝试未能开始新局")
    return False


def handle_possible_game_end(img, weird):
    """棋盘持续消失或画面持续不稳定时的处理。返回 True 表示已处理（开新局）。"""
    # 先关掉可能压在上面的广告/奖励弹窗（亮度骤降是它的特征）
    try:
        img2 = grab()
        if img2 is not None and float(img2.mean()) < POPUP_BRIGHT_RATIO * popup_baseline[0]:
            print("[检测] 画面异常变暗，疑似弹窗")
            dismiss_popup()
    except Exception:
        pass
    if img is not None:
        try:
            _os.makedirs("shots", exist_ok=True)
            cv2.imwrite(f"shots/end_{int(time.time())}.png", img)
        except Exception:
            pass

    # 关键确认：必须看到"结算页"的特征才去点「再来一局」。
    # weird 在"局面持续变化"时也会累加，而中局的吃子动画、选中框、将军
    # 提示本来就让画面一直在变——不做这层确认的话，一次误判就要跑
    # click_next_game 的十次尝试（每次含 3 秒轮询，合计约 30 秒），
    # 表现为"中局突然卡住半分钟不动"。
    btn = None
    try:
        btn = green_button()
    except Exception:
        btn = None
    if btn is None:
        print("\n[检测] 画面不稳定但没找到结算页按钮，判定为中局动画/识别抖动，"
              "不点「再来一局」（避免卡 30 秒）")
        return False

    print("\n[检测] 棋盘持续消失/不稳定 ≥8s 且发现结算按钮，判定对局结束")
    if not AUTO_NEXT:
        print("已停止（AUTO_NEXT=False）")
        return False
    if click_next_game():
        print("[检测] 新对局已开始")
        return True
    print("[检测] 点击「再来一局」未生效，已停止")
    return False


STOP = _threading.Event()      # GUI 停止按钮置位，循环下一处检查点即退出


def sleep_check(t):
    """可被 STOP 中断的 sleep。返回 True 表示应当停止。

    GUI 在进程内跑机器人线程，必须能随时叫停；原来直接 time.sleep
    会让线程最多卡住一个完整周期才响应。
    """
    end = time.time() + t
    while True:
        if STOP.is_set():
            return True
        left = end - time.time()
        if left <= 0:
            return STOP.is_set()
        time.sleep(min(0.1, left))


def _calib_backup():
    """把当前标定/模板备份起来。标定一旦失败就回滚，避免把可用状态搞成砖。

    以前是直接删：标定中途任何一步出错（截图失败、定位失败、收割异常），
    模板和标定就都没了，之后连 auto 都起不来（Recognizer 报"没有模板"）。
    """
    import shutil
    stamp = time.strftime("%m%d_%H%M%S")
    bak = _os.path.join("shots", f"calib_backup_{stamp}")
    _os.makedirs(bak, exist_ok=True)
    n = 0
    for f in ("board_calib.json", "color_ref.json"):
        if _os.path.exists(f):
            shutil.copy2(f, bak)
            n += 1
    if _os.path.isdir("templates"):
        dst = _os.path.join(bak, "templates")
        shutil.copytree("templates", dst, dirs_exist_ok=True)
        n += len(glob.glob(_os.path.join(dst, "*")))
    return bak if n else None


def _calib_restore(bak):
    """从备份恢复标定/模板。"""
    import shutil
    if not bak or not _os.path.isdir(bak):
        return
    for f in ("board_calib.json", "color_ref.json"):
        src = _os.path.join(bak, f)
        if _os.path.exists(src):
            shutil.copy2(src, f)
    tdir = _os.path.join(bak, "templates")
    if _os.path.isdir(tdir):
        _os.makedirs("templates", exist_ok=True)
        for f in _os.listdir(tdir):
            shutil.copy2(_os.path.join(tdir, f), _os.path.join("templates", f))
    print(f"  已回滚到备份（{bak}），原有模板/标定不变")


def reload_calib():
    """重新加载标定并重建识别器。

    关键：CALIB 是模块级变量，只在 import 时读一次。GUI 是在同一个进程里
    反复跑 main()，标定写进 board_calib.json 后不刷新的话，接下来跑 auto
    用的还是**上一款 App 的旧标定** —— 棋盘格点对不上，识别全错。
    """
    global CALIB, REC
    try:
        CALIB = _json.load(open("board_calib.json")) if _os.path.exists("board_calib.json") else None
    except Exception as e:
        print(f"  标定文件损坏，已忽略: {e}")
        CALIB = None
    try:
        REC = Recognizer()
    except Exception as e:
        print(f"  重建识别器失败（模板可能不完整）: {e}")
    return CALIB


def calibrate():
    """一键适配新 App：清空旧标定 -> 自动定位 -> 按初始局面收割模板 -> 自校验。

    前置：把新象棋开到一局新棋的初始局面（一步未走）。
    失败会自动回滚，不会把程序搞成不可用状态。
    """
    print("开始标定（需要一个一步未走的新局面）")
    if BACKEND == "adb":
        print(f"  后端: adb {SER}" + (f" + MuMu 共享内存截图" if MUMU else ""))
    else:
        print(f"  后端: Windows 窗口 hwnd={HW}")
    bak = _calib_backup()
    if bak:
        print(f"  已备份旧标定/模板 -> {bak}（失败会自动回滚）")
    for f in ("board_calib.json", "color_ref.json"):
        if _os.path.exists(f):
            os.remove(f)
            print(f"  删除旧 {f}")
    for f in glob.glob("templates/t_*.png"):
        os.remove(f)

    img = grab()
    if img is None:
        print("截屏失败：请确认模拟器/游戏在前台")
        _calib_restore(bak)
        return False
    # 打包产物里没有 shots/，不建目录的话 imwrite 会静默失败（返回 False
    # 而不抛异常），后面 harvest 读到空图才崩，现象离原因很远。
    _os.makedirs("shots", exist_ok=True)
    if not cv2.imwrite("shots/calib_shot.png", img):
        print("截图保存失败：shots/ 目录不可写")
        _calib_restore(bak)
        return False
    loc = bl.locate_affine(img, verbose=True) or bl.locate_board(img)
    if loc is None:
        print("找不到棋盘：请确认已打开棋局界面（一步未走的初始局面）")
        _calib_restore(bak)
        return False
    print(f"定位成功 cell=({loc['cell_x']:.2f},{loc['cell_y']:.2f}) 宽高比={loc['ratio']:.3f}")

    saved_argv = sys.argv
    try:
        import harvest
        sys.argv = ["harvest.py", "shots/calib_shot.png", "--apply"]
        harvest.main()
    except Exception as e:
        print(f"收割模板失败: {type(e).__name__}: {e}")
        _calib_restore(bak)
        return False
    finally:
        sys.argv = saved_argv

    if len(glob.glob("templates/t_*.png")) < 10:
        print(f"!! 只收割到 {len(glob.glob('templates/t_*.png'))} 张模板（正常约 30 张），"
              "多半不是标准初始局面")
        _calib_restore(bak)
        return False

    # 用新模板自校验
    try:
        rec = Recognizer()
        b = rec.recognize(img, loc)
        # 执黑时棋盘是翻的，收出来的模板也跟着翻；校验前先摆正，否则必然
        # 与 INITIAL_FEN 对不上，标定会被误判为失败而回滚
        b, flipped = orient_board(b)
        print(f"  棋盘朝向: {'红方在上（已自动摆正）' if flipped else '红方在下'}")
        fen = to_fen(b)
        print(board_diagram(b))
        ok_n = sum(1 for a, c in zip(fen, INITIAL_FEN) if a == c)
        print(f"初始局面校验: {ok_n}/{len(INITIAL_FEN)} 字符一致")
    except Exception as e:
        print(f"!! 自校验异常: {type(e).__name__}: {e}")
        _calib_restore(bak)
        return False
    if ok_n != len(INITIAL_FEN):
        print("!! 校验不一致：请确认是标准初始局面（一步未走），或换个皮肤再试")
        _calib_restore(bak)
        return False

    # 关键：让本进程立刻用上新标定，否则接下来跑 auto 还用旧的那套
    reload_calib()
    print(f"标定完成（CALIB 已刷新 cell_x={CALIB['cell_x']:.2f}），可以直接跑 auto")
    return True


def shutdown_engine():
    """停掉后台搜索并彻底关闭引擎进程。可重复调用。

    以前 main() 跑完直接返回，ENGINE 从来不 quit —— pikafish 进程就这样
    一个一个堆在后台（GUI 里每点一次"开始"多一个）。残留的那个往往还停
    在 ponder 状态持续搜索，和新引擎抢 CPU，整机就卡了。
    """
    global ENGINE
    eng, ENGINE = ENGINE, None
    if eng is None:
        return
    try:
        if eng.pondering:
            eng.ponder_miss()      # 先停后台搜索，否则进程吃满 CPU 不肯退
    except Exception:
        pass
    try:
        eng.quit()
    except Exception:
        pass


def main():
    """包装一层：无论正常返回、异常还是被调用方中断，都收干净引擎进程。"""
    try:
        return _main_impl()
    finally:
        shutdown_engine()


def _main_impl():
    global REC, ENGINE, MY_SIDE
    mode = sys.argv[1] if len(sys.argv) > 1 else "watch"
    dry = "--dry" in sys.argv
    if "--black" in sys.argv:
        MY_SIDE = "black"
    elif "--red" in sys.argv:
        MY_SIDE = "red"
    print(f"执{'红（先手）' if MY_SIDE == 'red' else '黑（后手）'}"
          f"   棋盘朝向将按红帅位置自动摆正")

    hit = init_backend()
    if not hit:
        print("既没有可用的 adb 设备，也找不到游戏窗口")
        return 1
    print(f"后端就绪: {hit}")
    # 每次运行都读一次最新标定：GUI 是同进程反复跑 main()，只靠 import 时
    # 加载的话，重新标定后跑 auto 用的还是旧标定。
    reload_calib()

    if mode == "calib":
        return 0 if calibrate() else 1

    REC = Recognizer()
    shutdown_engine()          # 万一上一轮没关干净，这里先收掉，绝不并存两个进程
    ENGINE = Engine(threads=ENGINE_THREADS, hash_mb=ENGINE_HASH)

    if mode == "watch":
        print("watch 模式：Ctrl+C 退出")
        last = None
        while True:
            img, loc, board = snapshot()
            if board is None:
                print(f"[{time.strftime('%H:%M:%S')}] 未识别到棋盘", flush=True)
            else:
                sig = flatten(board)
                if sig != last:
                    last = sig
                    print(f"\n[{time.strftime('%H:%M:%S')}] 局面变化：")
                    print(board_diagram(board))
                else:
                    print(".", end="", flush=True)
            if sleep_check(POLL): return 0

    elif mode == "once":
        play_once(dry=dry)

    elif mode == "auto":
        print("auto 模式：红方自动对弈，Ctrl+C 退出")
        expect = None          # 我方走完后的期望局面
        state = "boot"         # boot -> my_turn -> wait_opp -> my_turn ...
        last_sig, stable = None, 0
        refusals = 0
        wait_start = None
        prev_ok = None       # 上一个被接受的稳定局面，用于"差 2 格"快通道
        boot_wait = None
        weird = 0                # 连续"看不到稳定棋盘"的次数
        prev_frame = None    # 上一帧 (img, loc, board)，供画面未变时复用
        clear_ponder()
        while True:
            # 连续读不到棋盘时打开 verbose，把失败原因（定位失败/校验不过的
            # 具体原因）打出来——否则卡住时日志里只有一串点，无从下手。
            img, loc, board = snapshot(verbose=(refusals > 0 or weird >= 5),
                                       reuse=prev_frame)
            if board is not None:
                prev_frame = (img, loc, board)
            if board is None:
                weird += 1
                print(".", end="", flush=True)
                if weird >= 8 and handle_possible_game_end(img, weird):
                    expect, state, weird, refusals = None, "boot", 0, 0
                    last_sig, stable, boot_wait, prev_ok = None, 0, None, None
                    prev_frame = None
                    clear_ponder()
                    if sleep_check(3): return 0
                    continue
                if sleep_check(POLL): return 0
                continue
            sig = to_fen(board)
            stable = stable + 1 if sig == last_sig else 1
            last_sig = sig
            if stable < 2:            # 等动画结束、局面稳定
                # 快通道：一步棋恰好改变 2 格，若与上一稳定局面正好差 2 格，
                # 说明对方已落子且动画结束，无需再等第二帧确认（省 1~1.5s）
                if prev_ok and is_single_move(prev_ok, sig):
                    pass
                else:
                    weird += 1
                    # 这个分支以前完全静默，卡死时什么都看不到。
                    # 每 10 轮强制汇报一次当前读到的局面。
                    if weird % 10 == 0:
                        print(f"\n[不稳定] 已连续 {weird} 帧局面在变 "
                              f"state={state} 当前={sig[:34]}...", flush=True)
                    if weird >= 12 and handle_possible_game_end(img, weird):
                        expect, state, weird, refusals = None, "boot", 0, 0
                        last_sig, stable, boot_wait, prev_ok = None, 0, None, None
                        prev_frame = None
                        clear_ponder()
                        if sleep_check(3): return 0
                        continue
                    if sleep_check(POLL): return 0
                    continue
            weird = 0
            prev_ok = sig
            if img is not None:
                # 更新正常画面亮度基线（弹窗检测的参照）
                popup_baseline[0] = popup_baseline[0] * 0.9 + float(img.mean()) * 0.1
            if state == "boot" and sig == INITIAL_FEN:
                # 新局可能轮到黑方先行，先观察 BOOT_WAIT 秒再动手。
                # 期间只要对手落子（局面变化），下一帧 sig 就不再是初始局面，
                # 条件不成立 -> 直接落到 my_turn 出手（此时正好轮到红）。
                # 原来写死 8s，且第一帧设置完 boot_wait 后没有 continue（等于没等），
                # 落子失败回到 boot 才真的干等 8s —— 开局因此显得很慢。
                if boot_wait is None:
                    boot_wait = time.time()
                    print(f"[开局] 初始局面，观察 {BOOT_WAIT:g}s 谁先手")
                if time.time() - boot_wait < BOOT_WAIT:
                    if sleep_check(POLL): return 0
                    continue
            boot_wait = None
            if state == "wait_opp":
                if sig != expect:
                    # 对方走了。先看是否命中后台预搜索：命中就直接取用结果，
                    # 未命中也只多一次停止往返，随后照常冷启动搜索。
                    if PONDER["move"] and ENGINE is not None and ENGINE.pondering:
                        opp = move_from_fen_diff(expect, sig)
                        if opp == PONDER["move"] and sig == PONDER["expect"]:
                            pbm, pres = ENGINE.ponder_hit()
                            if pbm:
                                PENDING[0] = (pbm, pres, sig)
                                PONDER_STAT[0] += 1
                                print(f"[ponder] 命中 {opp}，已搜 "
                                      f"{pres['info'].get('depth', '?')} 层")
                            else:
                                print("[ponder] 命中但无结果，回退正常搜索")
                        else:
                            ENGINE.ponder_miss()
                            print(f"[ponder] 未命中（预测 {PONDER['move']}，"
                                  f"实走 {opp}）")
                        PONDER["move"] = PONDER["expect"] = None
                        PONDER_STAT[1] += 1
                        if PONDER_STAT[1] % 5 == 0:
                            h, t = PONDER_STAT
                            print(f"[ponder] 累计命中 {h}/{t} = {h / t * 100:.0f}%"
                                  f"（命中则省掉 {THINK_MS}ms 搜索）")
                    state, wait_start = "my_turn", None          # 对方走了
                else:
                    if wait_start is None:
                        wait_start = time.time()
                    waited = time.time() - wait_start
                    # 看门狗：安静等待，但超过 WAIT_MAX 就认定"我方上一手没生效"，
                    # 强制重新分析，避免永久僵死
                    if waited > WAIT_MAX:
                        print(f"\n[看门狗] 已等 {waited:.0f}s 局面未变，"
                              f"怀疑我方上一步没生效，强制重新分析")
                        state, wait_start = "my_turn", None
                        if sleep_check(POLL): return 0
                        continue
                    if int(waited) % 20 == 0 and int(waited) > 0:
                        print(f"\n[等待对方 {waited:.0f}s] 当前 {sig[:28]}...", flush=True)
                    # 画面静止期降频采样：等待可能持续几十秒，没必要 25Hz
                    if sleep_check(POLL_IDLE): return 0
                    continue
            if state in ("boot", "my_turn"):
                print(f"\n[{time.strftime('%H:%M:%S')}] 轮到我方")
                status, fen_after = play_once_prepared(board, loc, dry=dry)
                if status == "ok":
                    expect, refusals = fen_after, 0
                    state = "wait_opp"
                elif status == "mate":
                    # 引擎确认将死（应用无关信号），直接开下一局
                    print("[终局] 自动开下一局")
                    if AUTO_NEXT and click_next_game():
                        expect, state, weird, refusals = None, "boot", 0, 0
                        last_sig, stable, boot_wait, prev_ok = None, 0, None, None
                        prev_frame = None
                        clear_ponder()
                        if sleep_check(3): return 0
                        continue
                    print("已停止（点击再来一局失败）")
                    return 1
                elif status == "stale":
                    # 局面在分析期间变了，下一轮用最新局面重新分析，不计失败
                    last_sig = None
                    stable = 0
                else:
                    refusals += 1
                    if refusals >= 5:
                        print("\n连续多次无法落子：可能对局已结束、轮次判断出错"
                              "，或当前不是红方走。已停止。")
                        return 1
                    state = "wait_opp" if expect else "boot"
                    if sleep_check(2): return 0
            if sleep_check(POLL): return 0
    return 0


def play_once_prepared(board, loc, dry=False):
    """已知 board/loc 时分析并落子。

    返回 (状态, 落子后 FEN)：
      ok   - 成功落子
      stale- 分析完到点击之间局面变了（对方动画刚结束），不算失败，下轮重来
      fail - 引擎无着法 / 来源不是我方棋子 / 点击无效
    """
    t_start = time.time()
    fen0 = to_fen(board)
    print(board_diagram(board))
    fen = fen0 + f" {'w' if MY_SIDE == 'red' else 'b'} - - 0 1"

    # 若上一轮后台预搜索命中，结果已经算好了，直接取用——省掉整段搜索时间
    bm, res = None, None
    if PENDING[0] is not None:
        pend_bm, pend_res, pend_fen = PENDING[0]
        PENDING[0] = None
        if pend_fen == fen0:
            bm, res = pend_bm, pend_res
            print(f"[ponder] 沿用后台搜索结果，跳过 {THINK_MS}ms 搜索")
        else:
            print("[ponder] 局面与后台搜索不符，作废改走正常搜索")

    if bm is None:
        bm, res = ENGINE.analyse(fen, movetime=THINK_MS)
    info = res["info"]
    print(f"引擎建议: {bm}  ({info.get('depth', '?')}层)  走 {bm}")
    # 我方被将死/困毙：引擎返回 (none) + mate 0
    if bm in (None, "(none)") and info.get("scoretype") == "mate":
        print("★ 我方被将死，本局结束")
        return "mate", None
    if dry or not bm:
        return "fail" if not dry else "stale", None
    src, dst = clk.uci_to_cells(bm)
    mine = board.get(src)
    mine = mine[0] if isinstance(mine, tuple) else mine
    if not mine or (mine.isupper() != (MY_SIDE == 'red')):
        print(f"拒绝：{src} 处是 {mine!r}，不是我方的子（可能轮次判断有误）")
        return "fail", None

    # 点击前快速复核：只查起终点两格（毫秒级），避免按过期的分析结果落子
    img_f = grab()
    loc_f = None
    if img_f is not None:
        loc_f = bl.locate_with_calib(img_f, CALIB) or bl.locate_board(img_f)
    if loc_f is None:
        return "stale", None
    # 复核与点击都发生在屏幕坐标系上，必须用 raw_cell 还原朝向
    r_src, r_dst = raw_cell(src), raw_cell(dst)
    chk = REC.recognize_cells(img_f, loc_f, [r_src, r_dst])
    a_p = chk.get(r_src)
    d_p = chk.get(r_dst)
    a_p = a_p[0] if a_p else None
    d_p = d_p[0] if d_p else None
    if a_p != mine or (d_p is not None and d_p.isupper() == (MY_SIDE == 'red')):
        print(f"局面在分析期间已变化（起点={a_p} 终点={d_p}），放弃本轮点击")
        return "stale", None
    loc = loc_f

    click_point(*loc["points"][r_src])
    time.sleep(TAP_GAP)
    click_point(*loc["points"][r_dst])
    print(f"已落子 {mine} {src}->{dst}")

    # 期望局面直接由走子推算，省掉一次全盘识别
    fen_expect = moved_fen(board, src, dst)
    # 等落子动画落定。原来固定 sleep 0.35s（按最坏情况取值），白白占用了
    # 这段时间——期间我方不轮询，对手若秒回就会被拖到下一轮才发现。
    # 截图现在只要 9ms，改成轮询：动画一落定立刻返回。
    deadline = time.time() + MOVE_SETTLE
    while True:
        time.sleep(0.05)
        s2p = d2p = None
        img2 = grab()
        if img2 is not None:
            loc2 = bl.locate_with_calib(img2, CALIB) or bl.locate_board(img2)
            if loc2 is not None:
                chk2 = REC.recognize_cells(img2, loc2, [r_src, r_dst])
                s2 = chk2.get(r_src)
                d2 = chk2.get(r_dst)
                s2p = s2[0] if s2 else None
                d2p = d2[0] if d2 else None
        # 起点空 + 终点有子 = 动画确实落定
        if s2p is None and d2p is not None:
            print("落子确认生效")
            break
        if time.time() >= deadline:
            # 判定标准：起点变空 = 棋子确实离开了 = 落子生效。
            # 动画中间帧会让"起点/终点都显得空"，此时不能判失败；
            # 真正失败的特征是棋子仍留在起点。
            if s2p is not None and d2p is None:
                print(f"!! 落子可能未生效（棋子仍在起点 {s2p}）")
            else:
                print("落子确认：动画未完全落定，交下一轮校验")
            break
    # 绝杀检测（与应用无关）：走完后看对方是否已无路可走
    # 必须补全走子方字段，否则引擎拿到残缺 FEN 会静默保持旧局面
    side_after = "b" if MY_SIDE == "red" else "w"
    if not dry and opponent_mated(f"{fen_expect} {side_after} - - 0 1"):
        print("★ 绝杀！对方已无路可走，本局获胜")
        return "mate", fen_expect

    # 绝杀检测要用引擎，必须放在 ponder 之前——否则它会把后台搜索打断。
    # 我方落子后对手还在思考，这段时间正好让引擎按预测应手继续往下搜。
    if not dry and USE_PONDER and res.get("ponder"):
        pm = res["ponder"]
        exp_body = fen_apply_move(fen_expect, pm)
        if exp_body and ENGINE.ponder_start(
                f"{fen_expect} {side_after} - - 0 1", [], pm, THINK_MS,
                max_ms=PONDER_MAX_MS):
            PONDER["move"] = pm
            PONDER["expect"] = exp_body
            print(f"[ponder] 后台预搜索对手应手 {pm}")

    print(f"本手耗时 {time.time() - t_start:.2f}s（识别+思考+点击+复核）")
    # 截屏失败/棋盘消失：多半是结算框弹出，交给主循环的结束检测
    return "ok", fen_expect


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已退出")

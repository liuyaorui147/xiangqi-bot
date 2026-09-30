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
import advice
import board_locator as bl
import capture as cap
import clicker as clk
import mumu_cap as mumucap
from engine import Engine
from recognize import Recognizer, board_diagram, moved_fen, to_fen, validate_board

WINDOW = "天天象棋"
MY_SIDE = "red"          # 用户执红
SIDE_FLIPS = [0]         # 本局执子方换边次数，用于僵局熔断
SIDE_GUESS = [False]     # 朝向先验是否已用过（每局只定一次，避免反复改判）
THINK_MS = 350           # 思考时间（要更强可适当调大）
POLL = 0.35              # 轮询间隔
BOOT_WAIT = 2.5          # 新局开局观察窗口（等对手先动，黑先时对方会先落子）
BOOT_PATIENCE = 12.0     # 判黑（我方后手）时，最多再耐心等多久对手先落子
MATE_DEPTH = 2           # 将死检测的搜索深度（有无合法着法第 1 层就看清，2 留余量）
MATE_SEARCH_DEPTH = 8    # 残局找强制杀的搜索深度（固定深度，够看清连杀即可）
USE_PONDER = True        # 我方落子后让引擎后台预搜索对手应手（不降棋力，命中即省）

PONDER = {"move": None, "expect": None}   # 后台预搜索的对手应手 + 命中后应有的局面
PENDING = [None]                          # 命中后取回的 (bestmove, res, fen_body)
PONDER_STAT = [0, 0]                      # [命中数, 尝试数]，命中率决定实际收益
# 同一手连续"落子未生效"的记录。counter>=1 时下一次点击前要先复位 UI 的选中态，
# 否则第二次同样的两次点击会把刚到位的子又走回去（有状态的棋类 UI 通病）。
MISS_STAT = {"move": None, "count": 0}
CONFIRMED = [False]                       # 上一手是否真的确认落到新格（自动执子锁定用）
_LAST_FAIL = [""]                         # 上次打印的校验失败原因（同原因不重复刷屏）
FORCE_RESNAP = [False]                    # 引擎拒局后要求主循环丢帧重新识别
MISS_HEAL = [0]                           # 连续"落子未生效"的自愈轮数，攒够才认输

# 自动判定我方执红/执黑（长时间挂机必需：App 每局先后手不固定，写死红方会
# 落到"给对手的着法 + 点击被拒"）。关闭条件：命令行 --red/--black 显式指定，
# 或 bot_config.json 里 "auto_side": false。判定成功后 SIDE_LOCK 锁定，
# 每局结束开新局时会解锁重判。
AUTO_SIDE = True
SIDE_LOCK = None


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


def apply_side(side, reason, lock=False):
    """切换 / 确认我方执子方。

    MY_SIDE 决定喂给引擎的走子方（w/b）以及"哪些子算我方的"，判错会让引擎
    给出对手的着法、点击被 App 拒绝，表现为反复"落子未生效"直到停机。挂机时
    每局先后手由 App 决定，写死成红方不可靠，所以这里做自动判定：
      先验   棋盘朝向——App 通常把我方摆在屏幕下方（board 已归一化成"红在下"，
             flip=True 说明屏幕上是"红在上、黑在下"，于是先验为执黑）
      定案   开局谁先落子——红先手是规则级信号，不受界面摆法影响
      校验   首手确实下去之后锁定；若某手连续失败则怀疑判反，自动换边
    换边必须清掉后台预搜索：残留结果属于旧局面，也可能属于另一方。
    """
    global MY_SIDE, SIDE_LOCK
    side = "black" if side == "black" else "red"
    if MY_SIDE == side and (not lock or SIDE_LOCK == side):
        if lock:
            SIDE_LOCK = side
        return MY_SIDE
    MY_SIDE = side
    # 换边计数：红↔黑来回横跳却始终落不下子，说明根本不在对局页（实测卡在
    # 结算动画的棋盘残影上，29 子的残局照样能过校验）。攒够了就熔断。
    SIDE_FLIPS[0] += 1
    clear_ponder()
    if lock:
        SIDE_LOCK = side
        SIDE_FLIPS[0] = 0        # 首手已确认生效，僵局风险解除
        SIDE_GUESS[0] = False
    tag = "红（先手）" if side == "red" else "黑（后手）"
    tail = "" if SIDE_LOCK else "，待首手生效后确认"
    print(f"[执子] {tag}  依据：{reason}{tail}", flush=True)
    return MY_SIDE


def report_winrate(info, board, flip, tag="", deep=False):
    """算胜率并输出。返回平滑后的显示值（0~1），算不出来返回 None。

    三层保障，缺一层都会出现"胜率 100%"这种假象：
      1. flip —— info 是对手走子方视角时必须翻转，否则我方大优会报成接近 0；
      2. 子力自洽 —— 评估分超出局面子力能解释的范围就是识别错帧，丢弃；
      3. 平滑 —— 单帧最多挪 20%，避免错帧和浅搜抖动把曲线打飞。

    "胜率:" 前缀是 GUI 的解析约定，不能改。
    """
    if not info:
        return None
    ok, why = advice.eval_credible(board, info, MY_SIDE)
    if not ok:
        # 结算动画、弹窗期间会连续读到垃圾局面，不节流的话一秒刷好几条。
        # 每 10 秒最多报一次，其余静默丢弃——胜率仍然保持上一个可信值。
        global _WR_REJECT_T
        if time.time() - _WR_REJECT_T >= 10:
            _WR_REJECT_T = time.time()
            print(f"  [胜率] 本帧不可信：{why}")
        return WR_SMOOTHER.value
    raw = advice.win_percent(info, flip=flip, board=board, my_side=MY_SIDE,
                             strict=True)
    if raw is None:
        return WR_SMOOTHER.value
    LAST_WR_RAW[0] = raw
    shown = WR_SMOOTHER.update(raw)
    ev = advice.eval_text(info)
    if flip:
        # 显示给人的评估值也要跟着翻：否则"我方大优"会显示成负分
        try:
            if info.get("scoretype") == "cp":
                ev = f"{'+' if int(info['score']) <= 0 else ''}{-int(info['score'])}"
            elif info.get("scoretype") == "mate":
                ev = f"M{-int(info['score'])}"
        except (TypeError, ValueError):
            pass
    extra = ""
    if abs(shown - raw) > 0.02:
        extra = f"  (平滑前 {raw * 100:.1f}%)"
    if info.get("bound"):
        extra += f"  [{info['bound']}bound，仅供参考]"
    print(f"胜率: {shown * 100:.1f}%   评估 {ev}   "
          f"{tag}{'深度评估' if deep else ''}{extra}")
    return shown


def deep_eval_on_wait(fen_body, board_after, opp_to_move=True, dry=False):
    """等对手落子期间做一次深度评估，把胜率算准。

    时机：我方刚落子、画面进入静止等待。此时 CPU 和引擎都闲着，而对局
    节奏一点不受影响（人类/对手随时可能落子，我们会立刻中断等待转去搜索）。

    注意视角：我方落子后轮到对手走，FEN 的走子方是对手，所以引擎给出的
    score/wdl 是**对手视角**，必须 flip 成我方胜率——这正是漏翻转会报出
    反向胜率的地方。

    中途对手落子会打断这次评估（主循环发现画面变化后会切走），此时结果
    作废，不会污染显示。
    """
    if not EVAL_ON_WAIT or ENGINE is None or dry:
        return
    # 走子方必须按"谁走"算出来，不能写死成 'b'。原来写死成黑走，隐含假设
    # "对手总是黑方"——我方执黑时对手是红，这里就成了我方走子方，引擎给的
    # 是我方视角的分数，下面 flip=True 再翻一次，方向彻底反过来：实战日志里
    # 出现"评估 -6191、平滑前 0.0%"一路走低，实际是我方大优（执黑时那串
    # 深搜数值符号全是反的）。
    mine_ch = "w" if MY_SIDE == "red" else "b"
    side_ch = ("b" if mine_ch == "w" else "w") if opp_to_move else mine_ch
    fen = f"{fen_body} {side_ch} - - 0 1"
    try:
        # go() 内部会先停掉后台 ponder，不用在这里手动干预。
        # retries=0：评估失败不值得重启引擎（那要重加载 NNUE，卡好几秒）。
        _bm, res = ENGINE.analyse(fen, movetime=EVAL_MS, retries=0,
                                  timeout=EVAL_MS / 1000.0 * 2 + 1.0)
    except Exception as e:
        print(f"  [深度评估] 失败: {type(e).__name__}: {e}")
        return
    if res.get("timeout") or not res.get("info"):
        print("  [深度评估] 引擎无响应，沿用常规搜索的胜率")
        return
    d = res["info"].get("depth", "?")
    report_winrate(res["info"], board_after, flip=True, tag=f"{d}层 ", deep=True)


def neutral_tap(loc):
    """点棋盘外侧一处空白，复位上一次点击可能留下的"选中棋子"状态。

    棋类 App 的点击是有状态的：误判"落子未生效"后按同一对坐标再走一遍，第二
    次的 tap 会把刚到位的子重新选中，第三下点回去就把它走回原地。日志里
    h0g2 连点 8 次就是这么来的。事先点一下棋盘格之外的空白，能把这个状态机
    拉回中性。取点在棋盘左上角外侧，坐标不合法（出屏）时宁可不动手。
    """
    pts = loc.get("points") if loc else None
    if not pts or "cell_x" not in loc:
        return False
    x0, y0 = pts[(0, 0)]
    x, y = x0 - loc["cell_x"] * 0.6, y0 - loc["cell_y"] * 0.6
    if x < 4 or y < 4:
        return False
    click_point(x, y)
    time.sleep(TAP_GAP)
    return True


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
BOOT_PATIENCE = float(_CFG.get("boot_patience", BOOT_PATIENCE))
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
# 配置里 side 允许写 "auto"：交给 AUTO_SIDE 自动判定（挂机场景）。
_side_cfg = str(_CFG.get("side", MY_SIDE)).lower()
MY_SIDE = _side_cfg if _side_cfg in ("red", "black") else MY_SIDE
AUTO_SIDE = bool(_CFG.get("auto_side", AUTO_SIDE))
PONDER_MAX_MS = int(_CFG.get("ponder_max_ms", 3000))
# 等对手落子时的采样间隔。画面这段时间本就静止，25Hz 采样纯属浪费：
# frame_changed 一次约 9ms CPU，25 次/秒就是两成单核。降到 ~8Hz，
# 发现落子最多晚 80ms，几乎无感。
POLL_IDLE = float(_CFG.get("poll_idle", 0.12))
# 胜率专用深度评估：与"选着搜索"分开跑。
# 选着为了抢时间只有 350ms（约 15~18 层），复杂局面下分数会抖好几个百分点；
# 胜率是要给人看的，值得多花时间。等对手落子时我们有几十秒空闲，拿 1.5s
# 做一次深搜，既不影响节奏又能拿到稳定得多的 WDL。
# 这里必须用 movetime 而不是固定深度：depth 22 在中局可能跑几十秒，
# 而 analyse 一旦超时会重启引擎（重新加载 48MB NNUE，卡好几秒），
# 反而拖慢对局。movetime 保证引擎按时返回着法，超时不会发生。
EVAL_MS = int(_CFG.get("eval_ms", 1500))
EVAL_ON_WAIT = bool(_CFG.get("eval_on_wait", True))
# 扫「再来一局」按钮的间隔。结算页会停留很久，不必高频；太密则白白多做
# 几次颜色判定（每次约十几毫秒）。
END_CHECK_GAP = float(_CFG.get("end_check_gap", 2.0))
# 胜率平滑器：抗识别抖动 + 抗浅搜抖动，见 advice.WinRateSmoother
WR_SMOOTHER = advice.WinRateSmoother()
LAST_WR_RAW = [None]     # 最近一次可信的原始胜率，供日志对照
_WR_REJECT_T = 0.0       # 上次打印"本帧不可信"的时间（节流，避免刷屏）


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


LAST_SNAP_REASON = ""    # 上一次 snapshot 失败的具体原因，供主循环区分"等待态"


def is_empty_board_wait(reason=""):
    """是不是"棋盘存在但没棋子 / 压根没棋盘"的等待态。

    每次开局、每次自动续局都要经过"匹配中 → 摆子中"这段，此时画面上要么
    没有棋盘、要么是空棋盘。旧代码把这种帧当成异常局面，每 2 秒打一条
    [校验失败]、还累加 weird，攒够 8 次又去跑结算页检测（一次约 30 秒）。
    结果是每次续局都空转刷屏几十秒——挂机场景下一局接一局，白白浪费。

    判据：连王都没有。终局结算页是有棋子的（王还在），所以不会误判成等待态。
    """
    r = reason or LAST_SNAP_REASON
    if not r:
        return False
    if "定位失败" in r:
        return True          # 连棋盘都找不到 → 不在对局画面
    # "王数量异常 K=0 k=0"：格子读到了但一个王都没有 → 空棋盘/摆子中
    return "K=0" in r and "k=0" in r


def snapshot(tries=4, verbose=False, reuse=None):
    """抓一张 -> 定位 -> 识别 + 合法性校验。返回 (img, loc, board)。

    定位优先用冻结标定（棋盘像素位置恒定，mid-game 重新拟合反而易漂移），
    标定校验失败才回退自动拟合。识别结果要过常识校验，防止垃圾 FEN 喂引擎。

    reuse=(img, loc, board) 传上一帧结果；画面未变时直接复用，省掉识别开销。

    返回的 board 已归一化成"红在下"，像素坐标仍要用 raw_cell() 反变换后取。
    """
    global BOARD_FLIP, LAST_SNAP_REASON
    img = None
    last_reason = ""
    LAST_SNAP_REASON = ""
    for k in range(tries):
        img = grab()
        if img is None:
            time.sleep(0.4)
            continue
        if reuse is not None and not frame_changed(img):
            # 复用帧同样要同步朝向。BOARD_FLIP 决定 raw_cell() 怎么把归一化
            # 坐标反变换回屏幕坐标，而它是全局量、只在这里和下面赋值 —— 画面
            # 静止时走这条捷径直接 return，flip 就会残留上一次的值。换 App
            # 重新标定、上一局执黑、程序重启都可能让它与画面不符，于是点击
            # 落到镜像位置，落子前复核永远不过，表现成"一直思考不落子"。
            if len(reuse) >= 4 and reuse[3] is not None:
                BOARD_FLIP = reuse[3]
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
        LAST_SNAP_REASON = reason
        # 节流：结算页/摆子动画期每一帧都是"王数量异常"，逐帧打印会把日志
        # 刷爆（实测一局 11 条），真正有用的信息反而看不见。同一种原因连续
        # 出现只报一次，原因变了才再报。
        if verbose and reason != _LAST_FAIL[0]:
            print(f"  [校验失败] {reason}")
        _LAST_FAIL[0] = reason
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


def green_button(img=None, min_area=500, max_area=None, min_y_ratio=None):
    """找结算框上的绿色按钮（JJ象棋「再来一局」为绿底白字）。

    返回 (面积, cx, cy) 或 None。结算框消失后返回 None，可用来验证点击是否生效。

    max_area / min_y_ratio 是给"主动扫结算页"用的收紧条件。纯颜色判定太粗，
    实测对局界面里有面积 5000+、位置偏上的绿色元素会被误判成结算按钮，
    于是 bot 在对局中连点好几下。结算页那个按钮的实测特征：
    面积 600~1500、中心 y 约在屏幕 0.89 处（很靠下）。
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
            area = float(stats[i][4])
            if area <= min_area:
                continue
            cy = float(cent[i][1]) + h * 0.6
            if max_area is not None and area > max_area:
                continue
            if min_y_ratio is not None and cy < h * min_y_ratio:
                continue
            cand = (area, float(cent[i][0]), cy)
            if best is None or cand[0] > best[0]:
                best = cand
        return best
    except Exception:
        return None


def flipped_initial():
    """初始局面从黑方视角看的 FEN（万一新局轮黑先、App 翻转棋盘）。"""
    rows = INITIAL_FEN.split("/")[::-1]
    return "/".join(r.swapcase() for r in rows)


def is_initial_position(fen):
    """是否"一步未走"的开局局面（容忍 1~3 格识别噪声）。

    摆子刚完成时经常有一两个子被认到邻格——实测开局把红炮认到了 e2（标准
    是 b2/h2），于是严格判 `sig == INITIAL_FEN` 失败，开局观察期被整个跳过，
    bot 立刻按朝向先验出手，在对手还没走时抢先落子（未生效）。所以这里按
    字符一致率判定，差 3 个字符以内仍算开局。
    """
    if not fen:
        return False
    body = fen.split(" ")[0] if " " in fen else fen
    # 先卡满编：中局/残局子数必然少于 32，单靠字符差异会把"只差 3 个字符"
    # 的中局也误判成开局（实测踩过）。一个子被认到邻格会产生 2~4 个字符
    # 差异，所以满编之后再放宽到 8 个字符。
    if sum(1 for ch in body if ch.isalpha()) != 32:
        return False
    for ref in (INITIAL_FEN, flipped_initial()):
        if body == ref:
            return True
        if len(body) != len(ref):
            continue
        ok = sum(1 for a, b in zip(body, ref) if a == b)
        if len(ref) - ok <= 8:
            return True
    return False


ARENA_TPL = [None]       # 「10分钟」按钮模板灰度图缓存（None=未加载, False=加载失败）


def find_arena_10(img=None, min_score=0.75):
    """在开始界面找「10分钟」场次按钮（模板匹配，多尺度）。返回 (x,y) 或 None。

    排位开始界面底部有一排金色场次按钮（5分钟/10分钟/15分钟/超快），
    「10分钟」按钮带独特白字，模板匹配特征足够独特：对局页、结算页、
    弹窗都不会命中，所以命中即可认为当前正处于可发起对局的界面。
    按钮必然在屏幕下半部，命中位置偏上视为误匹配。
    """
    img = img if img is not None else grab()
    if img is None:
        return None
    if ARENA_TPL[0] is None:
        _app_dir = os.path.dirname(os.path.abspath(__file__))
        tpl_path = os.path.join(_app_dir, "templates", "arena_10.png")
        if getattr(sys, "frozen", False):
            exe_dir = os.path.dirname(sys.executable)
            cand = os.path.join(exe_dir, "templates", "arena_10.png")
            tpl_path = cand if os.path.exists(cand) else tpl_path
            meipass = getattr(sys, "_MEIPASS", None)
            if not os.path.exists(tpl_path) and meipass:
                tpl_path = os.path.join(meipass, "templates", "arena_10.png")
        t = cv2.imread(tpl_path)
        if t is None:
            print("[开局] 缺少模板 templates/arena_10.png，无法自动点场次")
            ARENA_TPL[0] = False
        else:
            ARENA_TPL[0] = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
    tg = ARENA_TPL[0]
    if tg is None or tg is False:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    best = None
    for sc in (0.9, 1.0, 1.1):
        t = cv2.resize(tg, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        if t.shape[0] >= gray.shape[0] or t.shape[1] >= gray.shape[1]:
            continue
        m = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(m)
        if best is None or mx > best[0]:
            best = (mx, loc, t.shape[1] // 2, t.shape[0] // 2)
    if not best or best[0] < min_score:
        return None
    mx, loc, hw, hh = best
    if loc[1] + hh < img.shape[0] * 0.6:   # 场次按钮只在屏幕下部
        return None
    return loc[0] + hw, loc[1] + hh


ARENA_CLICK_T = [0.0]    # 上次点场次按钮的时间，防连点


def start_arena_game(img):
    """开始界面点「10分钟场」发起匹配。返回 True=当前是开始界面（已处理/刚点过）。

    点完不等待：匹配中画面会变化，主循环的"等待棋局"分支自然接管。
    执子方不指定——匹配是真实随机的，开局后由朝向/先手信号自动判定。
    """
    btn = find_arena_10(img)
    if btn is None:
        return False
    if time.time() - ARENA_CLICK_T[0] < 8.0:
        return True             # 刚点过，等界面响应，别连点
    ARENA_CLICK_T[0] = time.time()
    x, y = int(btn[0]), int(btn[1])
    print(f"\n[开局] 开始界面点「10分钟场」({x},{y})，匹配随机先后手", flush=True)
    click_point(x, y)
    return True


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
        # 绿色按钮找不到时有两种可能，处置方式完全相反，必须先分清：
        #   a) 广告/奖励弹窗压在上面（特征是画面变暗）→ 关弹窗；
        #   b) 上一次点击已生效、结算页关闭、正在进匹配（画面正常）→ 停手。
        # 以前不分青红皂白就 dismiss_popup()，它会按返回键，把刚发起的
        # 匹配页直接退出去，接着又在非结算页上按配置坐标乱点，越搞越糟。
        if k >= 1 and green_button() is None:
            img0 = grab()
            dark = (img0 is not None
                    and float(img0.mean()) < POPUP_BRIGHT_RATIO * popup_baseline[0])
            if dark:
                print("[结算框] 疑似弹窗遮挡，先关弹窗")
                dismiss_popup()
            else:
                print("[结算框] 结算页已消失（应已进入匹配），停止点击")
                return True
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
        while time.time() < deadline:
            if sleep_check(0.3): return False
            if green_button() is None:      # 结算按钮已消失 -> 不在结算页了
                _, _, b = snapshot(tries=1)
                if b is not None and to_fen(b) in ok_fens:
                    print("[结算框] 新对局已开始（初始局面确认）")
                    return True
                # 按钮没了但子还没摆好 = 正在匹配/摆子。这同样算成功：
                # 继续在这儿等初始局面会白等（匹配可能要几十秒），而主循环
                # 本来就有"等待棋局"分支接管，不该在这一直点。
                print("[结算框] 结算页已关闭，等待摆子")
                return True
        if k == 0:
            print("[结算框] 结算动画未播完或点击未生效，快速重试...")
    print("[结算框] 多次尝试未能开始新局")
    return False


END_CHECK_T = [0.0]      # 上次扫「再来一局」按钮的时间
END_HITS = [0]           # 连续扫到绿色按钮的次数


def try_settlement(img, force=False):
    """周期性扫「再来一局」按钮，命中两次就开新局。返回 True=已开新局。

    force=True 跳过"间隔"和"连中两次"两道限制立刻扫一次。给僵局熔断用：
    那时 bot 正在结算动画期的残影上反复点棋盘，等不到下一轮常规扫描。

    结算页必须主动扫，靠"读不到棋盘"来触发是不够的，而且会漏：
      - 有时结算页上棋盘还在（终局局面），校验照样通过 → bot 当成对局中，
        在结算页上反复尝试走子，卡死不动；
      - 有时结算页把棋盘缩小上移，冻结标定读出来是垃圾 → 走"等待态"分支
        静默等待，同样永远轮不到结算检测。
    两种路径都会经过这里，所以放在两处都调用。

    连中两次才动手：green_button 是纯颜色判定，对局界面偶尔有绿色元素，
    误报一次就要跑 click_next_game 的十次尝试（约 30 秒）。
    """
    if not AUTO_NEXT:
        return False
    now = time.time()
    if not force and now - END_CHECK_T[0] < END_CHECK_GAP:
        return False
    END_CHECK_T[0] = now
    try:
        # 收紧条件只在主动扫描时用：结算按钮面积几百到一千五、位置很靠下。
        # 实测对局界面的绿色元素面积能到 5000+ 且位置偏上，不卡这两条就会
        # 在对局中误判成结算页，连点好几下干扰对局。
        btn = green_button(img, min_area=400, max_area=2600, min_y_ratio=0.82)
    except Exception:
        btn = None
    END_HITS[0] = END_HITS[0] + 1 if btn is not None else 0
    if not force and END_HITS[0] < 2:
        return False
    END_HITS[0] = 0
    print("\n[结算] 检测到「再来一局」按钮，本局已结束", flush=True)
    if click_next_game():
        print("[结算] 新对局已开始")
        return True
    print("[结算] 点击未生效，继续尝试")
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


def log_run_header(backend):
    """把这次运行的前提一次性写进日志（不进界面，界面地方有限）。

    排查问题时最常缺的就是"当时到底用的什么配置"：有没有换过 App、标定是新的
    还是旧的、模板是哪套、思考时间设了多少。这些在窗口关掉之后无从查起，
    必须一开始就在文件里留底。
    """
    import glob as g
    import logger
    if not logger.path():
        return
    if CALIB:
        kind = "仿射标定" if CALIB.get("affine") else "网格标定"
        calib = f"{kind} cell={CALIB.get('cell_x', 0):.1f}x{CALIB.get('cell_y', 0):.1f}"
    else:
        calib = "无（每次自动拟合）"
    try:
        pkg = current_package() or "(未取到)"
    except Exception:
        pkg = "(取包名失败)"
    logger.header("运行开始", [
        f"时间      {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"命令行    {' '.join(sys.argv)}",
        f"执子      {MY_SIDE}{'（自动判定）' if AUTO_SIDE else ''}",
        f"后端      {backend}",
        f"当前 App  {pkg}",
        f"标定      {calib}",
        f"模板      {len(g.glob('templates/t_*.png'))} 个",
        f"思考      {THINK_MS}ms  线程 {ENGINE_THREADS or '自动'}  哈希 {ENGINE_HASH}MB",
        f"节奏      poll={POLL}/{POLL_IDLE}s  tap_gap={TAP_GAP}s  settle={MOVE_SETTLE}s",
        f"开关      mumu_tap={MUMU_TAP}  自动下一局={AUTO_NEXT}",
        f"日志文件  {logger.path()}",
    ])
    # 界面上也留一行：出问题时要知道该去翻哪个文件
    print(f"日志文件: {logger.path()}")


def main():
    """包装一层：无论正常返回、异常还是被调用方中断，都收干净引擎进程。

    顺便在这层装日志——_sys.stdout 一进来就被接管，之后每一步都落盘。
    uninstall 放在最后：收引擎时万一有异常打印，也还能记进文件。
    """
    import logger
    try:
        logger.install()
        return _main_impl()
    finally:
        try:
            shutdown_engine()
        finally:
            logger.uninstall()


def _main_impl():
    global REC, ENGINE, MY_SIDE, SIDE_LOCK, AUTO_SIDE
    mode = sys.argv[1] if len(sys.argv) > 1 else "watch"
    dry = "--dry" in sys.argv
    if "--black" in sys.argv:
        MY_SIDE, AUTO_SIDE = "black", False
    elif "--red" in sys.argv:
        MY_SIDE, AUTO_SIDE = "red", False
    for _a in sys.argv:
        if _a.startswith("--side="):
            _v = _a.split("=", 1)[1].lower()
            if _v == "auto":
                AUTO_SIDE = True
            else:
                MY_SIDE = "black" if _v == "black" else "red"
                AUTO_SIDE = False
    # 界面/GUI 每次启动都是新一轮 runtime：上一轮锁定的执子方不能带过来
    SIDE_LOCK = None
    SIDE_FLIPS[0] = 0
    SIDE_GUESS[0] = False
    MISS_STAT["move"], MISS_STAT["count"] = None, 0
    print(f"执{'红（先手）' if MY_SIDE == 'red' else '黑（后手）'}"
          f"{'（自动判定，开局后确认）' if AUTO_SIDE else ''}"
          f"   棋盘朝向将按红帅位置自动摆正")

    hit = init_backend()
    if not hit:
        print("既没有可用的 adb 设备，也找不到游戏窗口")
        return 1
    print(f"后端就绪: {hit}")
    log_run_header(hit)
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
        pre_move_sig = None    # 我方落子**前**的局面，用来分辨"没生效"vs"对手走了"
        state = "boot"         # boot -> my_turn -> wait_opp -> my_turn ...
        last_sig, stable = None, 0
        refusals = 0
        wait_start = None
        prev_ok = None       # 上一个被接受的稳定局面，用于"差 2 格"快通道
        boot_wait = None
        weird = 0                # 连续"看不到稳定棋盘"的次数
        stales = 0               # 连续"落子前复核不通过"次数，防止无限循环
        # 上一帧 (img, loc, board, flip)，供画面未变时复用；flip 必须与 board
        # 一起带走，否则复用时会沿用旧朝向（见 snapshot 的 reuse 分支）
        prev_frame = None
        wait_log = 0.0           # 上一次打"等待对方"的秒数，避免同 1 秒刷屏
        wait_board_log = 0.0     # 上一次打"等待棋局"的秒数（匹配/摆子中）
        wait_board_t0 = time.time()
        end_check_t = 0.0        # 上次扫结算页按钮的时间
        end_hits = 0             # 连续扫到绿色按钮的次数
        opp_noise = 0            # 等对手时"变化不像一步棋"的连续次数
        clear_ponder()
        while True:
            # 连续读不到棋盘时打开 verbose，把失败原因（定位失败/校验不过的
            # 具体原因）打出来——否则卡住时日志里只有一串点，无从下手。
            img, loc, board = snapshot(verbose=(refusals > 0 or weird >= 5),
                                       reuse=prev_frame)
            if board is not None:
                prev_frame = (img, loc, board, BOARD_FLIP)
            if board is None:
                # 读不到棋盘的**任何**原因都可能是结算页：匹配中（真没棋盘）、
                # 摆子中（空盘）、结算页把棋盘缩小上移（读出 K=0 k=2 这种
                # 垃圾局面）。所以结算检测必须放在最前面，不能只挂在某一条
                # 分支上——实测挂在后面会导致结算页一直识别不出来。
                if try_settlement(img):
                    expect, state, weird, refusals, stales = None, "boot", 0, 0, 0
                    SIDE_LOCK = None
                    SIDE_FLIPS[0] = 0
                    SIDE_GUESS[0] = False
                    MISS_STAT["move"], MISS_STAT["count"] = None, 0
                    WR_SMOOTHER.reset()
                    last_sig, stable, boot_wait, prev_ok = None, 0, None, None
                    prev_frame = None
                    clear_ponder()
                    if sleep_check(3): return 0
                    continue
                # 开始界面（场次选择页）：主动点「10分钟场」发起对局。
                # 放在空盘等待之前——开始界面同样读不到棋盘，但那不是
                # "匹配中"，不点的话 bot 会一直干等。
                if start_arena_game(img):
                    weird = 0
                    if sleep_check(POLL_IDLE): return 0
                    continue
                # 匹配中 / 摆子中：画面上根本没有完整棋局。这是每局必经的
                # 阶段，不是异常——安静等就行。旧代码照样累加 weird 并打点，
                # 攒够 8 次还会去跑结算页检测，每次续局空转几十秒。
                if is_empty_board_wait():
                    if wait_board_log == 0.0:
                        wait_board_log = time.time()
                        print("\n[等待] 未识别到棋局（匹配/摆子中），静默等待…",
                              flush=True)
                    elif time.time() - wait_board_log >= 15:
                        wait_board_log = time.time()
                        print(f"\n[等待] 仍未识别到棋局 "
                              f"({time.time() - wait_board_t0:.0f}s)",
                              flush=True)
                    # 不累加 weird：这里不是"局面异常"，不该触发结算页检测
                    weird = 0
                    if sleep_check(POLL_IDLE): return 0
                    continue
                wait_board_log, wait_board_t0 = 0.0, time.time()
                weird += 1
                print(".", end="", flush=True)
                if weird >= 8 and handle_possible_game_end(img, weird):
                    expect, state, weird, refusals, stales = None, "boot", 0, 0, 0
                    SIDE_LOCK = None      # 新局重新判定先后手
                    SIDE_FLIPS[0] = 0
                    SIDE_GUESS[0] = False
                    MISS_STAT["move"], MISS_STAT["count"] = None, 0
                    WR_SMOOTHER.reset()   # 胜率别带着上一局的值
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
                        expect, state, weird, refusals, stales = None, "boot", 0, 0, 0
                        SIDE_LOCK = None      # 新局重新判定先后手
                        SIDE_FLIPS[0] = 0
                        SIDE_GUESS[0] = False
                        MISS_STAT["move"], MISS_STAT["count"] = None, 0
                        last_sig, stable, boot_wait, prev_ok = None, 0, None, None
                        prev_frame = None
                        clear_ponder()
                        if sleep_check(3): return 0
                        continue
                    if sleep_check(POLL): return 0
                    continue
            weird = 0
            prev_ok = sig
            # 结算页必须主动查，见 try_settlement 的说明
            if try_settlement(img):
                expect, state, weird, refusals, stales = None, "boot", 0, 0, 0
                SIDE_LOCK = None      # 新局重新判定先后手
                SIDE_FLIPS[0] = 0
                SIDE_GUESS[0] = False
                MISS_STAT["move"], MISS_STAT["count"] = None, 0
                WR_SMOOTHER.reset()   # 胜率别带着上一局的值
                last_sig, stable, boot_wait, prev_ok = None, 0, None, None
                prev_frame = None
                clear_ponder()
                if sleep_check(3): return 0
                continue
            if img is not None:
                # 更新正常画面亮度基线（弹窗检测的参照）
                popup_baseline[0] = popup_baseline[0] * 0.9 + float(img.mean()) * 0.1
            if AUTO_SIDE and SIDE_LOCK is None and state == "boot" \
                    and not SIDE_GUESS[0]:
                # 先验：这类 App 通常把我方的棋子摆在屏幕下方。board 已归一化成
                # "红在下"，flip=True 意味着屏幕上是"红在上、黑在下" -> 我执黑。
                # 只是先验，真正定案看下面"开局谁先落子"（红先手，规则级信号）。
                #
                # 只定一次（SIDE_GUESS）：原来每帧强行按朝向改判，于是开局期
                # 出现"朝向判黑 -> 观察期满判红 -> 下一帧朝向又把红改回黑"的
                # 拉锯（实测 5 秒内改判 3 次）。朝向只是弱先验，不能反复推翻
                # 更强的规则级信号，定完就交给后面的判据修正。
                guess = "black" if BOARD_FLIP else "red"
                SIDE_GUESS[0] = True
                if MY_SIDE != guess:
                    apply_side(guess, "棋盘朝向（我方通常在屏幕下方）")
            if state == "boot" and is_initial_position(sig):
                # 初始局面（32 子一步未走）只会出现在新局。上一局锁定的执子方
                # 在这里必须作废——否则会带着上一局的"执黑"进入红先手的新局，
                # 抢在对手前面落子，连点 4 次不生效后停机（实战 21:31 那次）。
                if SIDE_LOCK is not None:
                    SIDE_LOCK = None
                    SIDE_FLIPS[0] = 0
                    SIDE_GUESS[0] = False
                    print("[开局] 初始局面：作废上一局锁定的执子方，重新判定")
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
                # 观察期满对手还没动。这里**不能一律判红抢走**：朝向先验若判
                # 我方执黑，那我是后手，对手（人类）思考十几秒再落子很正常，
                # 照老逻辑 2.5s 就改判红、抢在对手前面落子 —— 实战里表现为
                # "执黑时开局乱点/不响应"。所以：判黑就继续耐心等到
                # BOOT_PATIENCE，届时对手仍无动作，才认定朝向先验错了、改判红。
                if AUTO_SIDE and SIDE_LOCK is None:
                    if MY_SIDE == "black" and \
                            time.time() - boot_wait < BOOT_PATIENCE:
                        if sleep_check(POLL_IDLE): return 0
                        continue
                    apply_side("red",
                               f"开局 {BOOT_PATIENCE:g}s 内对手未落子 → 我方先手")
            elif AUTO_SIDE and SIDE_LOCK is None and state == "boot":
                # boot 状态下局面不是初始局面 -> 推断是对手先走了。
                # 但这条**只在真开局成立**：bot 在中途重启时看到的也是"非初始
                # 局面"（那是它自己走过若干步的结果），照这条推断就会误判成
                # 执黑，于是 FEN 写成黑走，引擎拒局（实测 side=b 无 bestmove），
                # 表现为每步卡 8 秒后放弃。所以必须卡子数。
                n_now = sum(1 for v in board.values()
                            if (v[0] if isinstance(v, tuple) else v))
                if n_now >= 28:
                    apply_side("black", "开局对手已先落子")
            boot_wait = None
            if state == "wait_opp":
                if pre_move_sig and sig == pre_move_sig:
                    # 局面退回我方落子之前 —— 这是"上一手没生效"，不是对手走棋。
                    # 坑在于：未生效时 expect(理论落子后) 与实际局面正好差 2 格，
                    # is_single_move 会当成一步棋通过，于是 bot 认定"对手应了
                    # 一手"、接着再走同一手，每 3 秒一次无限循环（实测卡死 30+ 手）。
                    MISS_STAT["count"] = MISS_STAT.get("count", 0) + 1
                    print(f"\n[未生效] 局面回到落子前，我方上一手未生效"
                          f"（连续 {MISS_STAT['count']} 次），重走", flush=True)
                    expect, last_sig, stable = None, None, 0
                    prev_frame = None
                    state = "my_turn"
                    if sleep_check(POLL): return 0
                    continue
                if sig != expect:
                    # 变化必须确实是"一步棋"才能认定对手走了。
                    # expect 是落子后按理论推算的 FEN，实测帧总有 1~2 格出入
                    # （模板匹配在动画/高亮下会抖），只判 sig != expect 的话
                    # 噪声就被当成对手落子，bot 于是自己接着走——日志里出现
                    # 过 2~3 秒连走三手的自战。噪声连续多次才让步，避免
                    # expect 本身错了导致永久僵死。
                    if expect and not is_single_move(expect, sig):
                        opp_noise += 1
                        if opp_noise == 1:
                            print("\n[等待] 局面有变化但不像一步棋，"
                                  "按识别噪声处理", flush=True)
                        if opp_noise <= 6:
                            if sleep_check(POLL): return 0
                            continue
                    opp_noise = 0
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
                    advice.forget_diff()   # 对手这一步可能吃子，子力差跳变合法
                else:
                    if wait_start is None:
                        wait_start = time.time()
                        wait_log = 0.0
                    waited = time.time() - wait_start
                    # 看门狗：安静等待，但超过 WAIT_MAX 就认定"我方上一手没生效"，
                    # 强制重新分析，避免永久僵死
                    if waited > WAIT_MAX:
                        print(f"\n[看门狗] 已等 {waited:.0f}s 局面未变，"
                              f"怀疑我方上一步没生效，强制重新分析")
                        state, wait_start = "my_turn", None
                        if sleep_check(POLL): return 0
                        continue
                    # waited 落在 20.0~20.9 这一整秒里每个采样都会满足条件，原来
                    # 直接按 int(waited) % 20 判断，同一秒能刷出七八条同样的日志。
                    if waited - wait_log >= 20:
                        wait_log = waited
                        print(f"\n[等待对方 {waited:.0f}s] 当前 {sig[:28]}...",
                              flush=True)
                    # 画面静止期降频采样：等待可能持续几十秒，没必要 25Hz
                    if sleep_check(POLL_IDLE): return 0
                    continue
            if state in ("boot", "my_turn"):
                print(f"\n[{time.strftime('%H:%M:%S')}] 轮到我方")
                pre_move_sig = sig          # 落子前的局面，用于事后分辨"没生效"
                status, fen_after = play_once_prepared(board, loc, dry=dry)
                if status == "ok":
                    expect, refusals = fen_after, 0
                    stales = 0
                    MISS_HEAL[0] = 0       # 走出去了，之前未生效的自愈计数归零
                    state = "wait_opp"
                    # 我方这一手可能吃子，子力差跳 400+ 是正常的，别让守门员
                    # 把紧随其后的深搜极端评估（真杀势）当成错帧拦掉。
                    advice.forget_diff()
                    opp_noise = 0       # 新的一轮等待，噪声计数归零
                    # 首手真正落定且画面确认了 => 执子方判断成立，锁住不再改。
                    # 没确认落到新格的（只差动画）不算验证过，留待下一手。
                    if AUTO_SIDE and SIDE_LOCK is None and CONFIRMED[0]:
                        apply_side(MY_SIDE, "首手落子已确认生效", lock=True)
                elif status == "miss":
                    # 走子请求发出去了，但棋子没离开起点。这里绝对不能直接把
                    # "理论上的落子后局面"当成新局面（旧代码正是这么做，于是
                    # 下一轮照着同一个 false expectation 再发一次同手，最多见过
                    # 同一手连点 8 次）。静置一下让 App 的动画/状态机收尾，
                    # 重新全盘识别后再试；连续不成则怀疑执子方判反，自动换边。
                    last_sig, stable = None, 0
                    prev_frame = None
                    cnt = MISS_STAT["count"]
                    if SIDE_FLIPS[0] >= 3:
                        # 熔断：换边三次都落不下子，几乎可以肯定不在对局页。
                        # 继续换边只会红↔黑无限横跳（实测 20:52 起卡了整整
                        # 一分钟、每 3 秒重发同一手）。停手处理页面本身。
                        print(f"\n[僵局] 执子方已换边 {SIDE_FLIPS[0]} 次仍无法落子，"
                              f"判定不在对局页：关弹窗 + 强制扫结算页", flush=True)
                        dismiss_popup()
                        if try_settlement(img, force=True):
                            expect, state, weird, refusals, stales = None, "boot", 0, 0, 0
                            SIDE_LOCK = None
                            SIDE_FLIPS[0] = 0
                            SIDE_GUESS[0] = False
                            MISS_STAT["move"], MISS_STAT["count"] = None, 0
                            WR_SMOOTHER.reset()
                            last_sig, stable, boot_wait, prev_ok = None, 0, None, None
                            prev_frame = None
                            clear_ponder()
                            if sleep_check(3): return 0
                            continue
                        print("\n[僵局] 已停止（页面既非对局页也扫不到结算页，"
                              "请把模拟器停在对局或开始界面后重启）")
                        return 1
                    if AUTO_SIDE and SIDE_LOCK is None and cnt >= 2:
                        other = "black" if MY_SIDE == "red" else "red"
                        apply_side(other, f"同一手连续 {cnt} 次未生效，执子方可能判反")
                        MISS_STAT["count"] = 0
                        expect, boot_wait = None, None
                        state = "boot"
                    elif cnt >= 4:
                        # 直接停机太可惜：绝大多数情况是"并非我方回合"（对手
                        # 还没走完 / 刚进新局抢跑了）或被弹窗挡住，退回去重判
                        # 就能自愈。所以先自愈两轮，攒到 8 次才认输停机。
                        print(f"\n同一手连续 {cnt} 次未能落子：可能是弹窗挡住或"
                              f"并非我方回合，先自愈（关弹窗+复位+重判）")
                        dismiss_popup()
                        neutral_tap(loc)
                        MISS_STAT["move"], MISS_STAT["count"] = None, 0
                        MISS_HEAL[0] += 1
                        expect, boot_wait = None, None
                        prev_frame = None
                        clear_ponder()
                        state = "boot"
                        if MISS_HEAL[0] >= 3:
                            print("\n自愈无效（已重试 3 轮）：可能是 App 结算/"
                                  "弹窗挡住，或当前并非我方回合。已停止。")
                            return 1
                    else:
                        state = "boot" if not expect else "wait_opp"
                    if sleep_check(1.0): return 0
                    continue
                elif status == "mate":
                    # 引擎确认将死（应用无关信号），直接开下一局
                    print("[终局] 自动开下一局")
                    if AUTO_NEXT and click_next_game():
                        expect, state, weird, refusals, stales = None, "boot", 0, 0, 0
                        SIDE_LOCK = None      # 新局重新判定先后手
                        SIDE_FLIPS[0] = 0
                        SIDE_GUESS[0] = False
                        MISS_STAT["move"], MISS_STAT["count"] = None, 0
                        last_sig, stable, boot_wait, prev_ok = None, 0, None, None
                        prev_frame = None
                        clear_ponder()
                        if sleep_check(3): return 0
                        continue
                    print("已停止（点击再来一局失败）")
                    return 1
                elif status == "end":
                    # 画面已是结算页（落子前复核发现的）：这不是"落子失败"，
                    # 绝不能计入 refusals（攒够 5 次会直接停机）。重置状态
                    # 回到主循环，由结算检测去点「再来一局」。
                    last_sig, stable, stales = None, 0, 0
                    prev_frame = None
                    if sleep_check(POLL): return 0
                    continue
                elif status == "stale":
                    # 局面在分析期间变了，下一轮用最新局面重新分析，不计失败。
                    # 但"复核一直不过"必须封顶：它不计入 refusals，会变成引擎
                    # 反复思考、永远不落子的死循环（换 App 重新标定后最容易
                    # 撞见，界面上看起来就是卡在"思考"里不动）。
                    last_sig = None
                    stable = 0
                    stales += 1
                    if stales == 3:
                        prev_frame = None      # 丢掉复用帧，强制全量重识别
                        print("\n[诊断] 连续 3 次落子前复核不通过，"
                              "已强制重新识别棋盘", flush=True)
                    if stales >= 8:
                        print("\n连续 8 次落子前复核不通过：识别结果与画面不符\n"
                              "常见原因：① 换 App 后没点『重新标定』\n"
                              "          ② 标定用的不是一步未走的新局面\n"
                              "          ③ 当前不在对局界面。已停止。")
                        return 1
                else:
                    if FORCE_RESNAP[0]:
                        # 引擎拒局几乎都是喂进了非法局面（识别噪声/动画残影），
                        # 不是真的落子失败。丢掉复用帧强制全量重识别，且不计数
                        # ——算进 refusals 攒够 5 次会误停机（实战拒局 5 次）。
                        FORCE_RESNAP[0] = False
                        prev_frame = None
                        last_sig, stable = None, 0
                        state = "boot"
                        if sleep_check(POLL): return 0
                        continue
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

    # 阶段判定 -> 残局要换策略：加时 + 优先找杀
    ph, ph_txt = advice.phase(board)
    endgame = (ph == "end")
    think_ms = advice.endgame_think_ms(board, THINK_MS)
    print(f"局面 {ph_txt}"
          + (f" → 残局模式，思考加时到 {think_ms}ms" if think_ms != THINK_MS else ""))

    bm, res = None, None

    # 残局第一优先：找强制杀。常规"当前最优"搜索在残局常常给出看似不错、
    # 却赢不下来的着法（兑子是它的最爱），有杀时按杀走才是破解之道。
    if endgame and not dry:
        got = advice.find_forced_mate(ENGINE, fen, depth=MATE_SEARCH_DEPTH)
        if got:
            bm, mate_in, res = got
            print(f"★ 残局发现 {mate_in} 步强制杀：{bm}")

    # 若上一轮后台预搜索命中，结果已经算好了，直接取用——省掉整段搜索时间
    if bm is None and PENDING[0] is not None:
        pend_bm, pend_res, pend_fen = PENDING[0]
        PENDING[0] = None
        if pend_fen == fen0:
            bm, res = pend_bm, pend_res
            print(f"[ponder] 沿用后台搜索结果，跳过 {think_ms}ms 搜索")
        else:
            print("[ponder] 局面与后台搜索不符，作废改走正常搜索")

    if bm is None:
        bm, res = ENGINE.analyse(fen, movetime=think_ms)
    # 引擎拒局的兜底：走子方写反时引擎会**静默不出着法**（既不报错也不返回），
    # 于是每次白等超时上限（约 8 秒）。这里翻过来再试一次——能出着法就说明
    # 执子方判反了，顺势修正，别让整局卡死在这一步。
    if bm is None and not dry and ENGINE is not None:
        mine_ch = "w" if MY_SIDE == "red" else "b"
        alt_ch = "b" if mine_ch == "w" else "w"
        try:
            bm_alt, res_alt = ENGINE.analyse(fen0 + f" {alt_ch} - - 0 1",
                                             movetime=think_ms, retries=0,
                                             timeout=2.0)
        except Exception:
            bm_alt = None
        if bm_alt:
            other = "black" if MY_SIDE == "red" else "red"
            # 改判只在**开局完整局面**下才可信。中局的拒局绝大多数不是走子方
            # 写反，而是局面特殊——最典型的就是我方已被将死（无合法着法，引擎
            # 静默不响应），此时翻转成对方走当然能出着，据此把执子方改成对方
            # 纯属误导（实战日志：21:24:10 在已被将死的残局上把正确的黑改判成
            # 红）。真·开局期执子方未定案时，这条才救得了场。
            n_piece = sum(1 for v in board.values()
                          if (v[0] if isinstance(v, tuple) else v))
            if SIDE_LOCK is None and n_piece >= 28:
                # 执子方还没定案（开局）：翻转就能出着 = 之前判反了，顺势改判。
                bm, res = bm_alt, res_alt
                print(f"[走子方] 引擎拒局，翻转成 {alt_ch} 后出着 -> 执子方应为 "
                      f"{other}（原判 {MY_SIDE}）")
                apply_side(other, "引擎拒局翻转后出着", lock=True)
            else:
                # 执子方已定案、或非开局局面：拒局几乎都是将死/困毙、引擎瞬时
                # 问题或识别噪声，不是走子方写反（实测照旧改判会把正确的红翻成
                # 黑，白白乱一手、8 秒后才被朝向先验纠正）。不采用翻转出来的
                # 着法（那是对方视角），本轮放弃，下轮重新识别再算。
                print(f"[走子方] 引擎拒局（{n_piece} 子"
                      f"{'，执子方已锁定' if SIDE_LOCK else '，非开局局面'}"
                      f"）：多半是将死/困毙或识别噪声，不改判，本轮放弃")
                FORCE_RESNAP[0] = True
                bm = None
    # 拒局时 res/info 可能残缺，统一兜成 dict，别让 .get 抛异常
    info = (res or {}).get("info") or {}

    # 胜率：走子方是我方，所以不需翻转。这行同时给界面和日志用，
    # 界面认 "胜率:" 前缀。真正的深搜胜率在落子后的等待期另算。
    report_winrate(info, board, flip=False, tag=f"{ph_txt} ")
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
    # 结算页上棋盘会被缩小上移，但**冻结标定是固定坐标、不校验内容**，
    # 于是定位照样"成功"、复核会假通过，bot 就在结算页上一直点（实测
    # 连点 3 次未生效、白白耗掉 70 秒才等到结算检测轮到）。这里按结算页
    # 的收紧判据再挡一道，命中就交回主循环走"再来一局"。
    if green_button(img_f, min_area=400, max_area=2600, min_y_ratio=0.82):
        print("落子前复核：画面已是结算页，放弃本轮点击")
        return "end", None
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

    # 这一手上一轮没走成：先把 UI 的选中/落子状态机拉回中性，再重走。
    # 少了这一步，第二次同样的两次点击会把上一次实际已到位的子重新选中、
    # 走回原地（见 MISS_STAT 的注释）——日志里出现过同一手连点 8 次。
    if MISS_STAT["move"] == bm and MISS_STAT["count"] >= 1:
        if neutral_tap(loc):
            print("[UI复位] 上一次同一手未生效，先点空白取消可能的选中残留")

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
            CONFIRMED[0] = True
            MISS_STAT["move"], MISS_STAT["count"] = None, 0
            break
        if time.time() >= deadline:
            # 判定标准：起点变空 = 棋子确实离开了 = 落子生效。
            # 动画中间帧会让"起点/终点都显得空"，此时不能判失败；
            # 真正失败的特征是棋子仍留在起点。
            if s2p is not None and d2p is None:
                # 以前这里只打一行字然后照样返回 ok，上层于是把"理论上的
                # 落子后局面"当成实际局面，下一轮又重放同一手，形成死循环。
                # 现在明确返回 miss，交给上层决定是否先静置 / 换边。
                print(f"!! 落子可能未生效（棋子仍在起点 {s2p}）")
                CONFIRMED[0] = False
                MISS_STAT["move"] = bm
                MISS_STAT["count"] = MISS_STAT.get("count", 0) + 1
                return "miss", None
            # 到 deadline 还确认不了，说明不是普通动画延迟。以前这里照样
            # break 出去当成功返回，上层于是把"理论落子后局面"当真，下一轮
            # 照着这个假期望再发同一手——实测每 3 秒一次、无限循环。
            # 连续 2 次确认不了就按未生效处理，交给上层换边/静置。
            MISS_STAT["move"] = bm
            MISS_STAT["count"] = MISS_STAT.get("count", 0) + 1
            if MISS_STAT["count"] >= 2:
                print(f"!! 连续 {MISS_STAT['count']} 次无法确认落子，按未生效处理")
                CONFIRMED[0] = False
                return "miss", None
            print("落子确认：动画未完全落定，交下一轮校验")
            CONFIRMED[0] = False
            break
    # 绝杀检测（与应用无关）：走完后看对方是否已无路可走
    # 必须补全走子方字段，否则引擎拿到残缺 FEN 会静默保持旧局面
    side_after = "b" if MY_SIDE == "red" else "w"
    if not dry and opponent_mated(f"{fen_expect} {side_after} - - 0 1"):
        print("★ 绝杀！对方已无路可走，本局获胜")
        return "mate", fen_expect

    # 落子后、ponder 之前做一次深度评估。必须卡在这个位置：
    #   - 在绝杀检测之后：绝杀已定胜负，没必要再评估；
    #   - 在 ponder 之前：analyse 会中止后台搜索，放后面就把 ponder 冲掉了。
    # 此刻轮到对手走，引擎给出的是对手视角分数，内部会翻转成我方胜率。
    if not dry:
        b_after = dict(board)
        b_after.pop(src, None)
        b_after[dst] = board.get(src)
        deep_eval_on_wait(fen_expect, b_after, opp_to_move=True, dry=dry)

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
    # 被 WinUI 3 界面当子进程调用时，stdout 是管道：Python 会切成块缓冲，
    # 界面上就表现为"日志半天不刷一行"。强制行缓冲并统一 UTF-8。
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(line_buffering=True, encoding="utf-8",
                           errors="replace")
        except Exception:
            pass
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已退出")

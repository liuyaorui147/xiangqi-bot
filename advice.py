"""局面评估换算与残局应对。

三块东西，都被主循环和界面用到：

1) win_percent(info)  —— 把引擎输出换成人看得懂的胜率
2) phase(board)       —— 判定开局/中局/残局
3) 残局对策           —— 加长思考时间、优先找强制杀

分数视角说明（容易搞反）
------------------------
UCI 的 score 一律是**当前局面的走子方**视角。main 拼 FEN 时走子方写的是我方
（`to_fen(...) + " w"` 表示我方执红且该我走），所以这里的 info 天然是我方视角，
不用再做 side 换算。一旦将来出现"替对手分析"的场景，必须先把分数取反。
"""

from __future__ import annotations

WIN_DIVISOR = 400.0     # Elo 式换算的分母：越大越保守（400 时 1 兵优势≈60%）

# 子力价值（与主流象棋引擎同量级，单位=分）。只用于两件事：
#   1) 估算局面子力差，给评估分画一条"物理上限"；
#   2) 残局/大优时的辅助判断。
# 兵卒按未过河计（过河兵实际更值钱，所以下面给评估留了宽裕的余量）。
PIECE_VALUE = {
    "R": 900, "N": 400, "C": 450, "B": 200, "A": 200, "P": 100, "K": 0,
    "r": 900, "n": 400, "c": 450, "b": 200, "a": 200, "p": 100, "k": 0,
}

# 评估分相对"子力差"允许浮动的余量。棋子的位置价值、兵过河加成、尤其是
# **攻势**（有杀但还没算清 mate 时）都会让评估远超纯子力差，所以这条线必须
# 留得宽：实测真实大优局面能到 +2150，卡在 1200 会把真局面误杀。
# 见过真实事故：黑车被误认成黑卒 → 引擎算出 +6141（≈多六个车）→ 胜率 100%，
# 取 2600 能把这类荒谬值挡住，同时不误伤真实的攻势型优势。
EVAL_SLACK = 2600
EVAL_MAT_RATIO = 1.5    # 子力差越大，允许的位置价值浮动也越大

# 超限豁免：cp 高出子力上限时，若三条同时成立，判定是"真杀势"而非错帧。
# 事故由来：子力差 0 的中局，引擎 14 层算出 cp +6044，被纯子力上限当成错帧
# 拦掉，胜率就一直卡在旧值上。离线复核同一局面（depth 20）：cp 3137、
# wdl 1000 0 0 —— 确是绝杀势，bot 走的 b3b6 也和离线 bestmove 一致。
# 子力相当但已成杀势时 cp 到 3000~9999 是常态，线性上限必然误杀。
# 三条判据缺一不可，目的都是把"误放行"压到最低：
#   1. 搜得够深（浅层的大分值多半是搜索噪声）；
#   2. 引擎自己的 wdl 胜负模型也认为一边倒；
#   3. 子力差没有凭空跳变（错帧的典型特征是凭空多/少一个大子）。
EVAL_TRUST_DEPTH = 12
EVAL_TRUST_WDL = 0.95
EVAL_DIFF_JUMP = 400    # 子力差跳变阈值，约一个马/炮
_LAST_DIFF = [None]     # 上一次判定时的子力差，用于识别"凭空多子"
_LAST_N = [None]        # 上一次判定时的棋子总数，用于识别"真吃子"


def forget_diff():
    """宣告"子力差接下来跳变是合法的"，跳过凭空多子这条判据。

    吃子会让子力差一步跳 400+（吃车 900+），跟"识别错帧凭空多一个大子"
    的指纹一模一样。深搜恰好发生在我方落子之后，于是每一手吃子后的极端
    评估都会被这条判据误杀——实战里 +9999（真杀势）被连拦三次就是这么
    来的。所以只要局面变化被确认是**一步棋**（我方落子生效 / 对手应了
    一手），就作废上一帧的子力差，让这一帧的跳变不被当成错帧。
    """
    _LAST_DIFF[0] = None

# 胜率跳变钳制：单帧最多变化的幅度。识别抖动会让相邻两帧的局面完全不同，
# 若直接采信就会出现 1.2% → 100% 这种荒唐跳变。
WR_MAX_STEP = 0.20
WR_EMA_ALPHA = 0.55     # 新值权重，越大越灵敏

# 残局里用来判定"还有多少攻击力量"的车马炮（满编双方共 12）。
# 注意别把兵卒放进来：兵要过河才谈得上攻击力，算进来会让这条阈值失效
# ——初始局面两边就是 10 个兵，阈值会被它们撑得一直判不进残局。
BIG = "RNC rnc"

PHASE_CN = {"open": "开局", "mid": "中局", "end": "残局"}

# 残局判定阈值：任一条成立就算残局
END_BIG_MAX = 6         # 双方车马炮总数 ≤ 6（满编是 12）
END_PIECE_MAX = 14      # 双方总子数 ≤ 14（满编 32）

# 残局模式给引擎的时间：基准的倍数，以及封顶值
END_THINK_SCALE = 2.5
END_THINK_MAX = 3000

MATE_TIMEOUT = 1.0        # 找杀的等待上限
                          # 残局子少，正常情况毫秒级就能出结果（实测 8 层 0.00s）。
                          # 给这么紧是因为引擎遇到不接受的局面会**既不出着法
                          # 也不报错**，一直干等；这种情况下每步白卡几秒太亏，
                          # 宁可判定"没有杀"走常规搜索。
MATE_MAX_DEPTH = 12       # 深度上限，防止这里的开销失控


def _val(v):
    """board 的值可能是 'R'，也可能是 ('R', 0.98) 这种带置信度的元组。"""
    return v[0] if isinstance(v, tuple) else v


def board_pieces(board):
    """返回 (总子数, 车马炮总数)。"""
    n = big = 0
    for v in board.values():
        p = _val(v)
        if not p:
            continue
        n += 1
        if p in BIG:
            big += 1
    return n, big


def phase(board):
    """判断局面阶段，返回 (代码, 说明文字)。

    开局/中局/残局没有硬标准，这里用两条量化口径（任一条成立即判残局）：
      - 车马炮（攻击子）总数 ≤ 6  →  双方已经换掉大半武器
      - 总子数 ≤ 14              →  棋盘上没剩多少东西
    反过来，子数 ≥ 26 一定是开局。
    """
    n, big = board_pieces(board)
    if big <= END_BIG_MAX or n <= END_PIECE_MAX:
        return "end", f"残局（{n}子/{big}大子）"
    # 子多未必是开局：兑掉两个大子但保留士象兵时，仍可能停在开局谱着里，
    # 所以开局要"子多 且 武器还在"两条同时满足。
    if n >= 26 and big >= 10:
        return "open", f"开局（{n}子）"
    return "mid", f"中局（{n}子/{big}大子）"


def is_endgame(board):
    return phase(board)[0] == "end"


def endgame_think_ms(board, base_ms):
    """残局该给引擎多少时间。

    残局的胜负往往取决于一两步的精确算路（能不能抓住边卒、能不能形成杀），
    中局的 350ms 在这里不够看。但也不能无上限：模拟器里对手也在走，思考太久
    容易被判超时。封顶 3s。
    """
    if not is_endgame(board):
        return base_ms
    return int(min(END_THINK_MAX, max(base_ms * END_THINK_SCALE, base_ms + 400)))


def material_balance(board, my_side="red"):
    """我方子力 − 对方子力（分）。正数=我方子力占优。

    只算静态子力，不含位置价值。用途是给评估分画一条物理上限：
    评估可以因为位置/攻势偏离子力差，但偏离幅度有天花板。
    """
    mine_is_red = (my_side == "red")
    diff = 0
    for v in board.values():
        p = _val(v)
        if not p or p in "Kk":
            continue
        val = PIECE_VALUE.get(p, 0)
        diff += val if (p.isupper() == mine_is_red) else -val
    return diff


def eval_credible(board, info, my_side="red"):
    """评估分与局面子力是否自洽。返回 (可信?, 说明)。

    这是胜率准确性的守门员。识别错帧会造出物理上不可能的局面
    （比如黑车被认成黑卒，等于凭空少一个车），引擎在这种局面上的
    评估会离谱到 +6141 这种量级——而 WDL 模型照单全收，直接报 100%。
    子力差给了评估一个硬天花板，超过就判不可信，胜率保持上次的值。
    """
    if not info:
        return False, "无引擎输出"
    st = info.get("scoretype")
    if st == "mate":
        return True, ""          # 杀棋是确定结论，不受子力约束
    if st != "cp":
        return False, f"未知分数类型 {st!r}"
    try:
        cp = int(info.get("score"))
    except (TypeError, ValueError):
        return False, "无法解析分数"

    diff = material_balance(board, my_side)
    prev = _LAST_DIFF[0]
    # 棋子数变少 = 盘面上真少了一个子（有人吃子），这时子力差跳 400+ 是物理
    # 必然，不能当成"错帧凭空多子"。只凭主循环在落子成功时调 forget_diff()
    # 兜不住：动画未落定那一轮返回的是 miss，可子已经吃掉了，紧随其后的深搜
    # 就会被这条判据误杀（实战：-9999 真杀势、depth21、wdl 一边倒，只因跳变
    # 450 被拦）。所以这里自己看子数变化，比依赖调用时机可靠。
    n_now = sum(1 for v in board.values() if _val(v))
    if _LAST_N[0] is not None and n_now < _LAST_N[0]:
        prev = None
    _LAST_N[0] = n_now
    _LAST_DIFF[0] = diff
    # 位置价值/攻势能让评估偏离子力差，但偏离幅度有限。
    # 子力差大时允许更大的浮动（EX：多两个车的局面评估 2500 很正常）。
    limit = abs(diff) * EVAL_MAT_RATIO + EVAL_SLACK
    if abs(cp) <= limit:
        return True, ""
    if _extreme_but_trusted(info, diff, prev):
        return True, ""
    # 日志带上三条判据的实测值：下次再出现"真杀势被误杀"，一眼就能看出
    # 卡在深度、wdl 还是子力差跳变上，不用再离线复现。
    wdl = info.get("wdl")
    jump = "" if prev is None else f"，子力差跳变 {abs(diff - prev):.0f}"
    return False, (f"评估 {cp:+d} 超出子力上限 ±{limit:.0f}"
                   f"（子力差 {diff:+d}{jump}；depth={info.get('depth')}，"
                   f"wdl={wdl}），疑似识别错帧，忽略本帧胜率")


def _extreme_but_trusted(info, diff, prev_diff):
    """cp 超出子力上限，但看起来是真杀势而不是错帧？

    只对"深搜 + 引擎自报一边倒 + 子力差没凭空跳变"网开一面。三者都满足
    时，误判成错帧的代价（胜率长期卡在旧值）远大于误放行的代价。
    """
    try:
        depth = int(info.get("depth") or 0)
    except (TypeError, ValueError):
        depth = 0
    if depth < EVAL_TRUST_DEPTH:
        return False
    wdl = info.get("wdl")
    if not (isinstance(wdl, (list, tuple)) and len(wdl) == 3):
        return False
    try:
        w, _d, l = (float(x) for x in wdl)
    except (TypeError, ValueError):
        return False
    if max(w, l) < EVAL_TRUST_WDL * 1000:
        return False
    # 认错一个子（车→卒）会让子力差凭空跳一个大子，这是错帧的指纹
    if prev_diff is not None and abs(diff - prev_diff) > EVAL_DIFF_JUMP:
        return False
    return True


def win_percent(info, flip=False, board=None, my_side="red", strict=True):
    """把引擎评价换成我方胜率（0~1）。换不出来返回 None。

    flip=True 表示 info 是**对手走子方**视角（我方刚落子、轮到对手时的
    局面就是这种情况），必须把胜负反过来才是"我方胜率"。这里最容易错：
    UCI 的 score/wdl 一律是当前走子方视角，漏掉翻转就会在我方大优时报
    出接近 0 的胜率。

    strict=True 且传了 board 时，先过 eval_credible 子力自洽检查，
    不自洽返回 None（由调用方保持上一个可信值）。

    优先级：
      1. wdl (win/draw/loss 千分数) —— 引擎自己的胜负模型，最贴合实际；
      2. mate N —— N>0 是我们将杀对方，N<0 是被将杀；
      3. cp —— Elo 式换算：win% = 1/(1+10^(-cp/400))。
    """
    if not info:
        return None

    if strict and board:
        ok, _why = eval_credible(board, info, my_side)
        if not ok:
            return None

    wdl = info.get("wdl")
    if isinstance(wdl, (list, tuple)) and len(wdl) == 3:
        try:
            w, d, l = (float(x) for x in wdl)
        except (TypeError, ValueError):
            pass
        else:
            if flip:
                w, l = l, w
            return max(0.0, min(1.0, (w + d / 2.0) / 1000.0))

    st, sc = info.get("scoretype"), info.get("score")
    try:
        n = int(sc)
    except (TypeError, ValueError):
        return None
    if flip:
        n = -n

    if st == "mate":
        if n > 0:
            return 1.0
        if n < 0:
            return 0.0
        return None
    if st == "cp":
        return 1.0 / (1.0 + 10 ** (-n / WIN_DIVISOR))
    return None


class WinRateSmoother:
    """胜率平滑：抗识别抖动，同时保留真实的趋势变化。

    单帧直接用会出现 1.2% → 100% 这种跳变（真凶是识别错帧，但即便
    识别正常，浅搜的相邻帧分数也会抖几个百分点）。这里做两件事：
      1) EMA 平滑，让曲线连续；
      2) 跳变钳制，单帧最多挪 WR_MAX_STEP，防止单次错帧把显示打飞。

    局面存疑（win_percent 返回 None）时不更新，保持上一个可信值——
    宁可显示"旧一点但真"，也不要显示"新但假"。
    """

    def __init__(self, alpha=WR_EMA_ALPHA, max_step=WR_MAX_STEP):
        self.alpha = alpha
        self.max_step = max_step
        self.value = None
        self.stale = 0          # 连续多少帧没能更新

    def update(self, raw):
        """喂入新算出的胜率（None 表示本帧不可信）。返回平滑后的显示值。"""
        if raw is None:
            self.stale += 1
            return self.value
        self.stale = 0
        if self.value is None:
            self.value = raw
            return self.value
        target = self.value + (raw - self.value) * self.alpha
        # 钳制：单次最多移动 max_step
        delta = target - self.value
        if abs(delta) > self.max_step:
            target = self.value + (self.max_step if delta > 0 else -self.max_step)
        self.value = max(0.0, min(1.0, target))
        return self.value

    def reset(self):
        self.value = None
        self.stale = 0


def eval_text(info):
    """把 score 显示成人话：'+185' / 'M3' / '-95'。"""
    st, sc = info.get("scoretype"), info.get("score")
    if st == "mate":
        try:
            return f"M{sc}"
        except Exception:
            return "M?"
    if st == "cp":
        try:
            v = int(sc)
            return f"{'+' if v >= 0 else ''}{v}"
        except (TypeError, ValueError):
            pass
    return "?"


def find_forced_mate(engine, fen, depth=8):
    """找强制杀。返回 (bestmove, 几步杀, res)，没找到返回 None。

    res 要带回来：里面有 score/depth/wdl，调用方直接就能算胜率，不必再搜一次。

    中局用 movetime 搜索图的是"当前最优"，搜到第几层随缘；残局要的是
    "有没有明确的连杀"，所以用固定深度——有杀的话这个深度足够看清。

    注意不能用 `go mate N`：那是"限定搜索深度找杀"，找到的有可能是假杀；
    这里用的是常规搜索 + 检查 score 是不是 mate，引擎说 mate 就说明它在
    搜索深度内算出了必胜路线（代价：没有杀时这次搜索是纯开销）。
    """
    depth = min(depth, MATE_MAX_DEPTH)
    try:
        bm, res = engine.analyse(fen, depth=depth, retries=0, timeout=MATE_TIMEOUT)
    except Exception as e:
        print(f"  [残局] 找杀失败: {type(e).__name__}: {e}")
        return None
    # 超时意味着没搜完，这时候的 score 是中间值，拿它当"有杀"会误事。
    # 更不能让它走下去触发 engine 的重试/重启：那会重新加载 48MB 的 NNUE，
    # 一步卡上好几秒。这里直接判无杀，交给正常的加时搜索。
    if res.get("timeout"):
        print("  [残局] 找杀无响应（局面可能不被引擎接受），改用常规搜索")
        return None
    info = res.get("info") or {}
    if not bm or bm == "(none)":
        return None
    if info.get("scoretype") != "mate":
        return None
    try:
        n = int(info.get("score"))
    except (TypeError, ValueError):
        return None
    if n <= 0:      # 负数是我们将被杀，不是"找到了杀"
        return None
    return bm, n, res

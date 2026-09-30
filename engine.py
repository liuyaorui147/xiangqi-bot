"""Pikafish UCI 引擎封装。

核心设计：用 Ponder（后台预搜索）把对手的思考时间变成我方的搜索时间。

Ponder 是 UCI 的标准机制：我方落子后，引擎不闲着，而是按自己预测的最佳
应手继续往下搜；对手真下了这一步就发 ponderhit，搜索立即转成正式结果。
这是唯一一种"不牺牲棋力还能提速"的手段——命中的话省掉几乎整个搜索时间，
未命中也只是回到原来的冷启动搜索，没有额外损失。

实测（Pikafish 2026-09-25，350ms 思考）：
    冷启动搜索      350 ms
    ponder 命中      1 ms   （ponder 已跑够时间）
    ponder 未命中    0 ms   停止，随后按正确局面重新搜索
且ponderhit 后总搜索时间仍被控制在 movetime 内，深度与冷启动一致，
不会因为抢跑而变浅。
"""
import atexit
import os
import queue
import subprocess
import threading
import weakref
from collections import deque

# 引擎回显只保留最近若干行。以前是普通 list 无上限 append，
# 长时挂机（一晚几千手）就是缓慢的内存泄漏。
ECHO_MAX = 500

# 活着的 Engine（弱引用）。用于解释器退出时兜底杀进程。
# 必须是弱引用集合：用普通 list/set 会让 Engine 永远不被回收，
# __del__ 就不执行了，反而制造更难查的泄漏。
_LIVE = weakref.WeakSet()


def _reader_loop(ref, q, stdout):
    """读取某一代进程的输出（模块级函数，只持 Engine 的弱引用）。

    两条关键约束：
    1. 队列由参数捕获而不是读 self._q：restart 时旧读线程还没退出，若它
       继续往 self 上取队列，就会把已死进程的残留输出写进新进程的队列，
       导致 wait_for("bestmove") 匹配到上一局的 bestmove。
    2. **不能强引用 Engine**：读线程若持有 self，Engine 就永远达不到引用
       计数 0，__del__ 不执行，pikafish 进程会一个接一个残留在后台（每
       次 `ENGINE = Engine(...)` 泄漏一个），几个进程互相抢 CPU 导致卡顿。
       改成 weakref 后，对象一没人用就能被回收并 kill 掉子进程。

    ponder 期间引擎每秒吐几十条 info，若全部入队会无限堆积。此处只保留
    最新一条到 _ponder_info，不进队列。
    """
    try:
        for line in stdout:
            obj = ref()
            if obj is None:      # 引擎对象已被回收，没必要再读
                break
            line = line.rstrip("\r\n")
            obj._echo.append(line)
            if obj._pondering and line.startswith("info "):
                obj._ponder_info = line
                continue
            q.put(line)
    except (ValueError, OSError):
        pass      # 进程被 kill / stdout 已关闭，属正常退出


def _kill_all_at_exit():
    """退出时兜底：杀掉所有还活着的引擎进程。

    Windows 下子进程不会随父进程自动退出，GUI 关掉后 pikafish 会变成
    孤儿进程继续吃 CPU（尤其是还在 ponder 的那个）。
    """
    for e in list(_LIVE):
        try:
            p = getattr(e, "p", None)
            if p is not None and p.poll() is None:
                p.kill()
        except Exception:
            pass


atexit.register(_kill_all_at_exit)


class EngineTimeout(Exception):
    """等待引擎回应超时。

    以前 wait_for 超时会往结果里塞一个 "<等待 xxx 超时>" 占位符再静默返回，
    调用方把它当正常输出解析，超时就被吞了。现在超时是显式信号。
    """


def parse_info(line):
    """解析 UCI info 行。正确处理 'score cp N' / 'score mate N' 的三段结构。"""
    t = line.split()
    d, i = {}, 1
    while i < len(t):
        k = t[i]
        if k == "score":
            d["scoretype"] = t[i + 1] if i + 1 < len(t) else ""
            d["score"] = t[i + 2] if i + 2 < len(t) else ""
            i += 3
            continue
        if k == "wdl":
            # "wdl 123 456 421" 是三个数（胜/和/负的千分数），通用解析只会
            # 取到第一个。这里要一次性收三个，否则胜率算出来都是赢。
            try:
                d["wdl"] = tuple(int(x) for x in t[i + 1:i + 4])
            except (ValueError, IndexError):
                pass
            i += 4
            continue
        if k == "pv":
            d["pv"] = " ".join(t[i + 1:])
            break
        if i + 1 < len(t):
            d[k] = t[i + 1]
        i += 2
    # aspiration 窗口搜索失败后重搜会给出 upperbound/lowerbound：
    # 那不是真实分数，只是"真实值不超过/不低于它"。拿它算胜率会系统性偏
    # 高或偏低，必须标记出来让调用方跳过。
    if "upperbound" in t:
        d["bound"] = "upper"
    elif "lowerbound" in t:
        d["bound"] = "lower"
    return d


def _ponder_from_pv(info_line):
    """从 info 行取预测的对手应手。

    pv 是 "我方着法 对手应手 我方再应 ..."，所以对手应手是**第二着**，
    不是第一着——取错会让 ponder 在我方刚走过的局面上重复搜索。
    """
    if not info_line or " pv " not in info_line:
        return None
    rest = info_line.split(" pv ", 1)[1].split()
    return rest[1] if len(rest) > 1 else None


class Engine:
    def __init__(self, exe="engine/pikafish.exe", nnue="engine/pikafish.nnue", timeout=30.0,
                 threads=None, hash_mb=256, ponder=True, move_overhead=10):
        """threads=None 时按 CPU 核数自动取一个合理值（留 1/4 给模拟器和系统）。

        ponder=True 时开启后台预搜索（默认开）。引擎不支持 Ponder 选项时
        会自动降级为普通搜索，调用方无需分支处理。
        """
        self.exe = os.path.abspath(exe)
        self.nnue_path = os.path.abspath(nnue) if nnue else None
        self.timeout = timeout
        self._echo = deque(maxlen=ECHO_MAX)
        self._closed = False
        if threads is None:
            cores = os.cpu_count() or 4
            # 封顶 8 线程：再多收益递减，且模拟器本身还要吃 CPU
            threads = max(1, min(8, cores - max(2, cores // 4)))
        self.threads, self.hash_mb = threads, hash_mb
        self.move_overhead = move_overhead

        # --- ponder 状态 ---
        self._pondering = False      # 后台搜索进行中（由 reader 线程读取）
        self._ponder_move = None     # 正在预搜索的对手应手
        self._ponder_info = ""       # 后台搜索的最新一条 info（不入队，避免堆积）
        self._ponder_movetime = 300  # 后台搜索的 movetime，用于推算命中超时
        self._ponder_timer = None    # 后台搜索的到点停止定时器
        self._want_ponder = bool(ponder)
        self.ponder_enabled = False  # 握手后按引擎实际支持的选项确定
        self.ponder_hits = 0
        self.ponder_misses = 0

        self._spawn()
        self._handshake()

    # ------------------------------------------------------------------ 进程

    def _reader(self, q, stdout):
        """兼容入口：真正的工作在模块级 _reader_loop 里。"""
        _reader_loop(weakref.ref(self), q, stdout)

    def _spawn(self):
        flags = 0x08000000 if os.name == "nt" else 0   # 不创建控制台窗口
        kw = dict(
            cwd=os.path.dirname(self.exe),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            creationflags=flags,
        )
        try:
            self.p = subprocess.Popen([self.exe], **kw)
        except OSError:
            # 引擎崩溃后立刻重启偶发失败（句柄还没释放干净），退一步再试
            import time as _t
            _t.sleep(0.5)
            self.p = subprocess.Popen([self.exe], **kw)
        self._q = queue.Queue()
        # 新进程必须清掉 ponder 状态：旧进程的后台搜索已经随进程消失
        self._pondering = False
        self._ponder_move = None
        self._ponder_info = ""
        self._ponder_movetime = 300
        # 传 weakref 而不是 self：读线程不能成为 Engine 存活的理由
        self._t = threading.Thread(target=_reader_loop,
                                   args=(weakref.ref(self), self._q, self.p.stdout),
                                   daemon=True)
        self._t.start()
        _LIVE.add(self)      # 退出兜底用（弱引用，不影响回收）

    def _teardown(self):
        """停掉当前进程与它的读线程。restart / quit 前必须调用。"""
        self._pondering = False
        t = getattr(self, "_t", None)
        p = getattr(self, "p", None)
        if p is not None:
            try:
                if p.poll() is None:
                    p.kill()
            except Exception:
                pass
        if t is not None and t.is_alive():
            t.join(timeout=2.0)     # 等读线程把旧 stdout 读完退出，避免残留写队列
        # 顺序要紧：先 kill + join，再关管道。反过来关会让读线程读到一个
        # 已关闭的文件，在解释器退出阶段报 "Exception ignored ... EINVAL"。
        if p is not None:
            for f in (getattr(p, "stdin", None), getattr(p, "stdout", None)):
                try:
                    if f is not None:
                        f.close()
                except Exception:
                    pass

    def _handshake(self):
        self.send("uci")
        lines, timed_out = self.wait_for("uciok")
        if timed_out:
            raise EngineTimeout(f"引擎未回应 uciok（{self.timeout}s）: {self.exe}")
        self.id_name = next((l.split("id name ", 1)[1] for l in lines
                             if l.startswith("id name")), "?")
        self.options = {}
        for l in lines:
            if l.startswith("option name"):
                body = l[len("option name"):].strip()
                self.options[body.split(" type ")[0].strip()] = body
        if self.nnue_path and os.path.exists(self.nnue_path):
            key = next((k for k in self.options if "evalfile" in k.lower()), None)
            if key:
                self.send(f"setoption name {key} value {os.path.abspath(self.nnue_path)}")

        # Ponder 必须是 check 型选项才敢开；引擎不支持就静默降级
        self.ponder_enabled = self._want_ponder and "Ponder" in self.options
        opts = [("Threads", self.threads), ("Hash", self.hash_mb),
                ("Ponder", "true" if self.ponder_enabled else "false")]
        # 开 WDL（胜/和/负千分数）：这是引擎自己的胜负模型，比拿 cp 反推
        # 胜率准得多。引擎不支持时下面的 in options 判断会自动跳过。
        opts.append(("UCI_ShowWDL", "true"))
        # 本地管道延迟不到 1ms，默认 30ms 的 move overhead 是给网络对弈留的
        if "Move Overhead" in self.options:
            opts.append(("Move Overhead", self.move_overhead))
        for key, val in opts:
            if key in self.options:
                self.send(f"setoption name {key} value {val}")

        self.send("isready")
        _, timed_out = self.wait_for("readyok")
        if timed_out:
            raise EngineTimeout(f"引擎未回应 readyok（{self.timeout}s）: {self.exe}")

    def restart(self):
        """引擎崩溃后自愈：非法局面可能让进程直接退出。

        必须先把旧进程和它的读线程都收干净再起新的，否则旧线程的残留
        输出会串到新进程的队列里。
        """
        self._teardown()
        self._spawn()
        self._handshake()

    # ------------------------------------------------------------------ 基础 IO

    def send(self, cmd):
        if self._closed:
            raise RuntimeError("引擎已关闭，不能再发送命令")
        try:
            self.p.stdin.write(cmd + "\n")
            self.p.stdin.flush()
        except (OSError, ValueError, AttributeError):
            self.restart()
            self.p.stdin.write(cmd + "\n")
            self.p.stdin.flush()

    def wait_for(self, token, timeout=None):
        """收集直到包含 token 的行（含该行）为止的所有输出。

        返回 (lines, timed_out)。超时不再往结果里塞占位符——那会让调用方
        把超时当成一次正常（但内容古怪）的引擎回应。
        """
        timeout = timeout or self.timeout
        out = []
        try:
            while True:
                line = self._q.get(timeout=timeout)
                out.append(line)
                if token in line:
                    return out, False
        except queue.Empty:
            return out, True

    def set_position(self, fen="startpos", moves=None):
        """设置局面。

        fen 可以是 "startpos"，也可以是**不带前缀的** FEN 串——本方法自动补
        "fen " 前缀。以前要求调用方自己拼 "fen ..."（analyse 里就是这么干的），
        漏拼就生成非法 UCI 命令、引擎静默拒绝。这里同时容忍已带前缀的写法。
        """
        f = (fen or "startpos").strip()
        if f == "startpos" or f.startswith("fen "):
            arg = f
        else:
            arg = f"fen {f}"
        cmd = f"position {arg}"
        if moves:
            cmd += " moves " + " ".join(moves)
        self.send(cmd)

    def _collect(self, out, info_seed=""):
        """从一批输出行里取出 bestmove、最后一条 info，以及该 info 的原文。

        原文要单独留着：bestmove 行里没有 pv，取预测应手只能从 info 行取。

        info 取"最后一条不带 bound 的行"：带 upperbound/lowerbound 的是
        窗口重搜的边界值，不是真实分数。只有全部带 bound 时才退回最后一条
        （总比没有强），并保留 bound 标记让调用方知道这个分数不可靠。
        """
        bm, info, info_line = None, {}, ""
        last_any = ({}, "")
        for l in out:
            if l.startswith("bestmove"):
                bm = l.split()[1] if len(l.split()) > 1 else None
            elif l.startswith("info ") and " score " in l:
                d = parse_info(l)
                last_any = (d, l)
                if not d.get("bound"):
                    info, info_line = d, l
        if not info:
            info, info_line = last_any
        if not info and info_seed:
            info, info_line = parse_info(info_seed), info_seed
        return bm, info, info_line

    # ------------------------------------------------------------------ 搜索

    def go(self, movetime=None, depth=None, nodes=None, timeout=None):
        """返回 (bestmove, info)。info 为最后一条含 score 的 info 行解析出的字典。

        注意 "score cp 45" / "score mate 3" 是三段结构，不能按键值交错解析，
        否则后续字段（nodes / hashfull 等）会整体错位。
        """
        self._abort_ponder()      # 正式搜索前必须先停掉后台搜索
        parts = ["go"]
        if movetime is not None:
            parts += ["movetime", str(movetime)]
        if depth is not None:
            parts += ["depth", str(depth)]
        if nodes is not None:
            parts += ["nodes", str(nodes)]
        self.send(" ".join(parts))
        out, timed_out = self.wait_for("bestmove", timeout=timeout)
        bm, info, info_line = self._collect(out)
        if bm is None:
            self.restart()      # 可能喂了非法局面导致进程异常
        return bm, {"info": info, "raw": out[-1] if out else "", "timeout": timed_out,
                    "ponder": _ponder_from_pv(info_line)}

    def analyse(self, fen, movetime=300, moves=None, retries=1, depth=None,
                timeout=None):
        """带自愈重试的分析：无 bestmove 时 go() 内部已重启引擎，再试一次。

        depth 不为 None 时按固定深度搜索（忽略 movetime），用于将死检测这类
        "浅层就有确定答案"的判断——有没有合法着法，第 1 层就能看清。

        返回的 res 里带 "ponder"：引擎预测的对手应手，供 ponder_start 使用。
        """
        # 超时要留余量但不能过宽：引擎对非法局面（例如识别读到结算框、
        # 将帅照面）是**静默不响应**的，超时给 15s 就意味着一卡 15 秒、
        # 加上重试整整半分钟。取 4 倍 movetime 与 3 秒中的较大者。
        # depth 路径（将死检测）每步都跑，正常只要 1ms 级；但引擎对非法局面
        # 是**静默不响应**的，超时给 5s 就意味着每步白卡 5 秒、加重试 10 秒。
        # 中局识别抖动更容易喂进非法 FEN，这里必须收紧。
        #   movetime 路径：给足搜索时间的余量，但不能太宽 ...
        timeout = timeout if timeout is not None else (
            max(movetime / 1000.0 * 4, movetime / 1000.0 + 3.0)
            if depth is None else 1.5)
        self.set_position(fen, moves)
        bm, res = self.go(movetime=None if depth is not None else movetime,
                          depth=depth, timeout=timeout)
        for _ in range(retries):
            if bm:
                break
            # 引擎可能已重启，必须重新设置局面再搜索
            self.set_position(fen, moves)
            bm, res = self.go(movetime=None if depth is not None else movetime,
                              depth=depth, timeout=timeout)
        if not bm:
            res["ponder"] = None      # 没搜出着法时 pv 不可信，别拿去 ponder
        return bm, res

    # ------------------------------------------------------------------ Ponder

    def _cancel_ponder_timer(self):
        """撤销到点停止定时器。命中/未命中/关闭都已处理完，别再触发。"""
        t = getattr(self, "_ponder_timer", None)
        self._ponder_timer = None
        if t is not None:
            try:
                t.cancel()
            except Exception:
                pass

    def _on_ponder_timeout(self):
        """后台搜索到点：主动停止。

        `go ponder movetime N` 里的 movetime 对 Pikafish 无效 —— ponder 是
        无限搜索，实测 depth 会从 22 一路涨到 33 仍不停。于是在等待对手的
        几十秒里，引擎一直满线程吃 CPU，任务管理器看着就像卡死了。
        这里用定时器兜底：搜够一段时间就停，命中窗口（对手通常几秒内落子）
        不受影响，长时间等待则 CPU 归零。
        """
        try:
            if self._pondering and not self._closed:
                self._abort_ponder()
        except Exception:
            pass

    def _abort_ponder(self):
        """停止进行中的后台搜索。任何时候要发正式命令前都得先调一次。"""
        self._cancel_ponder_timer()
        if not self._pondering:
            return False
        self._pondering = False
        self._ponder_move = None
        try:
            self.send("stop")
            _, timed_out = self.wait_for("bestmove", timeout=5)   # 读掉残留
            if timed_out:
                # 引擎连 stop 都不理，只能重启；否则它会一直占着 stdin，
                # 后面每条命令都跟着一起卡住
                self.restart()
        except Exception:
            pass
        return True

    def ponder_start(self, fen, moves, ponder_move, movetime, max_ms=3000):
        """在"对手走了 ponder_move 之后"的局面开始后台搜索，不阻塞调用方。

        ponder_move 取自上一次 analyse 的 res["ponder"]（即 pv 第一着）。
        返回 True 表示已在后台搜索；False 表示未启用或缺少预测着法，
        调用方照常走冷启动搜索即可，无需区分。

        max_ms：后台搜索的时间上限，到点自动停止。Pikafish 的 ponder 是
        无限搜索（movetime 对它无效），不设上限就会在等待对手的整个过程中
        满线程占用 CPU。默认 3s —— 覆盖对手落子的高概率窗口，之后 CPU 归零。
        """
        if not self.ponder_enabled or self._closed or not ponder_move:
            return False
        self._abort_ponder()
        try:
            self.set_position(fen, list(moves or []) + [ponder_move])
        except RuntimeError:
            return False
        self._ponder_info = ""
        self._ponder_move = ponder_move
        self._ponder_movetime = movetime
        self._pondering = True        # 置位后再发 go，reader 才知道要丢弃 info
        self.send(f"go ponder movetime {movetime}")
        if max_ms and max_ms > 0:
            t = threading.Timer(max_ms / 1000.0, self._on_ponder_timeout)
            t.daemon = True
            self._ponder_timer = t
            t.start()
        return True

    def ponder_hit(self, timeout=None):
        """对手确实走了预测的着法：把后台搜索转成正式结果。

        返回 (bestmove, res)。引擎已在这条线上搜过一段时间，通常立即返回；
        若后台搜得还不够久，引擎会补足到 movetime 再返回——总时长与冷启动
        一致，深度不会变浅。

        拿不到 bestmove 时（典型是 ponder_start 喂了非法局面，引擎不响应）
        必须自愈并如实返回 None，让调用方退回冷启动搜索。这里若只是干等，
        一次非法局面就会让整个 bot 卡住几十秒。
        """
        if not self._pondering:
            return None, {"info": {}, "raw": "", "timeout": False,
                          "ponder": None, "hit": False}
        self._cancel_ponder_timer()   # 正在收结果，别让定时器并发发 stop
        seed = self._ponder_info
        self._pondering = False       # 之后的 info 要入队，供正常解析
        self._ponder_move = None
        if timeout is None:
            # 命中后最多再补一个 movetime，给两倍余量足够
            timeout = getattr(self, "_ponder_movetime", 300) / 1000.0 * 2 + 8
        self.send("ponderhit")
        out, timed_out = self.wait_for("bestmove", timeout=timeout)
        bm, info, info_line = self._collect(out, seed)
        if bm is None:
            # 后台搜索没结果：重启恢复可用状态，交由调用方走冷启动
            self.restart()
            return None, {"info": {}, "raw": "", "timeout": True,
                          "ponder": None, "hit": False}
        self.ponder_hits += 1
        return bm, {"info": info, "raw": out[-1] if out else "", "timeout": timed_out,
                    "ponder": _ponder_from_pv(info_line), "hit": True}

    def ponder_miss(self):
        """对手没走预测的着法：丢弃后台搜索结果。

        必须等引擎把 bestmove 吐出来并丢掉，否则它会残留在队列里，被下一次
        wait_for 当成那一次的回应——与 restart 串扰是同一类问题。
        """
        if not self._pondering:
            return False
        self.ponder_misses += 1
        self._abort_ponder()
        return True

    @property
    def pondering(self):
        return self._pondering

    # ------------------------------------------------------------------ 关闭

    def quit(self):
        """显式关闭引擎，可重复调用。先礼后兵：发 quit，超时就 kill。"""
        self._cancel_ponder_timer()   # 进程都要没了，别再触发 stop
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            p = getattr(self, "p", None)
            if p is not None and p.poll() is None:
                try:
                    if getattr(self, "_pondering", False):
                        self._pondering = False
                        p.stdin.write("stop\n")
                        p.stdin.flush()
                    p.stdin.write("quit\n")
                    p.stdin.flush()
                except Exception:
                    pass          # stdin 可能已关闭，直接走 kill
                try:
                    p.wait(timeout=3)
                except Exception:
                    pass
        finally:
            self._teardown()      # 无论如何保证进程与读线程收干净

    close = quit                  # 便于 with Engine() as e: 之外显式调用

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.quit()
        return False

    def __del__(self):
        """解释器退出阶段只做尽力而为的 kill。

        原来 __del__ 调 quit()，而 quit() 依赖 stdin 仍可写；解释器退出时
        stdin 常已关闭，清理就静默失败了。这里不再走 stdin。
        """
        try:
            p = getattr(self, "p", None)
            if p is not None and p.poll() is None:
                p.kill()
        except Exception:
            pass


if __name__ == "__main__":
    import time

    t0 = time.time()
    e = Engine()
    print(f"引擎名: {e.id_name}")
    print(f"就绪耗时: {time.time() - t0:.2f}s")
    print(f"Ponder 可用: {e.ponder_enabled}")
    print("-" * 60)

    FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"

    # 冷启动基准
    t = time.perf_counter()
    bm, res = e.analyse(FEN, movetime=350)
    cold = (time.perf_counter() - t) * 1000
    print(f"冷启动: {bm}  {cold:.0f} ms  depth={res['info'].get('depth')}")

    # ponder 命中路径。注意 moves 要带上我方刚走的 bm——漏了就变成
    # "红先局面下走黑棋"的非法局面，引擎不响应。
    pm = res.get("ponder")
    print(f"预测对手应手: {pm}")
    if pm:
        e.ponder_start(FEN, [bm], pm, movetime=350)
        time.sleep(1.0)                       # 模拟对手思考
        t = time.perf_counter()
        bm2, r2 = e.ponder_hit()
        hit = (time.perf_counter() - t) * 1000
        print(f"ponder 命中: {bm2}  {hit:.0f} ms  depth={r2['info'].get('depth')}")

        # 未命中路径
        e.ponder_start(FEN, [bm], pm, movetime=350)
        time.sleep(0.5)
        t = time.perf_counter()
        e.ponder_miss()
        miss = (time.perf_counter() - t) * 1000
        print(f"ponder 未命中停止: {miss:.0f} ms，随后按正确局面重新搜索")
        after = FEN.replace(" w ", " b ")     # 我方走完后轮到黑方
        t = time.perf_counter()
        bm3, r3 = e.analyse(after, moves=[pm], movetime=350)
        print(f"重新搜索: {bm3}  {(time.perf_counter() - t) * 1000:.0f} ms")

        # 异常路径：喂非法局面，必须限时自愈而不是卡死
        print("-" * 60)
        e.ponder_start(FEN, [], pm, movetime=350)   # 故意漏掉我方着法
        time.sleep(0.3)
        t = time.perf_counter()
        bm4, r4 = e.ponder_hit(timeout=6)
        bad = (time.perf_counter() - t) * 1000
        print(f"非法局面 ponder_hit: bm={bm4}  {bad:.0f} ms  hit={r4['hit']}")
        print(f"  未卡死（<8s）: {bad < 8000}")
        # 自愈后引擎仍可用
        bm5, r5 = e.analyse(FEN, movetime=200)
        print(f"  自愈后仍可用: {bm5}")

    print(f"命中统计: hits={e.ponder_hits} misses={e.ponder_misses}")
    e.quit()

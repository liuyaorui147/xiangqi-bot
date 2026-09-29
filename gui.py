"""象棋 AI 自动对弈助手 —— 图形界面。

打包后与命令行模式共用同一个可执行文件：
  象棋AI助手.exe            图形界面
  象棋AI助手.exe --cli auto 命令行自动对弈

注意：--noconsole 打包出来的窗口程序没有 stdout，用子进程调自己会
收不到任何输出（日志空白）。因此界面改为**进程内直接跑机器人线程**，
并把 sys.stdout 重定向到日志队列。
"""
import json
import os
import queue
import sys
import threading
import time

import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def data_root():
    """数据根目录：打包后 exe 同级优先（便于替换引擎/模板），回退 _internal。"""
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        meipass = getattr(sys, "_MEIPASS", None)
        for p in (exe_dir, meipass):
            if p and os.path.exists(os.path.join(p, "engine")):
                return p
        return exe_dir
    return SCRIPT_DIR


HERE = data_root()
CFG_PATH = os.path.join(HERE, "bot_config.json")

DEFAULTS = {"think_ms": 350, "threads": 0, "hash_mb": 256,
            "poll": 0.35, "wait_max": 90, "auto_next": True,
            "side": "red"}


def load_cfg():
    cfg = dict(DEFAULTS)
    if os.path.exists(CFG_PATH):
        try:
            cfg.update(json.load(open(CFG_PATH)))
        except Exception:
            pass
    return cfg


def save_cfg(cfg):
    with open(CFG_PATH, "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)


class QueueWriter:
    """把 print() 输出逐行送进界面日志队列。"""

    def __init__(self, q):
        self.q = q
        self.buf = ""

    def write(self, s):
        if not s:
            return
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self.q.put(("line", line.rstrip("\r")))

    def flush(self):
        pass

    def isatty(self):
        return False


class App:
    def __init__(self, root):
        self.root = root
        self.cfg = load_cfg()
        self.q = queue.Queue()
        self.thread = None
        self.moves = 0

        root.title("象棋 AI 自动对弈助手  ·  Pikafish")
        root.geometry("920x720")
        root.minsize(780, 580)

        self._build_header()
        self._build_settings()
        self._build_controls()
        self._build_board()
        self._build_log()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after_id = root.after(100, self._pump)
        self.refresh_status()

    # ---------- 界面 ----------
    def _build_header(self):
        f = ttk.Frame(self.root, padding=6)
        f.pack(fill="x")
        ttk.Label(f, text="象棋 AI 自动对弈助手",
                  font=("Microsoft YaHei", 15, "bold")).pack(side="left")
        self.status_var = tk.StringVar(value="状态：未启动")
        ttk.Label(f, textvariable=self.status_var,
                  font=("Microsoft YaHei", 10)).pack(side="right")
        self.backend_var = tk.StringVar(value="后端：检测中…")
        ttk.Label(f, textvariable=self.backend_var).pack(side="right", padx=14)

    def _build_settings(self):
        f = ttk.LabelFrame(self.root, text="引擎设置", padding=6)
        f.pack(fill="x", padx=8, pady=4)
        self.v_think = tk.StringVar(value=str(self.cfg["think_ms"]))
        self.v_thr = tk.StringVar(value=str(self.cfg["threads"]))
        self.v_hash = tk.StringVar(value=str(self.cfg["hash_mb"]))
        self.v_next = tk.BooleanVar(value=bool(self.cfg["auto_next"]))
        pad = {"padx": 6}
        ttk.Label(f, text="思考时间(毫秒)").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(f, textvariable=self.v_think, width=8).grid(row=0, column=1, **pad)
        ttk.Label(f, text="线程数(0=自动)").grid(row=0, column=2, sticky="w", **pad)
        ttk.Entry(f, textvariable=self.v_thr, width=8).grid(row=0, column=3, **pad)
        ttk.Label(f, text="哈希(MB)").grid(row=0, column=4, sticky="w", **pad)
        ttk.Entry(f, textvariable=self.v_hash, width=8).grid(row=0, column=5, **pad)
        ttk.Checkbutton(f, text="对局结束后自动开下一局",
                        variable=self.v_next).grid(row=0, column=6, **pad)
        ttk.Button(f, text="保存设置", command=self.on_save).grid(row=0, column=7, **pad)
        self.v_side = tk.StringVar(value=str(self.cfg.get("side", "red")))
        ttk.Label(f, text="我方执子").grid(row=1, column=0, sticky="w", **pad)
        ttk.Radiobutton(f, text="红（先手）", value="red", command=self._on_side,
                        variable=self.v_side).grid(row=1, column=1, sticky="w", **pad)
        ttk.Radiobutton(f, text="黑（后手）", value="black", command=self._on_side,
                        variable=self.v_side).grid(row=1, column=2, sticky="w", **pad)
        ttk.Label(f, text="执黑时棋盘会翻转，程序按红帅位置自动摆正",
                  foreground="#666").grid(row=1, column=3, columnspan=5, sticky="w", **pad)
        ttk.Label(f, text="思考时间越长棋力越强：350ms≈18-20层，1s≈20层",
                  foreground="#666").grid(row=2, column=0, columnspan=8, sticky="w", **pad)

    def _build_controls(self):
        f = ttk.Frame(self.root, padding=6)
        f.pack(fill="x")
        self.btn_start = ttk.Button(f, text="▶ 开始自动对弈", command=self.on_start)
        self.btn_stop = ttk.Button(f, text="■ 停止", command=self.on_stop, state="disabled")
        self.btn_once = ttk.Button(f, text="走一步(试算)", command=self.on_once_dry)
        self.btn_calib = ttk.Button(f, text="重新标定(换 App)", command=self.on_calib)
        self.btn_refresh = ttk.Button(f, text="刷新状态", command=self.refresh_status)
        self.btn_open = ttk.Button(f, text="打开目录", command=self.on_open_dir)
        for b in (self.btn_start, self.btn_stop, self.btn_once, self.btn_calib,
                  self.btn_refresh, self.btn_open):
            b.pack(side="left", padx=4)

    def _build_board(self):
        f = ttk.LabelFrame(self.root, text="当前局面", padding=4)
        f.pack(fill="both", expand=False, padx=8, pady=4)
        self.board = tk.Text(f, height=12, font=("Consolas", 11), bg="#fbfaf7")
        self.board.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(f)
        right.pack(side="left", fill="y", padx=6)
        ttk.Label(right, text="引擎着法", font=("Microsoft YaHei", 10, "bold")).pack(anchor="w")
        self.move_var = tk.StringVar(value="—")
        ttk.Label(right, textvariable=self.move_var, font=("Consolas", 16),
                  foreground="#c0392b").pack(anchor="w")
        ttk.Label(right, text="已落子").pack(anchor="w", pady=(8, 0))
        self.cnt_var = tk.StringVar(value="0")
        ttk.Label(right, textvariable=self.cnt_var, font=("Consolas", 14)).pack(anchor="w")
        self.fen_var = tk.StringVar(value="")
        ttk.Label(right, text="FEN", foreground="#666").pack(anchor="w", pady=(8, 0))
        ttk.Label(right, textvariable=self.fen_var, wraplength=240,
                  font=("Consolas", 7), foreground="#555").pack(anchor="w")

    def _build_log(self):
        f = ttk.LabelFrame(self.root, text="运行日志", padding=4)
        f.pack(fill="both", expand=True, padx=8, pady=4)
        self.log = scrolledtext.ScrolledText(f, font=("Consolas", 9),
                                             bg="#1e1e1e", fg="#d4d4d4")
        self.log.pack(fill="both", expand=True)
        for name, color in (("ok", "#6ec06e"), ("warn", "#e0b050"),
                            ("hi", "#6ab0f0"), ("mate", "#ff7b7b")):
            self.log.tag_config(name, foreground=color)

    # ---------- 逻辑 ----------
    def refresh_status(self):
        def work():
            txt = "后端：未检测到"
            try:
                sys.path.insert(0, HERE)
                import adb
                s = adb.first_device()
                if s:
                    txt = f"后端：adb {s}"
                else:
                    import capture
                    hit = capture.find_window("天天象棋")
                    txt = f"后端：Windows 窗口 {hit[0]}" if hit else "后端：未检测到设备/窗口"
            except Exception as e:
                txt = f"后端：检测失败 {e}"
            self.q.put(("__backend__", txt))
        threading.Thread(target=work, daemon=True).start()
        self.backend_var.set("后端：检测中…")

    def _start_bot(self, *args):
        if self.thread and self.thread.is_alive():
            messagebox.showinfo("提示", "正在运行中")
            return
        self.board.delete("1.0", "end")
        self.moves = 0
        self.cnt_var.set("0")
        self.move_var.set("—")
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.status_var.set("状态：运行中")
        # tkinter 变量必须在主线程读取：放到子线程里 get() 可能拿到空串，
        # 于是 MY_SIDE 被 `or "red"` 兜底 —— 界面明明选了黑方，跑起来却是
        # 红方（选完重启又变回红，正是这个值没落盘）。这里取好再传进去。
        side = self.v_side.get() or "red"
        self.thread = threading.Thread(target=self._run_bot, args=(side, *args),
                                       daemon=True)
        self.thread.start()

    def _on_side(self):
        """切换执红/执黑立即落盘。

        单选按钮只改内存里的变量，不写配置的话下次启动界面又显示成
        bot_config.json 里的旧值（用户看到的就是"明明选了黑，重启变红"）。
        """
        self.cfg["side"] = self.v_side.get()
        try:
            save_cfg(self.cfg)
        except Exception:
            pass

    def _run_bot(self, side, *args):
        """在界面进程内跑主程序，输出重定向到日志队列。"""
        try:
            sys.path.insert(0, HERE)
            os.chdir(HERE)
            import main
            main.STOP.clear()
            # 界面上选了就立刻生效，不必先点保存（保存只影响下次启动）
            main.MY_SIDE = side
            sys.argv = ["main.py", *args]
            self.q.put(("line", f"$ 启动 {' '.join(args)}"))
            old = sys.stdout
            sys.stdout = QueueWriter(self.q)
            try:
                try:
                    main.main()
                except Exception as e:
                    self.q.put(("line", f"[异常] {type(e).__name__}: {e}"))
            finally:
                sys.stdout = old
        except Exception as e:
            self.q.put(("line", f"[加载失败] {type(e).__name__}: {e}"))
        self.q.put(("__done__", ""))

    def _pump(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "__backend__":
                    self.backend_var.set(val)
                elif kind == "__done__":
                    self.status_var.set("状态：已停止")
                    self.btn_start.config(state="normal")
                    self.btn_stop.config(state="disabled")
                    self.thread = None
                else:
                    self._handle_line(val)
        except queue.Empty:
            pass
        self.after_id = self.root.after(100, self._pump)

    def _handle_line(self, line):
        tag = None
        if "★" in line or "绝杀" in line:
            tag = "mate"
        elif line.startswith("!!") or "失败" in line or "拒绝" in line:
            tag = "warn"
        elif "已落子" in line or "确认生效" in line:
            tag = "ok"
            self.moves += 1
            self.cnt_var.set(str(self.moves))
        elif "引擎建议" in line:
            tag = "hi"
            self.move_var.set(line.split("走")[-1].strip() or "—")
        elif "FEN:" in line:
            self.fen_var.set(line.split("FEN:")[-1].strip())
        elif "轮到我方" in line:
            self.board.delete("1.0", "end")
        elif len(line) > 3 and line[0].isdigit() and line[1] == " ":
            self.board.insert("end", line + "\n")
        self.log.insert("end", line + "\n", tag)
        self.log.see("end")

    # ---------- 按钮 ----------
    def on_start(self):
        self._start_bot("auto")

    def on_stop(self):
        try:
            sys.path.insert(0, HERE)
            import main
            main.STOP.set()
            self.status_var.set("状态：正在停止…")
            print("[界面] 已请求停止，等待当前步骤结束")
        except Exception:
            pass

    def on_once_dry(self):
        self._start_bot("once", "--dry")

    def on_calib(self):
        if not messagebox.askyesno("重新标定",
                                   "请把新象棋开到一局新棋的初始局面（一步未走），然后点“是”。"):
            return
        self._start_bot("calib")

    def on_save(self):
        try:
            self.cfg["think_ms"] = int(self.v_think.get())
            self.cfg["threads"] = int(self.v_thr.get())
            self.cfg["hash_mb"] = int(self.v_hash.get())
            self.cfg["auto_next"] = bool(self.v_next.get())
            self.cfg["side"] = self.v_side.get()
            save_cfg(self.cfg)
            messagebox.showinfo("已保存", "设置已写入 bot_config.json\n下次启动生效。")
        except ValueError:
            messagebox.showerror("错误", "思考时间/线程/哈希必须是数字")

    def on_open_dir(self):
        try:
            os.startfile(HERE)
        except Exception:
            pass

    def on_close(self):
        running = self.thread and self.thread.is_alive()
        if running:
            if not messagebox.askyesno("退出", "自动对弈还在运行，确定退出并停止吗？"):
                return
            try:
                sys.path.insert(0, HERE)
                import main
                main.STOP.set()
            except Exception:
                pass
            self.thread.join(timeout=3)
            # 线程可能是 daemon，主界面一销毁就被强杀，那时 main() 的
            # finally 不会执行 —— 必须在这里显式收掉引擎进程，否则
            # pikafish 会变成孤儿进程留在后台吃 CPU。
            try:
                sys.path.insert(0, HERE)
                import main
                main.shutdown_engine()
            except Exception:
                pass
        try:
            self.root.after_cancel(self.after_id)
        except Exception:
            pass
        self.root.destroy()


def cli_mode():
    """可执行文件带 --cli 参数时走命令行模式。"""
    sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a != "--cli"]
    os.chdir(HERE)
    sys.path.insert(0, HERE)
    import main
    try:
        sys.exit(main.main())
    except KeyboardInterrupt:
        print("\n已退出")


def main():
    if "--cli" in sys.argv:
        cli_mode()
        return
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

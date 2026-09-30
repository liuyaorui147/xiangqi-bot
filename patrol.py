"""对局巡检器：解析最新 run 日志，一屏给出健康度指标。

边跑边看用。用法：python patrol.py [tail行数]

指标分三块：
  1. 进度：本局落子数、耗时、当前执子方、最近胜率
  2. 异常计数：未生效/不可信胜率/校验失败/引擎拒局/等待过久
  3. 最近事件尾部：便于一眼看到卡在哪
"""
import glob
import os
import re
import sys
import time

KEYS = {
    "落子未生效": r"落子可能未生效",
    "胜率不可信": r"本帧不可信",
    "校验失败": r"校验失败",
    "引擎拒局": r"引擎.*(拒局|无响应)|bestmove.*None|引擎建议: None",
    # 注意：不要把 [执子] 行的条数当翻转次数。每局正常就有 2 条（朝向预判
    # → 首手确认），直接计数会把健康对局报成"翻转 4 次"。真正要盯的是
    # 同一局内颜色被改判，下面单独算。
    "结算续局": r"结算\]",
    "开局点场": r"\[开局\] 开始界面点",
    "ponder命中": r"\[ponder\] 命中",
    "点击放弃": r"放弃本轮点击",
}


def main():
    tail_n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    logs = sorted(glob.glob(os.path.join("logs", "run_*.log")),
                  key=os.path.getmtime)
    if not logs:
        print("没有日志")
        return
    p = logs[-1]
    lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
    age = time.time() - os.path.getmtime(p)
    print(f"日志 {os.path.basename(p)}  共{len(lines)}行  "
          f"最后更新{age:.0f}s前{'  ⚠ 疑似卡死' if age > 90 else ''}")

    moves = [l for l in lines if "已落子" in l]
    mated = [l for l in lines if "将死" in l or "绝杀" in l]
    games = [l for l in lines if "新对局已开始" in l]
    wr = []
    for l in lines:
        m = re.search(r"胜率:\s*([\d.]+)%", l)
        if m:
            wr.append(float(m.group(1)))
    side = None
    for l in lines:
        m = re.search(r"\[执子\] (\S+)", l)
        if m:
            side = m.group(1)

    print(f"续局数 {len(games)}   落子 {len(moves)}   "
          f"当前执子 {side or '未定'}   最近胜率 "
          f"{wr[-1]:.1f}%" if wr else "  (无胜率)")
    if wr:
        seg = wr[-40:]
        print(f"  胜率区间 {min(seg):.0f}~{max(seg):.0f}%  "
              f"末5: {' '.join(f'{v:.0f}' for v in wr[-5:])}")

    print("异常计数:")
    for name, pat in KEYS.items():
        c = sum(1 for l in lines if re.search(pat, l))
        flag = "  ⚠" if (c > 3 and name not in ("执子方翻转", "ponder命中", "结算续局")) else ""
        print(f"  {name:<8} {c}{flag}")
    # 真正的"执子方翻转"：同一局内颜色被改判（换局重置）
    flips, prev_side = 0, None
    for l in lines:
        if "新对局已开始" in l:
            prev_side = None
            continue
        m = re.search(r"\[执子\] (红|黑)", l)
        if m:
            if prev_side and m.group(1) != prev_side:
                flips += 1
            prev_side = m.group(1)
    print(f"  {'换边改判':<7} {flips}" + ("  ⚠ 执子方不稳定" if flips else ""))

    # 最近一次落子距现在多久：判断是否在等我方/卡住
    last_move_t = None
    for l in reversed(lines):
        m = re.match(r"\[(\d\d):(\d\d):(\d\d)\]", l)
        if m and ("已落子" in l or "轮到我方" in l):
            last_move_t = l[:10]
            break
    print(f"最近落子时刻 {last_move_t or '无'}  现在 {time.strftime('%H:%M:%S')}")

    print("--- 尾部 ---")
    for l in lines[-tail_n:]:
        print("   " + l[:110])


if __name__ == "__main__":
    main()

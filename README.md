# 象棋 AI 助手（xiangqi-bot）

用**截图识别 + 引擎决策 + 模拟点击**在安卓模拟器里自动下中国象棋的本地工具。
不修改游戏进程、不 hook、不抓包、不需要 root —— 全程只做三件事：看画面、算棋、点屏幕。

> 棋力来自 Pikafish 引擎（NNUE 评估）。本仓库不含引擎本体，
> 只负责把「模拟器画面」与「UCI 引擎」之间的链路接通。

---

## 快速开始

```bash
pip install -r requirements.txt

# 1. 下载 Pikafish 引擎，把可执行文件与权重放进 engine/
#    https://github.com/official-pikafish/Pikafish/releases
#    engine/pikafish.exe
#    engine/pikafish.nnue

# 2. 打开模拟器里的象棋 App，进入一局「一步未走」的新棋界面
python main.py calib          # 标定棋盘位置、收割棋子模板（失败会自动回滚）

# 3. 开始自动对弈
python main.py auto           # 全程托管
```

GUI 版本（推荐日常使用）：

```bash
python gui.py
```

---

## 工作模式

| 命令 | 作用 |
|---|---|
| `python main.py calib` | **换 App / 换皮肤后必做**。重新定位棋盘、收割 32 张棋子模板，自校验失败自动回滚 |
| `python main.py auto` | 自动对弈：识别对手落子 → 引擎决策 → 点击执行，自动开下一局 |
| `python main.py once` | 只走一步就退出，用于验证当前局面识别是否正确 |
| `python main.py watch` | 只观察不打棋，打印每个变化的局面（- 用于诊断识别精度） |
| `python main.py once --dry` | 干跑，算出最优着法但不真点击 |

可选参数：`--red` / `--black` 指定执子方（默认红），`--dry` 干跑。

---

## 它是怎么工作的

```
 模拟器画面
     │  MuMu SDK 共享内存截图（~5.5 ms/帧）或 adb screencap
     ▼
┌─────────────────────────────────────────────────────┐
│ board_locator   几何拟合 8×9 网格并锚定到棋盘左上角     │
│ recognize       剪裁格位 → 模板匹配 → 棋子类型 + FEN    │
│                 → validate_board 合法性校验            │
└─────────────────────────────────────────────────────┘
     │  FEN
     ▼
┌─────────────────────────────────────────────────────┐
│ engine          Pikafish UCI 封装                     │
│                 Ponder 后台预搜索 / 超时自愈重启         │
└─────────────────────────────────────────────────────┘
     │  最优着法 (UCI)
     ▼
┌─────────────────────────────────────────────────────┐
│ clicker         UCI → 格坐标 → MuMu SDK / adb tap      │
└─────────────────────────────────────────────────────┘
     │
     ▼
 回到第一步
```

主循环是 `boot → my_turn → wait_opp` 三态机，配有：

- **局面稳定性判定**：连续两帧相同才行动，避免吃到落子动画中间帧
- **<｜hy_place▁holder▁no▁813｜>点击前复核**：分析到点击之间若局面已变则放弃本轮（stale），不误点
- **看门狗**：等待超过 `wait_max` 秒怀疑我方上一手没生效，强制重新分析
- **Ponder 预搜索**：我方落子后让引擎后台预测对手应手，命中即省掉整个思考时间

### 模块职责

| 文件 | 职责 |
|---|---|
| `main.py` | 编排、状态机、落子流程、终局与弹窗处理 |
| `engine.py` | Pikafish UCI 封装，含 Ponder、超时、崩溃自愈 |
| `recognize.py` | 棋子模板匹配、FEN 编解码、局面合法性校验 |
| `board_locator.py` | 棋盘网格拟合与原点锚定（对 1.5 倍频假网格有硬约束） |
| `harvest.py` | 从初始局面自动收割棋子模板并标定颜色阈值 |
| `mumu_cap.py` | MuMu SDK 共享内存截图与触摸（比 adb 快约 100 倍） |
| `adb.py` | adb 设备发现、连接、截图、点击 |
| `capture.py` | Windows 窗口截图通道（模拟器不可用时备选） |
| `clicker.py` | 点击执行与 UCI ↔ 格坐标互转 |
| `gui.py` | Tk 界面与配置持久化 |

---

## 配置项

运行时读写 `bot_config.json`，GUI 里也能改：

| 键 | 默认 | 说明 |
|---|---|---|
| `side` | `red` | 执子方 |
| `think_ms` | `350` | 每步思考时间，**棋力与速度的唯一兑换旋钮** |
| `threads` | `0` | 引擎线程数，0 为自动（留 1/4 给模拟器） |
| `hash_mb` | `256` | 引擎哈希 |
| `poll` / `poll_idle` | `0.04` / `0.12` | 轮询间隔；对手思考时自动降频采样省 CPU |
| `mumu_tap` | `true` | 用 MuMu SDK 点击（~1 ms），失败自动回退 adb tap（~80 ms） |
| `ponder` | `true` | 后台预搜索 |
| `wait_max` | `90` | 等待对手落子的耐心上限 |
| `auto_next` | `true` | 终局后自动开下一局 |

---

## 性能指标

以下为实测数据（2026-09，Windows 11 + MuMu 模拟器，分辨率 900×1600）：

| 环节 | 耗时 |
|---|---|
| 单帧截图（MuMu SDK） | ~5.5 ms（adb screencap 约 640 ms） |
| 单次棋局识别 | ~2 s → 优化后 < 0.1 s 量级 |
| 发现对手落子 | ~122 ms |
| 点击前复核 | ~11 ms |
| 将死检测 | ~1 ms（`depth=2` 浅搜） |
| **用户感知端到端**（对手落子 → 我方棋子落下） | **~690 ms**（初版约 1.7–2 s） |

详细优化记录可见各模块文档字符串，其中标注了每处优化前后的实测数值。

---

## 依赖

```
numpy>=2.0
opencv-python>=4.10
```

GUI 用 Python 内置 `tkinter`，MuMu SDK 通过 `ctypes` 直调，均无需额外安装。
开发测试环境为 Python 3.13 / Windows 11。

打包成 exe：

```bash
pyinstaller 象棋AI助手.spec
```

---

## 已知限制

坦白说明，避免踩坑：

1. **识别精度依赖真实画面验证**。棋盘定位与决策链路已通过 10 局自对弈验证（0 问题），但「画面 → 局面」的模板匹配环节强依赖具体 App 皮肤，换 App 必须重跑 `calib`。
2. **没有单元测试**。4400 行代码目前依赖手工回归，坐标与 FEN 变换那一层（`uci_to_cells` / `raw_cell` / `orient_board` / `fen_apply_move`）是最容易出错也最值得被测试锁住的部分。
3. **全局状态耦合**。`main.py` 使用模块级全局变量，`orient_board()` 返回翻转标志但由调用方负责写回 `BOARD_FLIP`，新增调用点时务必留意。
4. 本工具仅用于个人学习与单机娱乐场景。

---

## 许可

MIT（见 `LICENSE`）。

注意：**本仓库不含 Pikafish 引擎本体**。Pikafish 为 GPL-3.0 授权，需单独从
[官方仓库](https://github.com/official-pikafish/Pikafish/releases) 获取。
本项目通过 UCI 子进程接口调用它，二者相互独立。

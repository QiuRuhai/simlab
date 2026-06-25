# M1 · XPBD 布料求解器（flag 英雄镜头）· 设计文档

- **日期**：2026-06-25
- **状态**：已通过 brainstorming 评审，待转实现计划
- **前置**：M0 行走骨架已完成（管线/契约/导出/渲染全部就绪并冻结）
- **范围**：把 `wave` 假数据换成真 XPBD 求解器，产出同样的 `SimResult`，复用 M0 整条下游管线

---

## 1. 目标、增量节奏、完成判据

**M1 干的事**：实现 small-steps XPBD 布料求解器，跑 `flag`（旗子）场景，产出 `SimResult` → 复用 M0 的 USD 导出 + Blender 渲染（**渲染端一行不改**）。

**核心取向**：先稳后飘 + 距离式弯曲——工程上都指向"稳"。求解器的稳定性是硬骨头，物理炸了一切白搭。

**内部增量节奏**（M1 自己的"行走骨架"，每步可验证、可开预览看一眼）：

| 子步 | 加什么 | 看到什么 |
|---|---|---|
| **M1a** | 测试骨架 + predict/积分 + 距离约束（结构+剪切）+ 固定 | 钉住的布在重力下垂坠、稳定不炸 |
| **M1b** | + 距离式弯曲 | 不再"软网"，有布的骨感 |
| **M1c** | + 均匀/阵风（`f_ext`） | 旗子摆动飘拂 ← **必须项达成** |
| **M1d**（stretch） | + 按面法线的气动风 | 真·猎猎翻飞（行波） |

**完成判据**：
- **客观**（测试护栏）：topology/integrator/constraints 单测全绿；目标分辨率 250 帧无 NaN/不炸；钉点位置全程不变；对称布局保持对称；边拉伸有界（≤~10%）。
- **主观**（肉眼）：旗子从 hoist 边垂下、重力下沉、风里可信地摆动飘拂（GGUI 预览 + 渲染都看）。
- **stretch**：可见行波翻飞。

**分辨率取舍**（对原 spec "长边~128" 的务实回调）：Jacobi 是弱收敛，高分辨率下布偏软/偏拉伸。**开发用 64×42 快速迭代，出片用 ~96×64（3:2）**；**128² 留给 M1.5 图着色之后**。计算量不是瓶颈（Metal 上 128² 也秒级），瓶颈是 Jacobi 收敛质量。

---

## 2. 求解器算法

**每 substep 循环**（`h = (1/fps) / substeps`），拆成 Taichi kernel：

```
1. predict          自由点(w>0): v += h·(g + f_ext·w);  x_prev = x;  x += h·v
                    钉住点(w=0): x 不动
2. 解约束 (Jacobi, 1 sweep/substep, λ 每 substep 归零)
   · 清零 dx[i]=0, count[i]=0
   · solve_distance  逐边: 见下式, α̃ = α_dist/h²
   · solve_bending   逐弯曲对: 同式, α̃ = α_bend/h²
   · apply_dx        自由点: x += ω · dx/max(1,count);  钉住点跳过
3. update_velocity  自由点: v = (x − x_prev)/h;  v *= (1 − damping)
```

**距离约束投影**（XPBD，单 sweep λ=0）：边 `(i,j,L0)`，`d=x_i−x_j`, `len=|d|`, `n=d/len`：
```
C  = len − L0
Δλ = −C / (w_i + w_j + α̃)
dx[i] += +w_i·Δλ·n ;  dx[j] += −w_j·Δλ·n ;  count[i]++ ; count[j]++
```
**Jacobi 要点**：共享顶点的边并行投影会写竞争 → `dx` 累加缓冲 + `count` 计数（`ti.atomic_add`），最后取平均 + 欠松弛 `ω` 一次性施加，天然无竞争。`α̃=α/h²` 使刚度与步长解耦。纯小步长 = 多 substep × 1 sweep；太软则加 substep 或加 sweep（`solver_iters` 旋钮，默认 1）。

**弯曲（距离式）**：每条内部边的两个对顶点 `(k,l)` 加一条距离约束，静止长度=初始距离，柔度 `α_bend`。投影同上式。

**风**（M1c 均匀+阵风）：`f_ext[i] = wind_dir · base·(1 + gust_amp·sin(2π·gust_freq·t) + noise)`，在 predict 里以 `f_ext·w` 作加速度。M1d 升级：按入射面法线投影累加（迎风吃满）。

**起步参数**（进 flag 预设，M1c/M1d 调手感）：

| 参数 | 起步值 | 参数 | 起步值 |
|---|---|---|---|
| substeps | 20 (h≈2.1ms) | α_dist (拉伸柔度) | ~1e-7（硬） |
| gravity | (0,0,−9.8) | α_bend (弯曲柔度) | ~1e-4（软） |
| 布尺寸/总质量 | 1.5×1.0 m / 0.2 kg | ω (欠松弛) | ~0.5 |
| solver_iters | 1 | damping | ~0.01 |

钉住 = hoist 竖边整条 `w=0`。

---

## 3. 数据结构 + Taichi + 测试 arch

**状态 field**（`[N]`）：`x / v / x_prev / inv_mass` + Jacobi 缓冲 `dx / count`。
**约束表 field**（建场景时算好上传）：`dist_edges[E,2] + rest_len[E]`、`bend_pairs[B,2] + bend_rest[B]`。
**topology.py**（纯 numpy，可独测）：`faces → 唯一边(结构+剪切) + 弯曲对角对`。M1 用 Jacobi，**不需要图着色**（M1.5 才上）。
**kernel 拆分**：`predict / solve_distance / solve_bending / apply_dx / update_velocity` + `forces.py` 风力 eval。

**测试 arch — 一个要提前消化的 Taichi 坑**：单测用 `ti.init(arch=ti.cpu)`（确定性、无 GPU、CI 友好），真跑用 metal。但 Taichi 全局状态、`ti.init` 基本一进程一次，对"逐个小系统单测约束"很别扭。约定：

> **求解器模块自己不调 `ti.init`**（由入口/测试夹具调）。**M1a 第一件事就是把测试骨架跑通**——`ti.cpu` 的 pytest 夹具（session 级 init）+ field 分配策略（`ti.FieldsBuilder` 作用域化，或固定最大尺寸复用），用一个能跑的小约束测试证明这套可行，**再堆约束**。这是 M1 版的"骨架风险"。

---

## 4. flag 场景 + 接入管线（渲染端零改动）

- `scenes/flag.py` + presets `'flag'`：用 `build_grid`（3:2）建网格，算 hoist 竖边钉点（`w=0`），带 wind/质量/柔度/substeps 等参数。
- **朝向**（关键，否则会变成"水平挂布"而非旗子）：把 `build_grid` 的平面网格映射进**竖直平面**——宽→X（fly 方向），高→Z（竖直），y=0。**hoist 边 = x=min 那条竖边（沿 Z）整条钉住**；重力 −Z 让它下沉，风沿 ±Y（垂直旗面）。网格居中于原点，M0 的相机（−Y 看向原点）正好正面框住旗面；构图微调留 M2。
- `build_scene('flag')` → 构造 `ClothSolver` → **跑求解器**：`for f: for s in substeps: substep(); positions[f]=x.to_numpy()` → 返回 `SimResult(positions, faces, uvs, fps, 'flag')`，faces/uvs 来自 `build_grid`。
- 之后**完全复用 M0**：SimResult → `usd_exporter` → `cache.usd` → `render flag`（M0 代码不动）→ mp4。
- `build_scene` 按名分派：`wave`→假数据生成器，`flag`→求解器 runner，两者都返回 `SimResult`。
- `run_sim` 加 `--substeps`（覆盖）、`--preview`（开 GGUI，不落盘）。

**题眼**：M1 只在"产出 SimResult"这一段换引擎，契约与下游纹丝不动。

---

## 5. GGUI 预览（`preview/ggui.py`）

`run_sim flag --preview` 开 `ti.ui.Window`：实时跑求解器 + 渲染布料三角网格 + 基本相机（拖拽）。**范围最小化**——只为肉眼盯稳定性和手感，**不是参数 GUI**。键位：`空格`暂停、`R`重置。用 metal，需显示器；**测试/无头不碰 GGUI**。

---

## 6. 测试不变量（TDD 清单，全在 `ti.cpu`）

- **topology**：结构+剪切边的数量/内容、弯曲对数量/内容、无重复边、索引合法。
- **integrator**：predict 正确施加重力（自由点 v/x 更新）、钉点不动；update_velocity = (x−x_prev)/h。
- **constraints**：
  - 单距离约束 → 投影后恢复静止长度；逆质量加权（重的动得少、钉住伴点不动）
  - `α=0` → 一次投影精确到位；`α` 越大修正越小（越软）
  - 钉点任意多次求解后位置不变
  - **静止构型不自发运动**（平铺+静止长度+无重力无风 → 一 substep 后位置不变，抓符号/构型错）
  - **对称保持**（对称钉+对称力 → 始终对称）
  - 距离式弯曲：共面/静止 → ~零修正；弯折 → 朝静止恢复
- **稳定性护栏**（`test_solver_stability.py`）：小旗子（如 16×10）跑 30–50 帧重力+均匀风 → 无 NaN、最大速度有界、最大边拉伸有界。回归/防爆护栏。

全在 `ti.cpu`，确定性、快。

---

## 7. 模块结构

```
src/xpbd/solver/
├── topology.py      faces→边/弯曲对 (纯 numpy, 可测)
├── integrator.py    predict / update_velocity kernel
├── constraints.py   solve_distance / solve_bending / apply_dx / pin kernel
├── forces.py        风力 eval kernel (重力是常量, 在 predict 里)
└── cloth.py         ClothSolver: 分配 field + 编排 substep 循环 + runner→SimResult
src/xpbd/scenes/flag.py     flag 网格+钉点+参数
src/xpbd/scenes/presets.py  + 'flag' 预设, build_scene 分派 (现有文件改动)
src/xpbd/preview/ggui.py    GGUI 预览
scripts/run_sim.py          + --substeps / --preview (现有文件改动)
tests/  test_topology / test_integrator / test_constraints / test_solver_stability
tests/conftest.py           + ti.cpu 测试夹具 (现有文件改动)
```

---

## 8. 环境与依赖

- **用现有 conda `taichi` 环境**（**不要**用 M0 遗留的 `.venv`）：
  - python 3.10.20、taichi 1.7.4、numpy 2.2.6、usd-core、pytest 9.1.1 —— 均已就绪
  - `xpbd` 已 editable 装入该环境（指向同一份 `src/`）
  - Metal 后端可用；M0 全部快测在此环境 14 passed
- 入口/命令用该环境的 python：`/opt/homebrew/Caskroom/miniconda/base/envs/taichi/bin/python`（或激活后 `python`）。
- **测试**：`python -m pytest`（单测 `ti.cpu`，确定性、不需 GPU、不需 Blender）。
- M0 的 `.venv`（py3.12，455M）已冗余，可删（gitignored，删了不影响）。

---

## 9. 风险与验证点

| 风险 | 何时验证 | 兜底 |
|---|---|---|
| Taichi `ti.init`-once + field 生命周期使单测别扭 | M1a 首步（测试骨架） | FieldsBuilder 作用域化 / 固定最大尺寸复用 |
| Jacobi 收敛弱 → 布偏软/偏拉伸 | M1a 起每步看预览 | 加 substep / solver_iters；回调分辨率；M1.5 图着色 |
| 距离式弯曲各向异性（沿网格方向不均） | M1b | 够用即可；真二面角留作升级 |
| flag 在 hoist 全钉下高频抖动/不稳 | M1c | 调 ω/damping/substeps；阵风频率别太高 |
| 气动风（M1d）引入不稳 | M1d（stretch） | 是 stretch，不稳就退回均匀+阵风交付 |

---

## 10. 非目标（M1 明确不做）

- **图着色**（M1.5）、**真二面角弯曲**（升级）——M1 用 Jacobi + 距离式弯曲。
- **碰撞 / 自碰撞**（v1.1）。
- **look 定稿**（M2）——M1 仍用 M0 的最简红色 look 出片即可（能看清物理就行）。
- 参数 GUI、风的湍流场、撕裂、多布料——均超出 M1。

# XPBD 布料模拟 → USD → Blender 渲染管线 · 设计文档

- **日期**：2026-06-24
- **状态**：已通过 brainstorming 评审，待转实现计划
- **范围**：v1 单段管线的设计（一个 spec → 一个实现计划 → 一轮实现）

---

## 1. 目标与范围

做一条 **模拟 → USD 缓存 → Blender 无头渲染 → mp4** 的布料管线。

**v1 定位**：均衡 + 长期框架。物理够真、画面够看即可，**重心是把可扩展的骨架立对**——之后持续加场景 / 约束 / demo（v1.1 碰撞等）。

**构建策略**：**行走骨架优先（walking-skeleton-first）**。先用假数据打通整条链路并冻结接口，再回填真物理。最大未知数（带动画的 USD → Blender 往返）在第一个里程碑就被验证掉，而不是等求解器写完才暴露。

---

## 2. 架构总览与数据流

两段式，两个**互不相干的 Python 解释器**，唯一接口是磁盘上的 USD 文件。

```
[模拟端 · venv python3.12]
run_sim.py <scene>
  └→ Taichi XPBD 求解器 (Metal / CPU)  ⇄  GGUI 实时预览 (--preview)
       └→ SimResult (positions[F,N,3] + faces[M,3] + uvs[N,2] + fps)
            └→ Exporter → out/<demo>/cache.usd   (+ 可选 cache.npz 调试)

         ── 唯一接口：out/<demo>/cache.usd ──

[渲染端 · Blender 内置 python3.13]
render.py <demo>
  └→ blender --background --python blender_render.py
       └→ wm.usd_import 原生导入 cache.usd（带动画的 mesh sequence cache）
            └→ 套用固化的 look (相机 / 灯光 / 材质)
                 └→ Cycles 渲帧 → out/<demo>/frames/*.png → <demo>.mp4

[开发期] Blender MCP ⇠ 交互式调光，满意后固化进 look.py
```

**核心原则**：模拟端与渲染端永不互相 `import`。`cache.usd` 就是它俩之间完整的 API。这买到的是解耦——改 look 不重跑模拟，改物理不重做 look，缓存还能拖进 Houdini / usdview。

---

## 3. 里程碑路线图

每个里程碑都有"完成 = 可验证产物"，且各自消化一类特定风险。

### M0 — 行走骨架（不碰真物理）
程序化假数据（旗子形状网格 + 行进正弦波 `z = A·sin(kx + ωt)`）跑通整条链路。
- **产物判据**：`run_sim.py wave` → `out/wave/cache.usd`；`render.py wave` → 能看到网格**明显在动**的 `wave.mp4`。
- **消化风险**：SimResult 契约、**带动画的 USD → Blender 5 往返**（全项目最大未知数，死在这）、无头渲染 + 拼帧、目录 / CLI 约定。
- **首步核查**：确认 taichi 有 3.12 / arm64 wheel。

### M1 — XPBD 求解器（假数据换成真物理）
small-steps XPBD（距离 + 弯曲 + 固定约束，先 Jacobi）。`flag` 场景。GGUI 预览。约束做 TDD。
- **英雄镜头 = flag**，它定义 M1 完成判据。
- **flag 预设具体参数**（落在 `presets.py`，均可调）：3:2 横向、长边分辨率 ~128、hoist 竖边整条钉住、重力 + 稳风 + 阵风、250 帧 @ 24fps。
- **产物判据**：旗子在 GGUI 与渲染里都飘得可信；约束单测全绿；250 帧不炸。

### M2 — Look 定稿
Blender 里布料材质 + 灯光 + 相机，满意后固化进 `look.py`。
- **调 look 的方式**（不硬依赖 MCP）：基线是 headless 迭代——改 `look.py` → 渲一帧 → 看 PNG → 再改；若 Blender GUI 开了 MCP 服务（localhost:9876），则升级为对活视口实时调 + 截图，迭代更快。
- **产物判据**：v1 那个"好看的旗子视频"。

### v1.1 — 碰撞（紧接 v1）
求解器加碰撞约束（布搭球 / 地面），管线已稳，只是多一种约束。

### M1.5（可选）— Jacobi → 图着色
收敛 / 性能升级。

---

## 4. SimResult 数据契约（项目脊椎）

只装"画一帧需要什么"，**不装任何 look，也不装求解器内部状态**。

```python
SimResult:
    positions : float32 [F, N, 3]   # 唯一逐帧变化的数据 = 顶点轨迹
    faces     : int32   [M, 3]      # 三角拓扑（静态，全程不变）
    uvs       : float32 [N, 2]      # 每顶点 UV（静态）→ 支撑贴纹理材质
    fps       : float                # → USD 时间码 & mp4 播放速度
    name      : str                  # demo 名，driver out/<name>/
```

**边界**：材质 / 着色器 / 灯光 / 相机 / 渲染设置一概**不进**契约，全留在 Blender 端的 `look.py`。UV 是网格的**内在属性**（纹理坐标系，建网格时就定下，贴不贴图都跟着走），所以进契约；而**用** UV 采样的旗面贴图是 look，留 Blender。

**三条硬约定**：
1. **法线不存**——点缓存惯例。Blender 从形变网格自动重算，GGUI 即时算。存了冗余且易不一致。
2. **坐标系 = Z-up + 米**，USD stage 显式写 `upAxis=Z, metersPerUnit=1.0` → sim / USD / Blender 同一套坐标，避免整体转 90°。
3. **时间码 = fps**（`timeCodesPerSecond=fps`，positions 在第 1..F 帧打时间采样）→ 播放速度正确。

调试用 `.npz` 可额外装 `velocities / inv_mass / 约束残差`，但**核心契约只有上面 5 个字段**——所有 demo、exporter、测试都依赖这个窄接口。

---

## 5. 管线接缝

### 接缝 A — USD 动画编码
一个 `UsdGeomMesh` prim（`/cloth`），拓扑静态、points 逐帧：
- `faceVertexIndices` / `faceVertexCounts`：**只写一次**（counts 全为 3）。
- `points`：**逐帧时间采样** `points.Set(positions[f], time=f)`。
- `primvars:st`（UV）：写一次，`interpolation="vertex"`。
- stage 元数据：`upAxis=Z`、`metersPerUnit=1.0`、`timeCodesPerSecond=fps`、`startTimeCode=1`、`endTimeCode=F`。
- （可选）逐帧 `extent` 包围盒，利于 usdview。

这是教科书式"形变网格 / 点缓存"，Blender 读到时间采样的 points 自动挂 Mesh Sequence Cache。**M0 验证它无头下真逐帧形变，而非冻在第 1 帧。**

### 接缝 B — Blender 端（`blender_render.py`）
`blender --background --python blender_render.py -- <args>`：清场 → `wm.usd_import`（不导相机 / 灯 / 材质）→ 设帧范围 → 套 `look.py` → Cycles 渲 **PNG 帧** 到 `frames/####.png` → 拼 mp4。
- **拼帧**：有 ffmpeg 用 ffmpeg，没有走 Blender 自带 FFMPEG（VSE 一遍过）。**自动探测，不强制装 ffmpeg。** PNG 帧始终落盘（可检视 / 断点续渲 / 改剪辑不重渲）。

### 接缝 C — CLI 契约
```
python scripts/run_sim.py <scene> [--frames N] [--substeps K] [--res N] [--preview] [--npz]
    → out/<scene>/cache.usd   (+ cache.npz)
python scripts/render.py  <demo>  [--samples N] [--res WxH] [--frames a:b] [--no-video]
    → out/<demo>/frames/####.png → out/<demo>/<demo>.mp4
```
`--preview` = 开 GGUI 实时窗口调参（不落盘）；不带就无头导出。两个入口**只通过 `out/<demo>/cache.usd` 通信**。

---

## 6. XPBD 求解器设计

**算法**：Small-steps XPBD（Müller 2020）——每帧切多 substep，每 substep 只解 1 次。更稳、更硬、数值阻尼小。

**每 substep 循环**：
1. 预测：`v += h·(g + f_ext/m); x_prev = x; x += h·v`
2. 解约束（重置 λ=0，逐约束投影）：
   - 距离约束：结构边 + 剪切对角边，`α̃ = α/h²`
   - 弯曲约束：相邻三角形二面角（抗折叠）
   - 固定约束：钉住顶点 `w=0`
3. 更新速度：`v = (x − x_prev) / h`（+ 可选阻尼）

**节奏**：`h = frame_dt / substeps`（24fps、20 substep → h≈2.1ms）。`α̃=α/h²` 用 substep 的 h → 刚度基本与 substep 数解耦。

**并行 / 写竞争**：共享顶点的边并行投影会写冲突。两条路：
- **Jacobi（起步）**：`dx` 累加 field + 每顶点约束计数，`ti.atomic_add` 累加，取平均 + 欠松弛（~0.2–0.5）防过冲。天然无竞争。
- **图着色（M1.5 目标）**：贪心着色在**静态拓扑上只算一次**（建场景时，纯 CPU），组间 Gauss-Seidel、组内并行无竞争，收敛更好。

**弯曲**：目标二面角；**退路** = 对角距离约束（跨边对顶点距离约束），简单且对布料常够用。

**风**（flag）：进 `f_ext`。v1 = 均匀方向力 + 正弦 / 噪声阵风；升级 = 按面法线投影（迎风吃满 → 行波褶皱）。

**数据**：`ti.field` 存 `x / v / x_prev / inv_mass` + 约束表（边对 + 静止长度、弯曲四元组、着色组）。Kernel 拆 `predict / solve_distance / solve_bending / solve_pin / update_velocity`。

**Metal 后端**：`ti.init(arch=ti.metal)`，CPU 兜底。

---

## 7. 模块结构

```
xpbd/
├── pyproject.toml             # python=3.12; deps: taichi,numpy,usd-core; dev: pytest
├── src/xpbd/
│   ├── solver/
│   │   ├── cloth.py           # substep 主循环（编排 kernels）
│   │   ├── constraints.py     # 距离/弯曲/固定 kernels
│   │   ├── integrator.py      # predict / update_velocity
│   │   ├── forces.py          # 风场 kernel（碰撞 v1.1 也落这）
│   │   └── topology.py        # faces→边/弯曲四元组/图着色（纯 numpy，可测）
│   ├── scenes/
│   │   ├── cloth.py           # 参数化 ClothParams + grid 生成 + UV
│   │   └── presets.py         # wave/flag/banner/... 预设字典
│   ├── io/
│   │   ├── result.py          # SimResult 契约
│   │   ├── usd_exporter.py    # → .usd（主力）
│   │   └── npz_exporter.py    # → .npz（调试兜底）
│   └── preview/
│       └── ggui.py            # GGUI 实时预览（即时算法线）
├── blender/                   # 渲染端: 跑在 Blender 内置 py3.13, 不进 venv, 只用 bpy
│   ├── blender_render.py      # 无头: usd_import + look + 渲帧 + 拼 mp4
│   ├── look.py                # 相机/灯光/材质 可复现定义 (look 是代码, 不存 .blend)
│   └── assets/                # 旗面贴图 / HDRI 等 look 输入 (仅渲染端消费)
├── scripts/
│   ├── run_sim.py             # <scene> → cache.usd
│   └── render.py              # <demo> → frames → mp4
├── tests/
│   ├── test_constraints.py    # 约束不变量（TDD）
│   ├── test_topology.py       # 边/弯曲/着色生成正确
│   └── test_exporters.py      # SimResult ↔ npz/usd 往返
└── out/                       # 产物（gitignore）
```

**相对原方案的四处外科手术**（每处有理由，不做无关重构）：
1. `scenes/cloth_drape.py` → 参数化 `cloth.py` + `presets.py`（demo 菜单逼出）。
2. 抽 `solver/topology.py`（拓扑 / 着色是纯 numpy、可独测、被复用）。
3. 加 `solver/forces.py`（风是 flag 一等公民且要升级，碰撞 v1.1 也归这）。
4. 加 `tests/test_topology.py`。

**每个 demo = 一组预设**：`(pin 钉法, wind 风场, pose 初始姿态, shape 长宽比, material, frames)`。求解器 + 管线一通，多加 demo 只是多一行预设。

---

## 8. 环境与依赖

- **模拟端**：`python3.12 -m venv .venv`（系统 python 是 3.14，对 taichi 太新；homebrew 有 3.12）。deps：taichi、numpy、usd-core；dev：pytest。
- **渲染端**：Blender 5.1.2 系统安装（`/opt/homebrew/bin/blender`），内置 Python 3.13，**子进程调用，不进 venv**。
- **硬件**：Apple M2 / arm64 → Taichi Metal 后端可用。
- **ffmpeg**：系统未装，不强制（Blender 自带 FFMPEG 输出兜底）。

---

## 9. 测试策略

**单测**（无 GGUI / Blender，快、确定性）：
- 单距离约束 → 恢复静止长度；逆质量大动得少。
- `α=0` → 一次投影精确到位；`α` 越大越软。
- 钉住顶点永不动。
- 静止构型不自发运动（平铺 + 静止长度 + 无重力 → 一 substep 后不变，抓符号错）。
- 对称保持（对称钉法 + 对称力 → 始终对称）。
- 能量不爆（250 帧最大速度有界，回归护栏）。
- 弯曲：共面→零修正；折叠→朝静止角推。
- 拓扑生成正确（边 / 弯曲四元组 / 着色无相邻同色）。
- exporter 往返：`SimResult→npz→SimResult` 恒等；`→usd→` 重读 positions/faces/uvs 浮点容差内一致，元数据保留。

**集成冒烟测**：`run_sim wave --frames 3` → 断言 cache.usd 有 3 个时间采样（不需 Blender）。

**Blender 往返**：M0 用 headless 渲染验证（`blender --background`，可经 Bash 驱动，最可靠）；MCP 若连上则作为交互式辅助。不进 CI（Blender headless 太重）。

---

## 10. 关键决策记录

- **两段式 + 进程隔离**：sim 与 render 是不同解释器，只通过 USD 文件通信。买解耦。
- **USD 为主力缓存格式**：策略上正确（USD 是未来交换格式，可跨工具），Blender 5.x 原生导入成熟，usd-core pip 干净。**npz + 手写 bpy importer 为文档化退路**，若 M0 发现 USD 动画导入不理想再启用。
- **行走骨架优先**：把集成未知数前置到 M0。
- **法线不存 / Z-up / 时间码=fps**：见 §4。
- **弯曲二面角带距离约束退路**：见 §6。
- **look 是代码、不存 .blend**：blender 端每次从空场景重建，look 可复现、可 diff、同仓库版本管理；不锁在二进制 .blend 里。
- **Blender 端同仓库独立文件夹**：`blender/` 与 `src/xpbd/` 同仓库分目录，隔离的是运行时（py3.13 vs py3.12），不另开仓库。

---

## 11. 风险与验证点

| 风险 | 何时验证 | 兜底 |
|---|---|---|
| 带动画 USD 无头导入 Blender 是否真逐帧形变 | M0 | npz + 手写 bpy importer |
| taichi 3.12 / arm64 wheel 是否存在 | M0 首步 | CPU 后端 / 降级 python |
| 二面角弯曲梯度难调 | M1 | 对角距离约束 |
| Jacobi 收敛慢 / 偏软 | M1 | 加 substep / 升级图着色 |

---

## 12. 非目标（v1 明确不做）

- **碰撞**（自碰撞 / 与球 / 地面）——v1.1。
- 缓存材质 / 灯光 / 相机进 USD——永远留 Blender 端。
- 图着色——M1 用 Jacobi，着色是 M1.5 升级。
- 法线烘焙、风的湍流场、布料撕裂、多布料交互——均超出 v1。

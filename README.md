# simlab — XPBD 布料模拟 → USD → Blender 渲染管线

一条把布料物理做成**可复现渲染**的两段式管线：Taichi XPBD 求解器算出几何 → 导出 USD 点缓存 → Blender 无头渲染成 mp4。

> **状态：M0 行走骨架已完成** ✅
> 用程序化假数据（正弦行进波）打通了 `模拟 → USD → Blender → mp4` 整条链路并冻结了接口。
> 真正的 XPBD 求解器（M1）尚在路上——当前的 `wave` demo 是**假数据**，不是真物理。

---

## 架构

两段式，**两个互不相干的 Python 解释器**，唯一接口是磁盘上的 USD 文件：

```
模拟端  (venv · Python 3.12)
  run_sim.py <scene>
    └─ Taichi XPBD 求解器        (M1; M0 暂用程序化假数据)
         └─ SimResult            (positions[F,N,3] + faces[M,3] + uvs[N,2] + fps)
              └─ USD Exporter ──▶ out/<demo>/cache.usd

         ──────── 唯一接口: cache.usd ────────

渲染端  (Blender 5.x 内置 Python · 子进程调用)
  render.py <demo>
    └─ blender --background --python blender_render.py
         └─ 原生导入 USD → 套 look → EEVEE 渲帧 → frames/*.png
              └─ stitch.py (Blender 自带 FFMPEG) ──▶ out/<demo>/<demo>.mp4
```

**为什么这么分**：改 look 不用重跑模拟，改物理不用重做 look，`cache.usd` 还能拖进 Houdini / usdview。模拟端（Taichi/USD）与渲染端（bpy）永不互相 import——那个 USD 文件就是它们之间完整的 API。

---

## 快速开始

### 依赖
- **Python 3.12**（模拟端；系统 3.13/3.14 对 Taichi 太新）
- **Blender 5.x**（渲染端；需在 `PATH` 上，`blender --version` 可用）
- **macOS / Apple Silicon**（Taichi Metal 后端；CPU 亦可）
- 系统 `ffmpeg` 非必需——拼帧自动回退到 Blender 自带的 FFMPEG

### 安装
```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

### 跑一个 demo
```bash
.venv/bin/python scripts/run_sim.py wave    # → out/wave/cache.usd  (48帧, Z-up, 24fps)
.venv/bin/python scripts/render.py  wave    # → out/wave/frames/*.png → out/wave/wave.mp4
```

常用开关：
```bash
run_sim.py wave --frames 120 --res 96 --npz   # 帧数 / 网格长边分辨率 / 同时导出调试 npz
render.py  wave --res 1280x720 --samples 64    # 渲染像素分辨率 / EEVEE 采样
render.py  wave --no-video                      # 只渲帧不拼视频
```

### 跑测试
```bash
.venv/bin/pytest              # 全部（含会启 Blender 的 slow e2e）
.venv/bin/pytest -m "not slow"  # 只跑快测
```

---

## 项目结构

```
src/xpbd/                 模拟端 Python 包（venv 内, 不依赖 bpy）
├── io/
│   ├── result.py         SimResult 数据契约（项目脊椎）
│   ├── usd_exporter.py   SimResult → .usd（时间采样 points + Z-up + st）
│   └── npz_exporter.py   SimResult ↔ .npz（调试兜底）
└── scenes/
    ├── cloth.py          build_grid 平面网格生成（M1 复用）
    ├── wave.py           M0 假数据：正弦行进波
    └── presets.py        场景预设 + build_scene 调度

blender/                  渲染端（跑在 Blender 内置 Python, 仅用 bpy）
├── blender_render.py     无头导入 USD + 套 look + 渲 PNG 帧
├── look.py              相机/灯光/材质（可复现, 不存 .blend）
└── stitch.py            VSE 拼帧 → mp4（Blender 自带 FFMPEG）

scripts/
├── run_sim.py            入口: <scene> → cache.usd
└── render.py             入口: <demo> → frames → mp4

tests/                    pytest（契约/导出器/场景/端到端）
docs/superpowers/         设计 spec + 实现计划
```

---

## 路线图

| 里程碑 | 内容 | 状态 |
|---|---|---|
| **M0** | 行走骨架：假数据打通 sim→USD→Blender→mp4 全链路 | ✅ 完成 |
| **M1** | 真 small-steps XPBD 求解器（距离/弯曲/固定约束）+ flag 旗子英雄镜头 | 🚧 下一步 |
| **M2** | look 定稿（布料材质 / 灯光 / 相机, Cycles 出片） | ⬜ |
| **v1.1** | 碰撞（布搭球 / 地面） | ⬜ |

---

## 文档

- [设计 spec](docs/superpowers/specs/2026-06-24-xpbd-cloth-pipeline-design.md) — 架构、SimResult 契约、USD 编码约定、关键决策
- [M0 实现计划](docs/superpowers/plans/2026-06-24-xpbd-cloth-pipeline-m0-walking-skeleton.md) — 11 个 TDD task 的逐步落地

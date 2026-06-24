# XPBD 布料管线 · M0 行走骨架 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用程序化假数据（正弦行进波网格）打通 `run_sim → SimResult → cache.usd → Blender 无头导入 → PNG 帧 → mp4` 整条链路，验证所有接缝，**不写任何真物理**。

**Architecture:** 两段式、两个互不相干的 Python 解释器。模拟端（venv py3.12）生成 `SimResult` 并导出 `out/<demo>/cache.usd`；渲染端（Blender 内置 py3.13，子进程调用）原生导入 USD、套最小 look、渲帧、拼 mp4。两端唯一接口是磁盘上的 `cache.usd`。

**Tech Stack:** Python 3.12 · numpy · usd-core(pxr) · taichi(仅装+冒烟，M1 才用) · pytest · Blender 5.1.2(bpy, EEVEE)

## Global Constraints

逐条来自 spec，每个 task 都隐含遵守：

- **模拟端 Python = 3.12**（系统 3.14 对 taichi 太新；用 `/opt/homebrew/bin/python3.12`）。**Blender = 系统 5.1.2，内置 py3.13，子进程调用，永不 import `xpbd` 包。**
- **坐标系 = Z-up + 米**：USD `upAxis=Z`、`metersPerUnit=1.0`。
- **USD 动画编码**：单个 `UsdGeomMesh`，拓扑静态（`faceVertexCounts`/`Indices` 写一次），`points` 逐帧时间采样；`timeCodesPerSecond=fps`、`startTimeCode=1`、`endTimeCode=F`。
- **法线不存**（Blender 重算）。
- **UV** 作为 primvar `st`，`interpolation="vertex"`。
- **SimResult 核心字段仅 5 个**：`positions[F,N,3] float32`、`faces[M,3] int32`、`uvs[N,2] float32`、`fps float`、`name str`。**不装任何 look。**
- **dtype 锁定**：positions/uvs = `float32`，faces = `int32`。
- **look 是代码**（`blender/look.py`），不存 `.blend`；每次渲染从空场景重建。
- **mp4**：有 `ffmpeg` 用 ffmpeg，否则用 Blender 自带 FFMPEG（`format='MPEG4'`、`codec='H264'`）。**PNG 帧始终落盘。**
- **M0 渲染引擎 = `BLENDER_EEVEE`**（快、已实测无头可用）；M2 才切 `CYCLES`。

**已实测确认的 Blender 5.1.2 API**（写代码直接用，勿改名）：
- `bpy.ops.wm.usd_import(filepath=, import_cameras=False, import_lights=False, import_materials=False, read_mesh_uvs=True, set_frame_range=True)` — `set_frame_range=True` 自动按 USD 设帧范围。
- VSE：`se = scene.sequence_editor_create(); strip = se.strips.new_image(name, filepath, channel, frame_start)`；后续帧 `strip.elements.append(basename)`；`scene.render.use_sequencer=True`。
- 引擎 id 是 `'BLENDER_EEVEE'`（5.x 不带 `_NEXT`）。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `pyproject.toml` | 包定义 + 依赖（py3.12, numpy, usd-core, taichi; dev: pytest） |
| `src/xpbd/__init__.py` | 包根 |
| `src/xpbd/io/result.py` | `SimResult` 数据契约 + 校验 |
| `src/xpbd/io/npz_exporter.py` | `SimResult ↔ .npz` 调试往返 |
| `src/xpbd/io/usd_exporter.py` | `SimResult → .usd`（关键写入接缝） |
| `src/xpbd/scenes/cloth.py` | `build_grid()` 平面网格（M1 复用） |
| `src/xpbd/scenes/wave.py` | `generate_wave()` M0 假数据动画 |
| `src/xpbd/scenes/presets.py` | `PRESETS` + `build_scene()` 调度 |
| `scripts/run_sim.py` | 入口：`<scene> → cache.usd` |
| `blender/look.py` | 最小 look（相机/灯/材质），M2 换内脏 |
| `blender/blender_render.py` | 无头：导 USD + look + 渲 PNG 帧 |
| `blender/stitch.py` | VSE：PNG 帧 → mp4 |
| `scripts/render.py` | 入口：`<demo> → frames → mp4`（调 Blender 子进程） |
| `tests/test_result.py` | SimResult 校验 |
| `tests/test_exporters.py` | npz/usd 往返 |
| `tests/test_scenes.py` | grid/wave 正确性 |
| `tests/test_run_sim.py` | run_sim 产出 USD |
| `tests/test_e2e_m0.py` | M0 端到端验收（含动起来的像素差） |

---

### Task 1: 项目骨架 + 开发环境

**Files:**
- Create: `pyproject.toml`
- Create: `src/xpbd/__init__.py`, `src/xpbd/io/__init__.py`, `src/xpbd/scenes/__init__.py`
- Create: `tests/__init__.py`, `tests/conftest.py`

**Interfaces:**
- Consumes: 无
- Produces: 可 `pip install -e .` 的包 `xpbd`；venv 在 `.venv/`；pytest 可跑。

- [ ] **Step 1: 写 `pyproject.toml`**

```toml
[project]
name = "xpbd"
version = "0.0.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "numpy>=1.26",
    "usd-core>=24.0",
    "taichi>=1.7",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: end-to-end tests that invoke Blender"]
```

- [ ] **Step 2: 建包骨架文件**

`src/xpbd/__init__.py`、`src/xpbd/io/__init__.py`、`src/xpbd/scenes/__init__.py`、`tests/__init__.py` 全部空文件即可。

`tests/conftest.py`：
```python
import sys
from pathlib import Path

# 让测试能 import scripts/ 下的入口模块
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
```

- [ ] **Step 3: 建 venv 并安装**

Run:
```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e ".[dev]"
```
Expected: 安装成功，无 wheel 缺失错误。**若 taichi 无 3.12/arm64 wheel 在此即暴露**（spec 风险表的 M0 首步核查）。

- [ ] **Step 4: 依赖冒烟验证（含 taichi Metal）**

Run:
```bash
.venv/bin/python -c "import numpy; from pxr import Usd, UsdGeom; print('numpy+pxr OK')"
.venv/bin/python -c "import taichi as ti; ti.init(arch=ti.metal); print('taichi metal OK')"
```
Expected: 两行分别打印 `numpy+pxr OK` 和 `taichi metal OK`（taichi 会打印一行 `[Taichi] Starting on arch=metal`）。若 Metal 失败，记录并改用 `arch=ti.cpu` 重试确认 taichi 本身可用。

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src tests
git commit -m "chore: 项目骨架 + py3.12 venv + 依赖冒烟"
```

---

### Task 2: SimResult 数据契约

**Files:**
- Create: `src/xpbd/io/result.py`
- Test: `tests/test_result.py`

**Interfaces:**
- Consumes: 无
- Produces:
  ```python
  @dataclass
  class SimResult:
      positions: np.ndarray  # float32 [F, N, 3]
      faces: np.ndarray      # int32  [M, 3]
      uvs: np.ndarray        # float32 [N, 2]
      fps: float
      name: str
      @property
      def num_frames(self) -> int   # F
      @property
      def num_verts(self) -> int    # N
      @property
      def num_faces(self) -> int    # M
  ```
  构造时强制 dtype 并校验形状一致；不一致抛 `ValueError`。

- [ ] **Step 1: 写失败测试 `tests/test_result.py`**

```python
import numpy as np
import pytest
from xpbd.io.result import SimResult


def _mini():
    positions = np.zeros((3, 4, 3), dtype=np.float32)   # F=3, N=4
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)  # M=2
    uvs = np.zeros((4, 2), dtype=np.float32)
    return SimResult(positions=positions, faces=faces, uvs=uvs, fps=24.0, name="t")


def test_counts():
    r = _mini()
    assert r.num_frames == 3
    assert r.num_verts == 4
    assert r.num_faces == 2


def test_forces_dtype():
    r = SimResult(
        positions=np.zeros((2, 4, 3)),          # 传 float64
        faces=np.array([[0, 1, 2], [0, 2, 3]]),  # 传 int64
        uvs=np.zeros((4, 2)),
        fps=24.0, name="t",
    )
    assert r.positions.dtype == np.float32
    assert r.faces.dtype == np.int32
    assert r.uvs.dtype == np.float32


def test_vert_count_mismatch_raises():
    positions = np.zeros((3, 4, 3), dtype=np.float32)  # N=4
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    uvs = np.zeros((5, 2), dtype=np.float32)           # N=5 ≠ 4
    with pytest.raises(ValueError):
        SimResult(positions=positions, faces=faces, uvs=uvs, fps=24.0, name="t")


def test_bad_position_shape_raises():
    with pytest.raises(ValueError):
        SimResult(
            positions=np.zeros((3, 4, 2), dtype=np.float32),  # 末维不是 3
            faces=np.array([[0, 1, 2]], dtype=np.int32),
            uvs=np.zeros((4, 2), dtype=np.float32),
            fps=24.0, name="t",
        )
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_result.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'xpbd.io.result'`

- [ ] **Step 3: 实现 `src/xpbd/io/result.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimResult:
    """模拟端 → 渲染端 的唯一数据契约：几何随时间变化，不含任何 look。"""

    positions: np.ndarray  # float32 [F, N, 3]
    faces: np.ndarray      # int32  [M, 3]
    uvs: np.ndarray        # float32 [N, 2]
    fps: float
    name: str

    def __post_init__(self) -> None:
        self.positions = np.ascontiguousarray(self.positions, dtype=np.float32)
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int32)
        self.uvs = np.ascontiguousarray(self.uvs, dtype=np.float32)

        if self.positions.ndim != 3 or self.positions.shape[2] != 3:
            raise ValueError(f"positions must be [F,N,3], got {self.positions.shape}")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"faces must be [M,3], got {self.faces.shape}")
        if self.uvs.ndim != 2 or self.uvs.shape[1] != 2:
            raise ValueError(f"uvs must be [N,2], got {self.uvs.shape}")

        n = self.positions.shape[1]
        if self.uvs.shape[0] != n:
            raise ValueError(f"uvs N={self.uvs.shape[0]} != positions N={n}")
        if self.faces.size and int(self.faces.max()) >= n:
            raise ValueError("face index out of range of vertex count")

    @property
    def num_frames(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_verts(self) -> int:
        return int(self.positions.shape[1])

    @property
    def num_faces(self) -> int:
        return int(self.faces.shape[0])
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_result.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/xpbd/io/result.py tests/test_result.py
git commit -m "feat: SimResult 数据契约 + 形状/dtype 校验"
```

---

### Task 3: npz 调试往返

**Files:**
- Create: `src/xpbd/io/npz_exporter.py`
- Test: `tests/test_exporters.py`

**Interfaces:**
- Consumes: `SimResult`
- Produces:
  ```python
  def save_npz(result: SimResult, path) -> Path
  def load_npz(path) -> SimResult
  ```

- [ ] **Step 1: 写失败测试（追加到 `tests/test_exporters.py`）**

```python
import numpy as np
from xpbd.io.result import SimResult
from xpbd.io.npz_exporter import save_npz, load_npz


def _scene():
    positions = np.random.rand(5, 6, 3).astype(np.float32)
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    uvs = np.random.rand(6, 2).astype(np.float32)
    return SimResult(positions=positions, faces=faces, uvs=uvs, fps=24.0, name="rt")


def test_npz_roundtrip_identity(tmp_path):
    r = _scene()
    p = save_npz(r, tmp_path / "cache.npz")
    assert p.exists()
    r2 = load_npz(p)
    np.testing.assert_array_equal(r.positions, r2.positions)
    np.testing.assert_array_equal(r.faces, r2.faces)
    np.testing.assert_array_equal(r.uvs, r2.uvs)
    assert r2.fps == r.fps
    assert r2.name == r.name
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_exporters.py::test_npz_roundtrip_identity -v`
Expected: FAIL，`No module named 'xpbd.io.npz_exporter'`

- [ ] **Step 3: 实现 `src/xpbd/io/npz_exporter.py`**

```python
from __future__ import annotations

from pathlib import Path

import numpy as np

from xpbd.io.result import SimResult


def save_npz(result: SimResult, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        positions=result.positions,
        faces=result.faces,
        uvs=result.uvs,
        fps=np.float64(result.fps),
        name=np.array(result.name),
    )
    # np.savez 会自动补 .npz 后缀
    return path if path.suffix == ".npz" else path.with_suffix(".npz")


def load_npz(path) -> SimResult:
    data = np.load(Path(path), allow_pickle=False)
    return SimResult(
        positions=data["positions"],
        faces=data["faces"],
        uvs=data["uvs"],
        fps=float(data["fps"]),
        name=str(data["name"]),
    )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_exporters.py::test_npz_roundtrip_identity -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/xpbd/io/npz_exporter.py tests/test_exporters.py
git commit -m "feat: npz 调试往返 exporter"
```

---

### Task 4: USD exporter（关键写入接缝）

**Files:**
- Create: `src/xpbd/io/usd_exporter.py`
- Test: `tests/test_exporters.py`（追加）

**Interfaces:**
- Consumes: `SimResult`
- Produces:
  ```python
  def export_usd(result: SimResult, path) -> Path   # 返回写出的 .usd 路径
  ```
  产物：`/cloth` 单 mesh，拓扑静态，`points` 在 timecode `1..F` 逐帧采样，`st` UV，stage `upAxis=Z / metersPerUnit=1 / timeCodesPerSecond=fps / start=1 / end=F`。

- [ ] **Step 1: 写失败测试（追加到 `tests/test_exporters.py`）**

```python
from xpbd.io.usd_exporter import export_usd


def test_usd_roundtrip(tmp_path):
    r = _scene()  # F=5, N=6, M=2
    out = export_usd(r, tmp_path / "cache.usd")
    assert out.exists()

    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(out))

    # stage 元数据
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert stage.GetTimeCodesPerSecond() == r.fps
    assert stage.GetStartTimeCode() == 1
    assert stage.GetEndTimeCode() == r.num_frames

    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/cloth"))
    assert mesh

    # 拓扑静态、三角形
    counts = list(mesh.GetFaceVertexCountsAttr().Get())
    assert counts == [3] * r.num_faces
    indices = list(mesh.GetFaceVertexIndicesAttr().Get())
    assert indices == r.faces.flatten().tolist()

    # points 逐帧时间采样
    samples = mesh.GetPointsAttr().GetTimeSamples()
    assert len(samples) == r.num_frames
    pts1 = np.array(mesh.GetPointsAttr().Get(Usd.TimeCode(1)), dtype=np.float32)
    np.testing.assert_allclose(pts1, r.positions[0], rtol=0, atol=1e-6)

    # UV primvar st
    st = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    assert st
    assert st.GetInterpolation() == UsdGeom.Tokens.vertex
    uvs = np.array(st.Get(), dtype=np.float32)
    np.testing.assert_allclose(uvs, r.uvs, rtol=0, atol=1e-6)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_exporters.py::test_usd_roundtrip -v`
Expected: FAIL，`No module named 'xpbd.io.usd_exporter'`

- [ ] **Step 3: 实现 `src/xpbd/io/usd_exporter.py`**

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Sdf, Usd, UsdGeom, Vt

from xpbd.io.result import SimResult


def export_usd(result: SimResult, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(result.fps)
    stage.SetFramesPerSecond(result.fps)
    stage.SetStartTimeCode(1)
    stage.SetEndTimeCode(result.num_frames)

    mesh = UsdGeom.Mesh.Define(stage, "/cloth")
    stage.SetDefaultPrim(mesh.GetPrim())

    # 拓扑静态：写一次
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(
        np.full(result.num_faces, 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(
        result.faces.reshape(-1).astype(np.int32)))

    # points 逐帧时间采样 + 逐帧 extent
    points_attr = mesh.CreatePointsAttr()
    extent_attr = mesh.CreateExtentAttr()
    for f in range(result.num_frames):
        frame_pts = result.positions[f].astype(np.float32)
        tc = Usd.TimeCode(f + 1)
        points_attr.Set(Vt.Vec3fArray.FromNumpy(frame_pts), tc)
        lo = frame_pts.min(axis=0)
        hi = frame_pts.max(axis=0)
        extent_attr.Set(Vt.Vec3fArray.FromNumpy(
            np.stack([lo, hi]).astype(np.float32)), tc)

    # UV 作为 primvar st（顶点插值，静态）
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray.FromNumpy(result.uvs.astype(np.float32)))

    stage.GetRootLayer().Save()
    return path
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_exporters.py -v`
Expected: 2 passed（npz + usd）

- [ ] **Step 5: Commit**

```bash
git add src/xpbd/io/usd_exporter.py tests/test_exporters.py
git commit -m "feat: USD exporter（时间采样 points + Z-up + st）"
```

---

### Task 5: 平面网格生成（M1 复用）

**Files:**
- Create: `src/xpbd/scenes/cloth.py`
- Test: `tests/test_scenes.py`

**Interfaces:**
- Consumes: 无
- Produces:
  ```python
  def build_grid(nx: int, ny: int, width: float, height: float
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]
      # 返回 (verts[N,3] float32, faces[M,3] int32, uvs[N,2] float32)
      # N=nx*ny, M=2*(nx-1)*(ny-1)。XY 平面、z=0、中心在原点；顶点序 idx=j*nx+i。
  ```

- [ ] **Step 1: 写失败测试 `tests/test_scenes.py`**

```python
import numpy as np
from xpbd.scenes.cloth import build_grid


def test_grid_counts_and_ranges():
    nx, ny, w, h = 5, 4, 2.0, 1.0
    verts, faces, uvs = build_grid(nx, ny, w, h)
    assert verts.shape == (nx * ny, 3)
    assert faces.shape == (2 * (nx - 1) * (ny - 1), 3)
    assert uvs.shape == (nx * ny, 2)
    assert verts.dtype == np.float32 and faces.dtype == np.int32

    # z 全 0；x,y 跨度等于 width/height 且居中
    assert np.allclose(verts[:, 2], 0.0)
    assert np.isclose(verts[:, 0].min(), -w / 2) and np.isclose(verts[:, 0].max(), w / 2)
    assert np.isclose(verts[:, 1].min(), -h / 2) and np.isclose(verts[:, 1].max(), h / 2)

    # uv 落在 [0,1]，含 0 和 1
    assert uvs.min() == 0.0 and uvs.max() == 1.0

    # 面索引合法
    assert faces.max() < nx * ny and faces.min() >= 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_scenes.py -v`
Expected: FAIL，`No module named 'xpbd.scenes.cloth'`

- [ ] **Step 3: 实现 `src/xpbd/scenes/cloth.py`**

```python
from __future__ import annotations

import numpy as np


def build_grid(nx: int, ny: int, width: float, height: float):
    """XY 平面规则网格，z=0，中心在原点。顶点序 idx = j*nx + i。"""
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be >= 2")

    i = np.arange(nx)
    j = np.arange(ny)
    ii, jj = np.meshgrid(i, j)            # 形状 [ny, nx]
    u = ii / (nx - 1)
    v = jj / (ny - 1)

    x = (u - 0.5) * width
    y = (v - 0.5) * height
    z = np.zeros_like(x)
    verts = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
    uvs = np.stack([u, v], axis=-1).reshape(-1, 2).astype(np.float32)

    faces = []
    for jc in range(ny - 1):
        for ic in range(nx - 1):
            v00 = jc * nx + ic
            v10 = jc * nx + ic + 1
            v01 = (jc + 1) * nx + ic
            v11 = (jc + 1) * nx + ic + 1
            faces.append((v00, v10, v11))   # CCW（从 +Z 看）
            faces.append((v00, v11, v01))
    faces = np.array(faces, dtype=np.int32)
    return verts, faces, uvs
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_scenes.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/xpbd/scenes/cloth.py tests/test_scenes.py
git commit -m "feat: build_grid 平面网格生成"
```

---

### Task 6: 正弦行进波动画 + 场景预设

**Files:**
- Create: `src/xpbd/scenes/wave.py`
- Create: `src/xpbd/scenes/presets.py`
- Test: `tests/test_scenes.py`（追加）

**Interfaces:**
- Consumes: `build_grid`, `SimResult`
- Produces:
  ```python
  # wave.py
  def generate_wave(verts, faces, uvs, *, name, frames, fps,
                    amplitude, wavelength, speed) -> SimResult
  # presets.py
  PRESETS: dict[str, dict]                       # 至少含 "wave"
  def build_scene(name, *, frames=None, res=None) -> SimResult
  ```

- [ ] **Step 1: 写失败测试（追加到 `tests/test_scenes.py`）**

```python
from xpbd.scenes.wave import generate_wave
from xpbd.scenes.presets import PRESETS, build_scene
from xpbd.io.result import SimResult


def test_wave_only_z_animates():
    verts, faces, uvs = build_grid(8, 6, 2.0, 1.5)
    r = generate_wave(verts, faces, uvs, name="wave", frames=10, fps=24.0,
                      amplitude=0.2, wavelength=1.0, speed=1.0)
    assert isinstance(r, SimResult)
    assert r.positions.shape == (10, 8 * 6, 3)
    # x,y 全程不变
    np.testing.assert_allclose(r.positions[:, :, 0], r.positions[0, :, 0], atol=1e-6)
    np.testing.assert_allclose(r.positions[:, :, 1], r.positions[0, :, 1], atol=1e-6)
    # z 真的动了：某后续帧与首帧不同
    assert not np.allclose(r.positions[5, :, 2], r.positions[0, :, 2])


def test_build_scene_wave():
    assert "wave" in PRESETS
    r = build_scene("wave", frames=12)
    assert isinstance(r, SimResult)
    assert r.num_frames == 12
    assert r.name == "wave"


def test_build_scene_res_override():
    r = build_scene("wave", frames=4, res=32)
    # res 是长边；wave 预设 nx>ny，所以 nx 变 32
    assert r.num_verts <= 32 * 32
    assert r.num_verts >= 32  # 至少一行
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_scenes.py -k "wave or build_scene" -v`
Expected: FAIL，`No module named 'xpbd.scenes.wave'`

- [ ] **Step 3: 实现 `src/xpbd/scenes/wave.py`**

```python
from __future__ import annotations

import numpy as np

from xpbd.io.result import SimResult


def generate_wave(verts, faces, uvs, *, name, frames, fps,
                  amplitude, wavelength, speed) -> SimResult:
    """假数据：沿 X 行进的正弦波，只动 z。z = A sin(k x - ω t)。"""
    verts = np.asarray(verts, dtype=np.float32)
    x = verts[:, 0]
    k = 2.0 * np.pi / wavelength
    omega = 2.0 * np.pi * speed

    positions = np.empty((frames, verts.shape[0], 3), dtype=np.float32)
    for f in range(frames):
        t = f / fps
        frame = verts.copy()
        frame[:, 2] = amplitude * np.sin(k * x - omega * t)
        positions[f] = frame

    return SimResult(positions=positions, faces=faces, uvs=uvs, fps=fps, name=name)
```

- [ ] **Step 4: 实现 `src/xpbd/scenes/presets.py`**

```python
from __future__ import annotations

from xpbd.io.result import SimResult
from xpbd.scenes.cloth import build_grid
from xpbd.scenes.wave import generate_wave

PRESETS: dict[str, dict] = {
    "wave": dict(
        nx=64, ny=48, width=2.0, height=1.5,
        frames=48, fps=24.0,
        amplitude=0.15, wavelength=1.0, speed=1.0,
    ),
}


def build_scene(name: str, *, frames: int | None = None, res: int | None = None) -> SimResult:
    if name not in PRESETS:
        raise KeyError(f"unknown scene '{name}'; known: {sorted(PRESETS)}")
    p = dict(PRESETS[name])
    if frames is not None:
        p["frames"] = frames
    if res is not None:
        if p["nx"] >= p["ny"]:           # res = 长边分辨率, 取较长轴
            aspect = p["ny"] / p["nx"]
            p["nx"] = int(res)
            p["ny"] = max(2, round(res * aspect))
        else:
            aspect = p["nx"] / p["ny"]
            p["ny"] = int(res)
            p["nx"] = max(2, round(res * aspect))

    verts, faces, uvs = build_grid(p["nx"], p["ny"], p["width"], p["height"])
    return generate_wave(
        verts, faces, uvs, name=name, frames=p["frames"], fps=p["fps"],
        amplitude=p["amplitude"], wavelength=p["wavelength"], speed=p["speed"],
    )
```

- [ ] **Step 5: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_scenes.py -v`
Expected: 全部 passed（grid + wave + build_scene 共 4 个）

- [ ] **Step 6: Commit**

```bash
git add src/xpbd/scenes/wave.py src/xpbd/scenes/presets.py tests/test_scenes.py
git commit -m "feat: 正弦行进波动画 + wave 场景预设"
```

---

### Task 7: run_sim.py 入口

**Files:**
- Create: `scripts/run_sim.py`
- Test: `tests/test_run_sim.py`

**Interfaces:**
- Consumes: `build_scene`, `export_usd`, `save_npz`
- Produces:
  ```python
  def main(argv: list[str] | None = None) -> Path   # 返回 cache.usd 路径
  # CLI: run_sim.py <scene> [--frames N] [--res N] [--npz] [--out DIR]
  ```

- [ ] **Step 1: 写失败测试 `tests/test_run_sim.py`**

```python
from pathlib import Path
import run_sim  # 经 conftest.py 加入 sys.path


def test_run_sim_writes_usd_with_timesamples(tmp_path):
    out_dir = tmp_path / "wave"
    usd = run_sim.main(["wave", "--frames", "3", "--out", str(out_dir)])
    assert usd == out_dir / "cache.usd"
    assert usd.exists()

    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(usd))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/cloth"))
    assert len(mesh.GetPointsAttr().GetTimeSamples()) == 3


def test_run_sim_npz_flag(tmp_path):
    out_dir = tmp_path / "wave"
    run_sim.main(["wave", "--frames", "2", "--out", str(out_dir), "--npz"])
    assert (out_dir / "cache.npz").exists()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_run_sim.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'run_sim'`

- [ ] **Step 3: 实现 `scripts/run_sim.py`**

```python
from __future__ import annotations

import argparse
from pathlib import Path

from xpbd.io.npz_exporter import save_npz
from xpbd.io.usd_exporter import export_usd
from xpbd.scenes.presets import build_scene


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description="跑场景 → out/<scene>/cache.usd")
    ap.add_argument("scene", help="场景名，如 wave")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--res", type=int, default=None, help="长边分辨率")
    ap.add_argument("--npz", action="store_true", help="同时写调试 cache.npz")
    ap.add_argument("--out", type=str, default=None, help="输出目录（默认 out/<scene>）")
    args = ap.parse_args(argv)

    result = build_scene(args.scene, frames=args.frames, res=args.res)
    out_dir = Path(args.out) if args.out else Path("out") / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)

    usd_path = export_usd(result, out_dir / "cache.usd")
    print(f"[run_sim] wrote {usd_path}  (F={result.num_frames}, N={result.num_verts})")
    if args.npz:
        npz_path = save_npz(result, out_dir / "cache.npz")
        print(f"[run_sim] wrote {npz_path}")
    return usd_path


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_run_sim.py -v`
Expected: 2 passed

- [ ] **Step 5: 手动跑一次产出真缓存**

Run: `.venv/bin/python scripts/run_sim.py wave`
Expected: 打印 `wrote out/wave/cache.usd (F=48, N=3072)`；`out/wave/cache.usd` 存在。

- [ ] **Step 6: Commit**

```bash
git add scripts/run_sim.py tests/test_run_sim.py
git commit -m "feat: run_sim 入口（scene → cache.usd）"
```

---

### Task 8: Blender 最小 look + 无头渲帧

**Files:**
- Create: `blender/look.py`
- Create: `blender/blender_render.py`

**Interfaces:**
- Consumes: `out/<demo>/cache.usd`（由 run_sim 产出）
- Produces: `out/<demo>/frames/####.png`
  - `look.setup(scene, engine="BLENDER_EEVEE", samples=16)` — 建相机（瞄原点）/太阳光/世界背景/红色 Principled 材质，设引擎。
  - `blender_render.py` CLI（`--` 之后）：`--usd --out-dir --res WxH --engine --samples`
- **注意**：这两个文件跑在 Blender 内置 py3.13，**不能 import `xpbd`**；只用 `bpy`/`mathutils`。

- [ ] **Step 1: 实现 `blender/look.py`**

```python
"""M0 最小 look：只为看清网格在动。M2 会替换内脏，但保持 setup() 签名。"""
import bpy
import mathutils


def setup(scene, engine: str = "BLENDER_EEVEE", samples: int = 16) -> None:
    scene.render.engine = engine
    if engine == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.device = "CPU"
    elif hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = samples

    # 相机：放在 -Y、抬高，瞄准原点
    cam_data = bpy.data.cameras.new("Cam")
    cam = bpy.data.objects.new("Cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    loc = mathutils.Vector((0.0, -4.0, 2.2))
    cam.location = loc
    cam.rotation_euler = (mathutils.Vector((0, 0, 0)) - loc).to_track_quat("-Z", "Y").to_euler()

    # 太阳光
    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_data.energy = 3.0
    sun = bpy.data.objects.new("Sun", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_euler = (0.6, 0.2, 0.3)

    # 世界背景
    world = bpy.data.worlds.new("W")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.05, 0.06, 1.0)

    # 简单材质贴到所有导入的 mesh
    mat = bpy.data.materials.new("Cloth")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.8, 0.2, 0.2, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.7
    for ob in scene.objects:
        if ob.type == "MESH":
            ob.data.materials.clear()
            ob.data.materials.append(mat)
```

- [ ] **Step 2: 实现 `blender/blender_render.py`**

```python
"""无头：导入 cache.usd → 套 look → 渲 PNG 帧。在 Blender 内运行。"""
import argparse
import os
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import look  # noqa: E402  同目录


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--res", default="480x360")
    ap.add_argument("--engine", default="BLENDER_EEVEE")
    ap.add_argument("--samples", type=int, default=16)
    return ap.parse_args(argv)


def main():
    args = parse_args()

    # 从空场景开始
    bpy.ops.wm.read_factory_settings(use_empty=True)

    bpy.ops.wm.usd_import(
        filepath=args.usd,
        import_cameras=False,
        import_lights=False,
        import_materials=False,
        read_mesh_uvs=True,
        set_frame_range=True,   # 按 USD 自动设帧范围
    )

    scene = bpy.context.scene
    look.setup(scene, engine=args.engine, samples=args.samples)

    w, h = (int(v) for v in args.res.lower().split("x"))
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.image_settings.file_format = "PNG"

    frames_dir = Path(args.out_dir) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    # 末尾带分隔符 → Blender 自动追加 4 位帧号 0001.png ...
    scene.render.filepath = os.path.join(str(frames_dir), "")

    bpy.ops.render.render(animation=True)
    print(f"[blender_render] frames -> {frames_dir} "
          f"({scene.frame_start}..{scene.frame_end})")


main()
```

- [ ] **Step 3: 手动验证无头渲帧（先小批 3 帧）**

先产一份短缓存，再渲：
```bash
.venv/bin/python scripts/run_sim.py wave --frames 3
blender --background --python blender/blender_render.py -- \
  --usd out/wave/cache.usd --out-dir out/wave --res 320x240
ls -1 out/wave/frames/*.png | wc -l
```
Expected: `out/wave/frames/` 下有 3 个 png（`0001.png 0002.png 0003.png`），`wc -l` 输出 `3`。

- [ ] **Step 4: 验证帧确实在动（像素差）**

Run:
```bash
.venv/bin/python -c "
import pathlib
a = pathlib.Path('out/wave/frames/0001.png').read_bytes()
c = pathlib.Path('out/wave/frames/0003.png').read_bytes()
print('FRAMES_DIFFER:', a != c)"
```
Expected: `FRAMES_DIFFER: True`（几何在动 → 渲染输出逐帧不同；若 False 说明动画没生效，回查 `set_frame_range` 与 points 时间采样）。

- [ ] **Step 5: Commit**

```bash
git add blender/look.py blender/blender_render.py
git commit -m "feat: Blender 无头导入 USD + 最小 look + 渲 PNG 帧"
```

---

### Task 9: Blender 拼帧 stitch.py（VSE → mp4）

**Files:**
- Create: `blender/stitch.py`

**Interfaces:**
- Consumes: `out/<demo>/frames/*.png`
- Produces: `out/<demo>/<name>.mp4`
  - CLI（`--` 之后）：`--frames-dir --out-dir --name --fps`
- **注意**：跑在 Blender py3.13，只用 `bpy`。用自带 FFMPEG，无需系统 ffmpeg。

- [ ] **Step 1: 实现 `blender/stitch.py`**

```python
"""无头：把 PNG 帧序列经 VSE 编码成 mp4（Blender 自带 FFMPEG）。"""
import argparse
import glob
import os
import sys

import bpy


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--fps", type=float, default=24.0)
    return ap.parse_args(argv)


def main():
    args = parse_args()
    frames = sorted(glob.glob(os.path.join(args.frames_dir, "*.png")))
    if not frames:
        raise SystemExit(f"no png frames in {args.frames_dir}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    se = scene.sequence_editor_create()
    strip = se.strips.new_image(name="frames", filepath=frames[0],
                                channel=1, frame_start=1)
    for fp in frames[1:]:
        strip.elements.append(os.path.basename(fp))

    scene.frame_start = 1
    scene.frame_end = len(frames)
    scene.render.fps = max(1, int(round(args.fps)))
    scene.render.use_sequencer = True
    scene.render.image_settings.media_type = "VIDEO"   # Blender 5.x: 必须先设, 否则 file_format 枚举隐藏 FFMPEG
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"

    stem = os.path.join(args.out_dir, args.name)
    scene.render.filepath = stem
    bpy.ops.render.render(animation=True)

    # FFMPEG 容器输出会把帧范围拼进文件名，找出来重命名成 <name>.mp4
    final = stem + ".mp4"
    produced = sorted(glob.glob(stem + "*.mp4"))
    if produced and produced[0] != final:
        os.replace(produced[0], final)
    print(f"[stitch] wrote {final}")


main()
```

- [ ] **Step 2: 手动验证拼帧**

承接 Task 8 产出的 3 帧：
```bash
blender --background --python blender/stitch.py -- \
  --frames-dir out/wave/frames --out-dir out/wave --name wave --fps 24
ls -la out/wave/wave.mp4
```
Expected: `out/wave/wave.mp4` 存在且大小 > 0。

- [ ] **Step 3: Commit**

```bash
git add blender/stitch.py
git commit -m "feat: VSE 拼帧 stitch.py（Blender 自带 FFMPEG → mp4）"
```

---

### Task 10: render.py 入口

**Files:**
- Create: `scripts/render.py`
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `out/<demo>/cache.usd`；子进程调 `blender_render.py` 与 `stitch.py`
- Produces:
  ```python
  def main(argv: list[str] | None = None) -> Path   # 返回 mp4 路径（或 --no-video 时 frames 目录）
  # CLI: render.py <demo> [--res WxH] [--engine E] [--samples N] [--no-video]
  def read_fps_from_usd(path) -> float
  ```

- [ ] **Step 1: 写失败测试 `tests/test_render.py`**（只测纯函数 `read_fps_from_usd`，不跑 Blender）

```python
import render  # 经 conftest.py 加入 sys.path


def test_read_fps_from_usd(tmp_path):
    import run_sim
    usd = run_sim.main(["wave", "--frames", "2", "--out", str(tmp_path / "wave")])
    assert render.read_fps_from_usd(usd) == 24.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_render.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'render'`

- [ ] **Step 3: 实现 `scripts/render.py`**

```python
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLENDER = shutil.which("blender") or "/opt/homebrew/bin/blender"
BLENDER_DIR = ROOT / "blender"


def read_fps_from_usd(path) -> float:
    from pxr import Usd
    stage = Usd.Stage.Open(str(path))
    return float(stage.GetTimeCodesPerSecond() or 24.0)


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description="渲染 out/<demo>/cache.usd → mp4")
    ap.add_argument("demo")
    ap.add_argument("--res", default="480x360")
    ap.add_argument("--engine", default="BLENDER_EEVEE")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args(argv)

    out_dir = ROOT / "out" / args.demo
    usd = out_dir / "cache.usd"
    if not usd.exists():
        raise SystemExit(f"{usd} not found — 先跑 run_sim.py {args.demo}")

    # 1) 渲 PNG 帧
    subprocess.run([
        BLENDER, "--background", "--python", str(BLENDER_DIR / "blender_render.py"), "--",
        "--usd", str(usd), "--out-dir", str(out_dir),
        "--res", args.res, "--engine", args.engine, "--samples", str(args.samples),
    ], check=True)

    frames_dir = out_dir / "frames"
    if args.no_video:
        print(f"[render] frames at {frames_dir}")
        return frames_dir

    # 2) 拼 mp4：有 ffmpeg 用 ffmpeg，否则用 Blender 自带
    fps = read_fps_from_usd(usd)
    mp4 = out_dir / f"{args.demo}.mp4"
    if shutil.which("ffmpeg"):
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(fps),
            "-pattern_type", "glob", "-i", str(frames_dir / "*.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4),
        ], check=True)
    else:
        subprocess.run([
            BLENDER, "--background", "--python", str(BLENDER_DIR / "stitch.py"), "--",
            "--frames-dir", str(frames_dir), "--out-dir", str(out_dir),
            "--name", args.demo, "--fps", str(fps),
        ], check=True)

    print(f"[render] video at {mp4}")
    return mp4


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_render.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/render.py tests/test_render.py
git commit -m "feat: render 入口（调 Blender 渲帧 + 拼帧，ffmpeg 自动探测）"
```

---

### Task 11: M0 端到端验收

**Files:**
- Create: `tests/test_e2e_m0.py`

**Interfaces:**
- Consumes: `run_sim.main`, `render.main`
- Produces: M0 完成判据的自动化护栏（标记 `slow`，调 Blender）

- [ ] **Step 1: 写端到端测试 `tests/test_e2e_m0.py`**

```python
import shutil
import pytest

import run_sim
import render


@pytest.mark.slow
def test_m0_pipeline_end_to_end(tmp_path, monkeypatch):
    # 在 tmp 下跑，避免污染仓库 out/
    monkeypatch.chdir(tmp_path)
    # render.py 用模块级 ROOT 定位 out/ 与 blender/，这里把 ROOT 指到真仓库 blender/，out 指到 tmp
    repo_root = render.ROOT

    # 1) 模拟 → cache.usd（写到 tmp/out/wave）
    out_dir = tmp_path / "out" / "wave"
    usd = run_sim.main(["wave", "--frames", "8", "--out", str(out_dir)])
    assert usd.exists()

    # 2) 渲染：直接调 blender 子进程（复用 render 的常量），输出到 tmp/out/wave
    import subprocess
    subprocess.run([
        render.BLENDER, "--background", "--python",
        str(repo_root / "blender" / "blender_render.py"), "--",
        "--usd", str(usd), "--out-dir", str(out_dir), "--res", "240x180",
    ], check=True)
    frames = sorted((out_dir / "frames").glob("*.png"))
    assert len(frames) == 8

    # 3) 帧之间确有差异（几何在动）
    assert frames[0].read_bytes() != frames[-1].read_bytes()

    # 4) 拼帧成 mp4
    subprocess.run([
        render.BLENDER, "--background", "--python",
        str(repo_root / "blender" / "stitch.py"), "--",
        "--frames-dir", str(out_dir / "frames"), "--out-dir", str(out_dir),
        "--name", "wave", "--fps", "24",
    ], check=True)
    mp4 = out_dir / "wave.mp4"
    assert mp4.exists() and mp4.stat().st_size > 0
```

- [ ] **Step 2: 跑端到端测试**

Run: `.venv/bin/pytest tests/test_e2e_m0.py -v -m slow`
Expected: 1 passed（会启动 Blender 两次，约数十秒）。

- [ ] **Step 3: 跑全量单测确认没回归**

Run: `.venv/bin/pytest -v -m "not slow"`
Expected: 全绿（result/exporters/scenes/run_sim/render 的快测）。

- [ ] **Step 4: 真出片一次（人眼验收）**

Run:
```bash
.venv/bin/python scripts/run_sim.py wave
.venv/bin/python scripts/render.py wave
```
Expected: `out/wave/wave.mp4` 生成；播放能看到一张红色网格上正弦波从一侧行进到另一侧。**这就是 M0 的"完成"。**

- [ ] **Step 5: Commit**

```bash
git add tests/test_e2e_m0.py
git commit -m "test: M0 端到端验收（run_sim→render→mp4 + 动画像素差护栏）"
```

---

## Self-Review

**1. Spec coverage（对照 spec 各节）：**
- §2 契约 → Task 2 ✓；§3 M0 判据（`run_sim wave`→usd、`render wave`→会动的 mp4）→ Task 7/10/11 ✓
- §5A USD 编码（时间采样 points / 静态拓扑 / st / upAxis=Z / 时间码）→ Task 4 ✓
- §5B Blender 端（usd_import 不带相机灯材质 / 渲帧 / 拼帧 / ffmpeg 自动探测 / PNG 落盘）→ Task 8/9/10 ✓
- §5C CLI 契约 → Task 7/10 ✓（M0 子集；`--preview`/`--substeps` 属 M1，未实现，符合范围）
- §7 模块结构 → 本计划文件表覆盖 M0 相关文件 ✓（`solver/*`、`scenes/cloth.py` 的拓扑/着色、`forces.py`、`preview/ggui.py` 属 M1/M2，M0 不建）
- §8 环境（py3.12 venv / Blender 子进程 / Metal / 不强制 ffmpeg）→ Task 1/8/9/10 ✓
- §9 测试（exporter 往返、集成冒烟）→ Task 3/4/7/11 ✓（约束/拓扑/对称/能量等不变量属 M1）
- §11 风险（USD 动画无头往返、taichi wheel）→ Task 8 Step4（像素差）、Task 1 Step3-4 ✓

**2. Placeholder scan：** 无 TBD/TODO；每个代码步骤含完整代码；每个命令含期望输出。✓

**3. Type consistency：** `SimResult(positions,faces,uvs,fps,name)` 在 Task 2/3/4/6/7 一致；`build_scene(name,*,frames,res)`、`build_grid(nx,ny,width,height)`、`generate_wave(...,name,frames,fps,amplitude,wavelength,speed)`、`export_usd(result,path)→Path`、`save_npz/load_npz`、`run_sim.main(argv)→Path`、`render.main(argv)→Path`、`read_fps_from_usd(path)→float`、`look.setup(scene,engine,samples)` 跨 task 引用一致。✓

**范围说明：** 本计划只覆盖 **M0 行走骨架**。M1（真 XPBD 求解器 + flag）、M2（look 定稿）落地后各自成计划。

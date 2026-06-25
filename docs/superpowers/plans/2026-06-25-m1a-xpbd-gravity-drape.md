# M1a · XPBD 求解器：测试骨架 + 重力垂坠 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 small-steps XPBD 求解器的第一段——`ClothSolver`（预测/积分 + 距离约束 + 固定约束，Jacobi），让钉住的布在重力下稳定垂坠，产出 `SimResult` 接入 M0 管线。**先把 Taichi 测试骨架这个基础风险落地。**

**Architecture:** 一个 `@ti.data_oriented` 的 `ClothSolver` 类：`__init__` 分配 `ti.field`，kernel 当方法。每 substep = predict → clear → solve_distance(Jacobi 累加) → apply_dx(平均+欠松弛) → update_velocity。求解器模块自己**不**调 `ti.init`（测试用 cpu 夹具、flag 入口用 metal）。`flag` 场景把网格映射进竖直平面、钉住 hoist 边、跑求解器 → `SimResult` → 复用 M0 的 USD/渲染。

**Tech Stack:** conda `taichi` 环境（py3.10.20 · taichi 1.7.4 · numpy 2.2.6 · usd-core · pytest 9.1.1）· Metal/CPU 后端

## Global Constraints

逐条来自 spec，每个 task 隐含遵守：

- **环境 = conda `taichi`**（**不要**用 M0 的 `.venv`）。所有命令用：`/opt/homebrew/Caskroom/miniconda/base/envs/taichi/bin/python`（下文简记 `$PY`）。测试：`$PY -m pytest`。
- **求解器模块永不调 `ti.init`**。单测经 `ti_cpu` 夹具调 `ti.init(arch=ti.cpu)`（function 级，每测试全新 runtime → 全新 field，已实测可行）；`flag` 入口调 `ti.init(arch=ti.metal)`。
- **small-steps XPBD**：`h = (1/fps)/substeps`；compliance `α̃ = α/h²`；Jacobi **1 sweep/substep**，`dx` 累加（Taichi 的 `+=` 在并行 for 里**自动原子化**，无需显式 atomic_add），`apply` 按 `count` 平均 + 欠松弛 `ω`。
- **距离约束投影**（单 sweep，λ=0）：边 `(i,j)`，`d=x_i−x_j, ln=|d|, n=d/ln`，`Δλ=−(ln−rest)/(w_i+w_j+α̃)`，`dx[i]+=w_i·Δλ·n; dx[j]+=−w_j·Δλ·n`。
- **Z-up，gravity=(0,0,−9.8)**。钉住点 `w=0`：predict 不动它、apply 跳过它 → 全程不动。
- **M1a 只做距离 + 固定**。弯曲（距离式）是 M1b、风是 M1c——**本计划不做**。无图着色、无二面角。
- **SimResult 契约不变**：`positions f32[F,N,3]`、`faces i32[M,3]`、`uvs f32[N,2]`、`fps`、`name`。
- **flag 朝向**：宽→X、高→Z、y=0，居中原点；**hoist = x=min 那条竖边（沿 Z）整条钉住**。
- **起步参数**：substeps=20、α_dist=1e-7、ω=0.5、damping=0.01、flag 尺寸 1.5×1.0 m、总质量 0.2 kg（w=N/mass）。
- **dtype**：positions/uvs=float32，faces/edges=int32。

**已实测确认的 Taichi 1.7.4 模式**（直接用，勿改）：
- `ti.init(arch=ti.cpu)` 每测试一次 + 重新分配不同尺寸 field → 可行（测试骨架）。
- `@ti.data_oriented` 类：field 在 `__init__`，`@ti.kernel` 方法访问 `self.x` 等；`self` 上的 Python 标量（`self.h/alpha/omega`）作编译期常量。
- `d.norm(1e-12)` 安全归一；`field.from_numpy(...)`/`field.to_numpy()` 往返。
- 距离投影 20 substep 把 1.5 收敛到 ~1.0；钉点零位移。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `tests/conftest.py` | **改**：加 `ti_cpu` 夹具（function 级 `ti.init(ti.cpu)`） |
| `src/xpbd/solver/__init__.py` | 新建（空） |
| `src/xpbd/solver/topology.py` | `build_edges(faces)` 提取唯一边（结构+剪切），纯 numpy |
| `src/xpbd/solver/cloth.py` | `ClothSolver` 类：field + predict/solve_distance/apply_dx/update_velocity + substep/run |
| `src/xpbd/scenes/flag.py` | `build_flag(...)`：竖直平面映射 + hoist 钉点 + 跑求解器 → SimResult |
| `src/xpbd/scenes/presets.py` | **改**：`build_scene` 分派 `flag`，加 `substeps` 参数 |
| `scripts/run_sim.py` | **改**：加 `--substeps` |
| `tests/test_taichi_harness.py` | 测试骨架冒烟 |
| `tests/test_topology.py` | 边提取正确 |
| `tests/test_integrator.py` | predict/update_velocity + 钉点不动 |
| `tests/test_constraints.py` | 距离约束不变量 |
| `tests/test_solver_stability.py` | 小布垂坠：无 NaN、拉伸有界、确实下沉 |
| `tests/test_flag_scene.py` | run_sim flag → cache.usd + hoist 不动 |

---

### Task 1: Taichi 测试骨架 + solver 包

**Files:**
- Modify: `tests/conftest.py`
- Create: `src/xpbd/solver/__init__.py`（空）
- Test: `tests/test_taichi_harness.py`

**Interfaces:**
- Consumes: 无
- Produces: pytest 夹具 `ti_cpu`（function 级，先 `ti.init(arch=ti.cpu)` 再 yield）。后续所有求解器测试都请求它。

- [ ] **Step 1: 加 `ti_cpu` 夹具到 `tests/conftest.py`**（在现有内容后追加）

```python
import pytest
import taichi as ti


@pytest.fixture
def ti_cpu():
    """每个用例全新 ti.cpu runtime（→ 全新 field），确定性、无 GPU。"""
    ti.init(arch=ti.cpu)
    yield
```

- [ ] **Step 2: 建空包文件**

`src/xpbd/solver/__init__.py` —— 空文件。

- [ ] **Step 3: 写冒烟测试 `tests/test_taichi_harness.py`**

```python
import numpy as np
import taichi as ti


def test_cpu_kernel_runs(ti_cpu):
    f = ti.field(ti.f32, shape=4)

    @ti.kernel
    def fill():
        for i in f:
            f[i] = ti.f32(i) * 2.0

    fill()
    np.testing.assert_array_equal(f.to_numpy(), np.array([0, 2, 4, 6], dtype=np.float32))


def test_fresh_runtime_each_test(ti_cpu):
    # 不同尺寸 field 也能分配（证明 ti.init-per-test 隔离）
    g = ti.Vector.field(3, ti.f32, shape=7)
    g.fill(1.0)
    assert g.to_numpy().shape == (7, 3)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `$PY -m pytest tests/test_taichi_harness.py -v`
（`$PY` = `/opt/homebrew/Caskroom/miniconda/base/envs/taichi/bin/python`）
Expected: 2 passed（每个测试前会打印一行 `[Taichi] Starting on arch=cpu`）

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py src/xpbd/solver/__init__.py tests/test_taichi_harness.py
git commit -m "test: Taichi ti.cpu 测试夹具 + solver 包骨架"
```

---

### Task 2: topology.build_edges

**Files:**
- Create: `src/xpbd/solver/topology.py`
- Test: `tests/test_topology.py`

**Interfaces:**
- Consumes: `build_grid` 产出的 `faces`（来自 `xpbd.scenes.cloth`，已存在）
- Produces: `build_edges(faces: np.ndarray) -> np.ndarray`（int32 `[E,2]`，每行 `i<j`，去重排序；含结构边 + 每个 quad 的剪切对角边）

- [ ] **Step 1: 写失败测试 `tests/test_topology.py`**

```python
import numpy as np
from xpbd.scenes.cloth import build_grid
from xpbd.solver.topology import build_edges


def test_edges_unique_sorted_valid():
    _, faces, _ = build_grid(4, 3, 2.0, 1.5)   # nx=4, ny=3
    edges = build_edges(faces)
    assert edges.dtype == np.int32 and edges.shape[1] == 2
    # 每行 i<j
    assert np.all(edges[:, 0] < edges[:, 1])
    # 无重复
    assert len(np.unique(edges, axis=0)) == len(edges)
    # 索引合法
    assert edges.min() >= 0 and edges.max() < 4 * 3


def test_edge_count_matches_grid_formula():
    nx, ny = 4, 3
    _, faces, _ = build_grid(nx, ny, 2.0, 1.5)
    edges = build_edges(faces)
    # 结构: 水平 (nx-1)*ny + 竖直 nx*(ny-1); 剪切对角: (nx-1)*(ny-1)
    expected = (nx - 1) * ny + nx * (ny - 1) + (nx - 1) * (ny - 1)
    assert len(edges) == expected
```

- [ ] **Step 2: 跑测试确认失败**

Run: `$PY -m pytest tests/test_topology.py -v`
Expected: FAIL，`No module named 'xpbd.solver.topology'`

- [ ] **Step 3: 实现 `src/xpbd/solver/topology.py`**

```python
from __future__ import annotations

import numpy as np


def build_edges(faces: np.ndarray) -> np.ndarray:
    """从三角面提取唯一无向边：结构边 + 每个 quad 的剪切对角边。

    返回 int32 [E,2]，每行 i<j，去重并按字典序排序。
    （build_grid 把每个 quad 切成 2 个三角，对角边作为三角边自然包含进来。）
    """
    faces = np.asarray(faces, dtype=np.int64)
    e = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    e = np.sort(e, axis=1)          # 每条边 i<j
    e = np.unique(e, axis=0)        # 去重 + 排序
    return e.astype(np.int32)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `$PY -m pytest tests/test_topology.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/xpbd/solver/topology.py tests/test_topology.py
git commit -m "feat: topology.build_edges 提取结构+剪切边"
```

---

### Task 3: ClothSolver 积分核心（无约束）

**Files:**
- Create: `src/xpbd/solver/cloth.py`
- Test: `tests/test_integrator.py`

**Interfaces:**
- Consumes: 无（纯 Taichi）
- Produces:
  ```python
  @ti.data_oriented
  class ClothSolver:
      def __init__(self, positions0, edges, inv_mass, *, fps=24.0, substeps=20,
                   alpha_dist=1e-7, omega=0.5, damping=0.01, gravity=(0.,0.,-9.8))
      def predict(self)            # @ti.kernel
      def update_velocity(self)    # @ti.kernel
      def substep(self)            # 本任务: 仅 predict + update_velocity
      def run(self, frames) -> np.ndarray  # [F,N,3] float32
  ```
  本任务 `substep` 暂不含约束（Task 4 补）。`rest` 由 `positions0` + `edges` 在 `__init__` 算好。

- [ ] **Step 1: 写失败测试 `tests/test_integrator.py`**

```python
import numpy as np
from xpbd.solver.cloth import ClothSolver


def _two_particles(pinned0):
    pos = np.array([[0, 0, 1.0], [1, 0, 1.0]], dtype=np.float32)
    edges = np.array([[0, 1]], dtype=np.int32)
    w = np.array([0.0 if pinned0 else 1.0, 1.0], dtype=np.float32)
    return pos, edges, w


def test_free_particle_falls_under_gravity(ti_cpu):
    pos, edges, w = _two_particles(pinned0=True)  # 粒子0钉住, 粒子1自由
    s = ClothSolver(pos, edges, w, substeps=1, damping=0.0)
    # 只跑积分(本任务 substep 无约束): 一帧=1 substep
    out = s.run(1)
    # 自由粒子1 应在 z 上下落 (z 减小)
    assert out[0, 1, 2] < 1.0
    # 钉住粒子0 不动
    np.testing.assert_allclose(out[0, 0], pos[0], atol=1e-6)


def test_update_velocity_matches_displacement(ti_cpu):
    pos, edges, w = _two_particles(pinned0=True)
    s = ClothSolver(pos, edges, w, substeps=1, damping=0.0)
    s.predict()
    s.update_velocity()
    v = s.v.to_numpy()
    # 钉点速度保持 0
    np.testing.assert_allclose(v[0], [0, 0, 0], atol=1e-6)
    # 自由点速度为负 z (下落)
    assert v[1, 2] < 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `$PY -m pytest tests/test_integrator.py -v`
Expected: FAIL，`No module named 'xpbd.solver.cloth'`

- [ ] **Step 3: 实现 `src/xpbd/solver/cloth.py`**（含本任务用到的成员；Task 4 再补约束 kernel）

```python
from __future__ import annotations

import numpy as np
import taichi as ti


@ti.data_oriented
class ClothSolver:
    """small-steps XPBD 布料求解器（M1a: 距离约束 + 固定；调用方负责 ti.init）。"""

    def __init__(self, positions0, edges, inv_mass, *, fps=24.0, substeps=20,
                 alpha_dist=1e-7, omega=0.5, damping=0.01, gravity=(0.0, 0.0, -9.8)):
        positions0 = np.ascontiguousarray(positions0, dtype=np.float32)
        edges = np.ascontiguousarray(edges, dtype=np.int32)
        inv_mass = np.ascontiguousarray(inv_mass, dtype=np.float32)

        self.N = int(positions0.shape[0])
        self.E = int(edges.shape[0])
        self.substeps = int(substeps)
        self.h = (1.0 / fps) / substeps
        self.alpha_dist = float(alpha_dist)
        self.omega = float(omega)
        self.damping = float(damping)

        self.x = ti.Vector.field(3, ti.f32, shape=self.N)
        self.v = ti.Vector.field(3, ti.f32, shape=self.N)
        self.x_prev = ti.Vector.field(3, ti.f32, shape=self.N)
        self.w = ti.field(ti.f32, shape=self.N)
        self.dx = ti.Vector.field(3, ti.f32, shape=self.N)
        self.cnt = ti.field(ti.i32, shape=self.N)
        self.edge = ti.Vector.field(2, ti.i32, shape=self.E)
        self.rest = ti.field(ti.f32, shape=self.E)
        self.g = ti.Vector(list(gravity))

        self.x.from_numpy(positions0)
        self.v.fill(0.0)
        self.w.from_numpy(inv_mass)
        self.edge.from_numpy(edges)
        rest = np.linalg.norm(positions0[edges[:, 0]] - positions0[edges[:, 1]], axis=1)
        self.rest.from_numpy(rest.astype(np.float32))

    @ti.kernel
    def predict(self):
        for i in self.x:
            self.x_prev[i] = self.x[i]
            if self.w[i] > 0:
                self.v[i] += self.h * self.g
                self.x[i] += self.h * self.v[i]

    @ti.kernel
    def update_velocity(self):
        for i in self.x:
            if self.w[i] > 0:
                self.v[i] = (self.x[i] - self.x_prev[i]) / self.h
                self.v[i] *= (1.0 - self.damping)

    def substep(self):
        self.predict()
        self.update_velocity()

    def run(self, frames: int) -> np.ndarray:
        out = np.empty((frames, self.N, 3), dtype=np.float32)
        for f in range(frames):
            for _ in range(self.substeps):
                self.substep()
            out[f] = self.x.to_numpy()
        return out
```

- [ ] **Step 4: 跑测试确认通过**

Run: `$PY -m pytest tests/test_integrator.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/xpbd/solver/cloth.py tests/test_integrator.py
git commit -m "feat: ClothSolver 积分核心 (predict/update_velocity/run)"
```

---

### Task 4: 距离约束（solve_distance + apply_dx）

**Files:**
- Modify: `src/xpbd/solver/cloth.py`（加 3 个 kernel，substep 接入约束）
- Test: `tests/test_constraints.py`

**Interfaces:**
- Consumes: Task 3 的 `ClothSolver`
- Produces：在 `ClothSolver` 上新增
  ```python
  def _clear(self)          # @ti.kernel: dx=0, cnt=0
  def solve_distance(self)  # @ti.kernel: Jacobi 累加距离修正
  def apply_dx(self)        # @ti.kernel: x += ω·dx/cnt (自由点)
  ```
  `substep` 改为 predict → _clear → solve_distance → apply_dx → update_velocity。

- [ ] **Step 1: 写失败测试 `tests/test_constraints.py`**

```python
import numpy as np
from xpbd.solver.cloth import ClothSolver


def test_distance_restores_rest_length(ti_cpu):
    # 两自由粒子 rest=1.0, 拉到 1.5, 无重力 → 收敛回 ~1.0
    pos = np.array([[0, 0, 0], [1.5, 0, 0]], dtype=np.float32)
    edges = np.array([[0, 1]], dtype=np.int32)
    w = np.array([1.0, 1.0], dtype=np.float32)
    s = ClothSolver(pos, edges, w, substeps=20, damping=0.0, gravity=(0, 0, 0))
    # 强制 rest=1.0 (初始距离是 1.5, 这里手动设)
    s.rest.from_numpy(np.array([1.0], dtype=np.float32))
    out = s.run(1)
    d = np.linalg.norm(out[0, 0] - out[0, 1])
    assert abs(d - 1.0) < 0.05


def test_pinned_partner_does_not_move(ti_cpu):
    pos = np.array([[0, 0, 0], [1.5, 0, 0]], dtype=np.float32)
    edges = np.array([[0, 1]], dtype=np.int32)
    w = np.array([0.0, 1.0], dtype=np.float32)  # 粒子0钉住
    s = ClothSolver(pos, edges, w, substeps=20, damping=0.0, gravity=(0, 0, 0))
    s.rest.from_numpy(np.array([1.0], dtype=np.float32))
    out = s.run(1)
    np.testing.assert_allclose(out[0, 0], [0, 0, 0], atol=1e-6)        # 钉点不动
    assert abs(np.linalg.norm(out[0, 0] - out[0, 1]) - 1.0) < 0.05     # 距离→rest


def test_rest_configuration_stationary(ti_cpu):
    # 平铺、约束在静止长度、无重力 → 一帧后位置基本不变 (抓符号/构型错)
    from xpbd.scenes.cloth import build_grid
    from xpbd.solver.topology import build_edges
    verts, faces, _ = build_grid(4, 3, 2.0, 1.5)
    edges = build_edges(faces)
    w = np.ones(verts.shape[0], dtype=np.float32)
    s = ClothSolver(verts, edges, w, substeps=5, damping=0.0, gravity=(0, 0, 0))
    out = s.run(1)
    np.testing.assert_allclose(out[0], verts, atol=1e-5)


def test_larger_compliance_is_softer(ti_cpu):
    # α 越大 → 一次投影修正越小
    pos = np.array([[0, 0, 0], [1.5, 0, 0]], dtype=np.float32)
    edges = np.array([[0, 1]], dtype=np.int32)
    w = np.array([1.0, 1.0], dtype=np.float32)

    def corrected_distance(alpha):
        s = ClothSolver(pos, edges, w, substeps=1, damping=0.0,
                        gravity=(0, 0, 0), alpha_dist=alpha, omega=1.0)
        s.rest.from_numpy(np.array([1.0], dtype=np.float32))
        s.predict(); s._clear(); s.solve_distance(); s.apply_dx()
        return np.linalg.norm(s.x.to_numpy()[0] - s.x.to_numpy()[1])

    stiff = corrected_distance(1e-9)   # 几乎硬 → 接近 1.0
    soft = corrected_distance(1e-1)    # 很软 → 仍接近 1.5
    assert abs(stiff - 1.0) < abs(soft - 1.0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `$PY -m pytest tests/test_constraints.py -v`
Expected: FAIL，`AttributeError: 'ClothSolver' object has no attribute 'solve_distance'`（或 `_clear`）

- [ ] **Step 3: 给 `ClothSolver` 加约束 kernel + 改 substep**（修改 `src/xpbd/solver/cloth.py`）

在 `update_velocity` 之后、`substep` 之前插入三个 kernel：

```python
    @ti.kernel
    def _clear(self):
        for i in self.x:
            self.dx[i] = ti.Vector([0.0, 0.0, 0.0])
            self.cnt[i] = 0

    @ti.kernel
    def solve_distance(self):
        a = self.alpha_dist / (self.h * self.h)
        for e in self.edge:
            i, j = self.edge[e][0], self.edge[e][1]
            d = self.x[i] - self.x[j]
            ln = d.norm(1e-12)
            n = d / ln
            dl = -(ln - self.rest[e]) / (self.w[i] + self.w[j] + a)
            self.dx[i] += self.w[i] * dl * n       # Taichi += 自动原子化
            self.dx[j] += -self.w[j] * dl * n
            self.cnt[i] += 1
            self.cnt[j] += 1

    @ti.kernel
    def apply_dx(self):
        for i in self.x:
            if self.w[i] > 0 and self.cnt[i] > 0:
                self.x[i] += self.omega * self.dx[i] / self.cnt[i]
```

并把 `substep` 改成：

```python
    def substep(self):
        self.predict()
        self._clear()
        self.solve_distance()
        self.apply_dx()
        self.update_velocity()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `$PY -m pytest tests/test_constraints.py tests/test_integrator.py -v`
Expected: 全部 passed（注意 integrator 的 `test_free_particle_falls` 现在 substep 含约束，但那两粒子初始即在 rest=1 距离、重力下仍会整体下落，断言仍成立）

- [ ] **Step 5: Commit**

```bash
git add src/xpbd/solver/cloth.py tests/test_constraints.py
git commit -m "feat: 距离约束 Jacobi 投影 (solve_distance/apply_dx)"
```

---

### Task 5: 稳定性护栏（小布垂坠）

**Files:**
- Test: `tests/test_solver_stability.py`

**Interfaces:**
- Consumes: `ClothSolver`, `build_grid`, `build_edges`
- Produces: 无新代码——纯护栏测试，验证组装好的求解器在真小布上稳定。

- [ ] **Step 1: 写测试 `tests/test_solver_stability.py`**

```python
import numpy as np
from xpbd.scenes.cloth import build_grid
from xpbd.solver.topology import build_edges
from xpbd.solver.cloth import ClothSolver


def _pinned_small_cloth():
    # 16x10 网格, 竖直平面 (宽→x, 高→z), 钉住 x=min 那条竖边
    nx, ny = 16, 10
    verts, faces, _ = build_grid(nx, ny, 1.5, 1.0)
    pos = np.zeros_like(verts)
    pos[:, 0] = verts[:, 0]
    pos[:, 2] = verts[:, 1]
    edges = build_edges(faces)
    w = np.full(nx * ny, (nx * ny) / 0.2, dtype=np.float32)
    w[np.isclose(pos[:, 0], pos[:, 0].min())] = 0.0
    return pos, edges, w


def test_drape_is_stable_and_sinks(ti_cpu):
    pos, edges, w = _pinned_small_cloth()
    s = ClothSolver(pos, edges, w, substeps=20)
    out = s.run(40)  # 40 帧重力垂坠
    last = out[-1]
    # 无 NaN/Inf
    assert np.all(np.isfinite(last))
    # 确实下沉: 自由点平均 z 低于初始
    free = w > 0
    assert last[free, 2].mean() < pos[free, 2].mean() - 0.01
    # 边拉伸有界: 没有橡皮筋拉长 (最大边长 < 静止长度的 1.5 倍)
    rest = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)
    cur = np.linalg.norm(last[edges[:, 0]] - last[edges[:, 1]], axis=1)
    assert (cur / rest).max() < 1.5


def test_pinned_edge_stays_fixed(ti_cpu):
    pos, edges, w = _pinned_small_cloth()
    s = ClothSolver(pos, edges, w, substeps=20)
    out = s.run(40)
    pinned = w == 0
    np.testing.assert_allclose(out[-1][pinned], pos[pinned], atol=1e-5)
```

- [ ] **Step 2: 跑测试**

Run: `$PY -m pytest tests/test_solver_stability.py -v`
Expected: 2 passed（若 `test_drape_is_stable_and_sinks` 因 Jacobi 太软导致拉伸>1.5 而失败，是真信号——调高 substeps 或调低 α_dist 直到通过，并把调好的值记进报告）

- [ ] **Step 3: Commit**

```bash
git add tests/test_solver_stability.py
git commit -m "test: 求解器稳定性护栏 (垂坠/无NaN/拉伸有界/钉边不动)"
```

---

### Task 6: flag 场景 + 接入 run_sim

**Files:**
- Create: `src/xpbd/scenes/flag.py`
- Modify: `src/xpbd/scenes/presets.py`（`build_scene` 分派 `flag` + 加 `substeps`）
- Modify: `scripts/run_sim.py`（加 `--substeps`）
- Test: `tests/test_flag_scene.py`

**Interfaces:**
- Consumes: `build_grid`, `build_edges`, `ClothSolver`, `SimResult`
- Produces:
  ```python
  # flag.py
  def build_flag(*, res=64, frames=250, fps=24.0, substeps=20) -> SimResult
  # presets.py
  build_scene(name, *, frames=None, res=None, substeps=None) -> SimResult  # 分派 wave/flag
  ```

- [ ] **Step 1: 写失败测试 `tests/test_flag_scene.py`**

```python
import numpy as np
from xpbd.scenes.flag import build_flag
from xpbd.io.result import SimResult


def test_build_flag_returns_simresult():
    r = build_flag(res=16, frames=4, substeps=10)
    assert isinstance(r, SimResult)
    assert r.name == "flag"
    assert r.num_frames == 4
    # 竖直朝向: y 基本为 0, z 有展开 (高度方向)
    p0 = r.positions[0]
    assert np.allclose(p0[:, 1], 0.0, atol=1e-6)
    assert p0[:, 2].ptp() > 0.5            # z 方向铺开 (高度~1.0)


def test_flag_hoist_edge_pinned():
    r = build_flag(res=16, frames=6, substeps=10)
    p0, plast = r.positions[0], r.positions[-1]
    # x=min 那条竖边的点全程不动
    xmin = p0[:, 0].min()
    hoist = np.isclose(p0[:, 0], xmin)
    np.testing.assert_allclose(plast[hoist], p0[hoist], atol=1e-5)
    # 非钉点确实动了 (重力)
    assert not np.allclose(plast[~hoist], p0[~hoist], atol=1e-4)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `$PY -m pytest tests/test_flag_scene.py -v`
Expected: FAIL，`No module named 'xpbd.scenes.flag'`

- [ ] **Step 3: 实现 `src/xpbd/scenes/flag.py`**

```python
from __future__ import annotations

import numpy as np
import taichi as ti

from xpbd.io.result import SimResult
from xpbd.scenes.cloth import build_grid
from xpbd.solver.cloth import ClothSolver
from xpbd.solver.topology import build_edges


def build_flag(*, res: int = 64, frames: int = 250, fps: float = 24.0,
               substeps: int = 20, width: float = 1.5, height: float = 1.0,
               mass: float = 0.2) -> SimResult:
    """旗子场景：竖直平面网格 + hoist 竖边钉住 + 重力垂坠（M1a 无风）。"""
    nx = int(res)
    ny = max(2, round(res * (height / width)))     # 3:2 → ny≈res*2/3

    verts_xy, faces, uvs = build_grid(nx, ny, width, height)
    # 映射进竖直平面: 宽→X, 高→Z, y=0
    pos = np.zeros_like(verts_xy)
    pos[:, 0] = verts_xy[:, 0]
    pos[:, 2] = verts_xy[:, 1]

    n = nx * ny
    inv_mass = np.full(n, n / mass, dtype=np.float32)   # w = 1/m = N/mass
    inv_mass[np.isclose(pos[:, 0], pos[:, 0].min())] = 0.0   # 钉 hoist 竖边

    edges = build_edges(faces)

    ti.init(arch=ti.metal)     # flag 入口负责 init (求解器模块自己不 init)
    solver = ClothSolver(pos, edges, inv_mass, fps=fps, substeps=substeps)
    positions = solver.run(frames)

    return SimResult(positions=positions, faces=faces, uvs=uvs, fps=fps, name="flag")
```

- [ ] **Step 4: 改 `build_scene` 分派 flag**（`src/xpbd/scenes/presets.py`）

把 `build_scene` 改为（保留原 wave 分支）：

```python
def build_scene(name: str, *, frames: int | None = None, res: int | None = None,
                substeps: int | None = None) -> SimResult:
    if name == "flag":
        from xpbd.scenes.flag import build_flag
        kw = {}
        if frames is not None:
            kw["frames"] = frames
        if res is not None:
            kw["res"] = res
        if substeps is not None:
            kw["substeps"] = substeps
        return build_flag(**kw)

    if name not in PRESETS:
        raise KeyError(f"unknown scene '{name}'; known: {sorted(PRESETS) + ['flag']}")
    # —— 以下为原 wave 路径，保持不变 ——
    p = dict(PRESETS[name])
    if frames is not None:
        p["frames"] = frames
    if res is not None:
        if p["nx"] >= p["ny"]:
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

- [ ] **Step 5: 给 run_sim 加 `--substeps`**（`scripts/run_sim.py`）

在 argparse 里加一行（`--res` 之后）：
```python
    ap.add_argument("--substeps", type=int, default=None, help="每帧 substep 数 (求解器场景)")
```
并把 `build_scene` 调用改为：
```python
    result = build_scene(args.scene, frames=args.frames, res=args.res, substeps=args.substeps)
```

- [ ] **Step 6: 跑测试确认通过 + 全量回归**

Run: `$PY -m pytest tests/test_flag_scene.py -v`
Expected: 2 passed
Run: `$PY -m pytest -m "not slow" -q`
Expected: 全绿（M0 的 14 个 + M1a 新增，无回归）

- [ ] **Step 7: 手动出片验收（人眼看垂坠）**

Run:
```bash
$PY scripts/run_sim.py flag --frames 60 --res 48
$PY scripts/render.py flag
```
Expected: `out/flag/flag.mp4` 生成；播放能看到一块布从左侧竖边垂下、在重力下下沉并稳定（**还不会飘——风是 M1c**）。out/ 是 gitignored，不要 git-add。

- [ ] **Step 8: Commit**

```bash
git add src/xpbd/scenes/flag.py src/xpbd/scenes/presets.py scripts/run_sim.py tests/test_flag_scene.py
git commit -m "feat: flag 场景 (竖直+hoist钉住+重力垂坠) 接入 run_sim --substeps"
```

---

## Self-Review

**1. Spec coverage（对照 M1 spec）：**
- §1 增量节奏 M1a（测试骨架+积分+距离+固定→垂坠）→ Task 1–6 ✓；M1b/M1c/M1d 不在本计划（按 spec 分块）✓
- §2 算法（substep 循环、Jacobi、α̃=α/h²、距离投影、起步参数）→ Task 3/4 + Global Constraints ✓
- §3 数据结构（field、约束表、topology）+ 测试 arch（ti.cpu、模块不 init、夹具）→ Task 1/2/3 ✓
- §4 flag 场景（竖直朝向、hoist 钉、接 SimResult、run_sim --substeps）→ Task 6 ✓
- §6 测试不变量（topology/integrator/距离约束/静止构型/稳定性护栏）→ Task 2/3/4/5 ✓（对称、弯曲共面属 M1b 弯曲后）
- §8 环境（conda taichi、$PY、不 .venv）→ Global Constraints + 全 task 命令 ✓
- §9 风险（Taichi 生命周期）→ Task 1 首先消化 ✓

**2. Placeholder scan：** 无 TBD/TODO；每个代码步骤含完整 proven 代码；命令含期望输出。✓

**3. Type consistency：** `ClothSolver(positions0, edges, inv_mass, *, fps, substeps, alpha_dist, omega, damping, gravity)`、`.predict/_clear/solve_distance/apply_dx/update_velocity/substep/run`、`build_edges(faces)->[E,2]`、`build_flag(*, res, frames, fps, substeps)->SimResult`、`build_scene(name,*,frames,res,substeps)` 跨 task 一致。✓

**范围说明：** 本计划只覆盖 **M1a**。M1b（距离式弯曲）、M1c（风→飘）、M1d（气动 stretch）落地后各自成计划。

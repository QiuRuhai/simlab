# M1b · 距离式弯曲 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 `ClothSolver` 加**距离式弯曲**约束——每条内部边的两个对顶点之间一条距离约束，抗折叠，让布有"骨感"而非软网。flag 接入后垂坠应更挺。

**Architecture:** 弯曲复用已验证的 Jacobi 距离投影：新增 `topology.build_bend_pairs(faces)`（每条被 2 个三角共享的内部边 → 两对顶点），`ClothSolver` 增加 `bend_pair`/`bend_rest` field + `solve_bending()` kernel（与 `solve_distance` 同式，柔度 `α_bend`），substep 在 `solve_distance` 后、`apply_dx` 前插入 `solve_bending`（两者累加进**同一个 `dx`**，一次 apply）。`bend_pairs` 在 `ClothSolver` 里**可选**（默认无弯曲，不破坏 M1a 调用）。

**Tech Stack:** conda `taichi` 环境（py3.10.20 · taichi 1.7.4 · numpy 2.2.6 · pytest）· Metal/CPU

## Global Constraints

- **环境 = conda `taichi`**（**不要** `.venv`）。命令用 `/opt/homebrew/Caskroom/miniconda/base/envs/taichi/bin/python`（记作 `$PY`）。测试 `$PY -m pytest`。
- **弯曲 = 距离约束**：对每条内部边的两对顶点 `(k,l)`，静止长度 = 初始距离，柔度 `α_bend`（起步 `1e-4`，比拉伸软）。投影公式与距离约束完全一致（`Δλ=−(ln−rest)/(w_k+w_l+α̃)`，`α̃=α_bend/h²`）。
- **累加进同一个 `dx`**：substep = `predict → _clear → solve_distance → solve_bending → apply_dx → update_velocity`。距离与弯曲修正都累加进 `dx`/`cnt`，最后一次 `apply_dx` 平均施加。
- **`bend_pairs` 可选**：`ClothSolver.__init__` 新增 `bend_pairs=None, alpha_bend=1e-4`。`None`/空 → `B=0`，substep 跳过 `solve_bending`（`if self.B > 0`）。**M1a 的既有调用（只传 positions0/edges/inv_mass）行为不变。**
- bend field 按 `shape=max(1, B)` 分配（避免 shape=0），但仅在 `B>0` 时调用 `solve_bending`。
- **Taichi 的 `+=` 在并行 for 里自动原子化**——不要加显式 atomic_add。
- 求解器模块**不调 `ti.init`**（测试用 `ti_cpu` 夹具）。dtype：positions/uvs float32，edges/faces/bend int32。
- **不破坏 M1a 测试**（distance/integrator/stability/flag 全绿）。

**已实测确认（探针）：** `build_bend_pairs(build_grid(4,3).faces)` → 13 对 = 内部边数；索引合法、对内顶点相异、静止长度>0。距离投影 + Jacobi 累加在 M1a 已验证（含 Metal 冒烟）。

---

## 文件结构

| 文件 | 改动 |
|---|---|
| `src/xpbd/solver/topology.py` | **加** `build_bend_pairs(faces)` |
| `src/xpbd/solver/cloth.py` | **改** `ClothSolver`：`__init__` 加 bend 参数/field，加 `solve_bending` kernel，substep 插入 |
| `src/xpbd/scenes/flag.py` | **改**：算 `bend_pairs` 传入 solver |
| `tests/test_topology.py` | **加** bend pairs 测试 |
| `tests/test_constraints.py` | **加** 弯曲不变量（平铺零修正 / 折叠恢复） |
| `tests/test_solver_stability.py` | **加** 带弯曲的 flag 仍稳定 |

---

### Task 1: topology.build_bend_pairs

**Files:**
- Modify: `src/xpbd/solver/topology.py`（追加函数）
- Test: `tests/test_topology.py`（追加）

**Interfaces:**
- Consumes: `faces`（build_grid 产出）
- Produces: `build_bend_pairs(faces: np.ndarray) -> np.ndarray`（int32 `[B,2]`，每条被 2 个三角共享的内部边对应一对"对顶点"）

- [ ] **Step 1: 写失败测试（追加到 `tests/test_topology.py`）**

```python
from xpbd.solver.topology import build_bend_pairs


def test_bend_pairs_count_matches_interior_edges():
    nx, ny = 4, 3
    _, faces, _ = build_grid(nx, ny, 2.0, 1.5)
    bp = build_bend_pairs(faces)
    assert bp.dtype == np.int32 and bp.shape[1] == 2
    # 内部边数 = 总边数 − 边界边数
    total = (nx - 1) * ny + nx * (ny - 1) + (nx - 1) * (ny - 1)   # 23
    boundary = 2 * (nx - 1) + 2 * (ny - 1)                         # 10
    assert len(bp) == total - boundary                            # 13
    # 索引合法、每对两顶点相异
    assert bp.min() >= 0 and bp.max() < nx * ny
    assert np.all(bp[:, 0] != bp[:, 1])
```

- [ ] **Step 2: 跑测试确认失败**

Run: `$PY -m pytest tests/test_topology.py::test_bend_pairs_count_matches_interior_edges -v`
（`$PY` = `/opt/homebrew/Caskroom/miniconda/base/envs/taichi/bin/python`）
Expected: FAIL，`cannot import name 'build_bend_pairs'`

- [ ] **Step 3: 实现（追加到 `src/xpbd/solver/topology.py`）**

```python
from collections import defaultdict


def build_bend_pairs(faces: np.ndarray) -> np.ndarray:
    """每条被恰好 2 个三角共享的内部边 → 它两侧的对顶点 (k,l) 组成一条弯曲距离约束。

    返回 int32 [B,2]（按字典序排序）。开放边界的边（只属 1 个三角）不产生弯曲对。
    """
    faces = np.asarray(faces, dtype=np.int64)
    edge_to_opp: dict[tuple[int, int], list[int]] = defaultdict(list)
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v, opp in ((a, b, c), (b, c, a), (c, a, b)):
            edge_to_opp[(min(u, v), max(u, v))].append(opp)
    pairs = [(o[0], o[1]) for o in edge_to_opp.values() if len(o) == 2]
    if not pairs:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array(sorted(pairs), dtype=np.int32)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `$PY -m pytest tests/test_topology.py -v`
Expected: 全部 passed（原有边测试 + 新 bend 测试）

- [ ] **Step 5: Commit**

```bash
git add src/xpbd/solver/topology.py tests/test_topology.py
git commit -m "feat: topology.build_bend_pairs 提取弯曲对顶点"
```

---

### Task 2: ClothSolver 弯曲约束

**Files:**
- Modify: `src/xpbd/solver/cloth.py`
- Test: `tests/test_constraints.py`（追加）

**Interfaces:**
- Consumes: Task 1 的 `build_bend_pairs`（间接，经测试/flag 传入）
- Produces：`ClothSolver.__init__` 新增关键字参数 `bend_pairs=None, alpha_bend=1e-4`；新增 `solve_bending()` kernel；`substep` 插入 `solve_bending`。
  - `bend_pairs`：int32 `[B,2]` 或 `None`。`self.B = len(bend_pairs) if bend_pairs is not None else 0`。
  - 行为：`B==0` 时 substep 不调 `solve_bending`，与 M1a 完全一致。

- [ ] **Step 1: 写失败测试（追加到 `tests/test_constraints.py`）**

```python
def test_bending_flat_no_correction(ti_cpu):
    # 两三角共享边 0-1, 对顶点 2/3; 平铺 → 距离与弯曲 C 均为 0 → 无运动
    pos = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, -1, 0]], dtype=np.float32)
    edges = np.array([[0, 1]], dtype=np.int32)
    bend = np.array([[2, 3]], dtype=np.int32)
    w = np.ones(4, dtype=np.float32)
    s = ClothSolver(pos, edges, w, bend_pairs=bend, substeps=1, damping=0.0, gravity=(0, 0, 0))
    before = s.x.to_numpy().copy()
    s.predict(); s._clear(); s.solve_distance(); s.solve_bending(); s.apply_dx()
    np.testing.assert_allclose(s.x.to_numpy(), before, atol=1e-6)


def test_bending_restores_rest_distance(ti_cpu):
    # 对顶点 2-3 初始距离 2.0 (= bend rest); 折叠拉近到 0.6 → solve_bending 推回 ~2.0
    pos = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, -1, 0]], dtype=np.float32)
    edges = np.array([[0, 1]], dtype=np.int32)      # 共享边在静止, 不干扰
    bend = np.array([[2, 3]], dtype=np.int32)
    w = np.ones(4, dtype=np.float32)
    s = ClothSolver(pos, edges, w, bend_pairs=bend, substeps=20, damping=0.0, gravity=(0, 0, 0))
    folded = pos.copy()
    folded[2, 1] = 0.3
    folded[3, 1] = -0.3                              # 2-3 距离 2.0 → 0.6
    s.x.from_numpy(folded)
    out = s.run(1)
    d = np.linalg.norm(out[0, 2] - out[0, 3])
    assert abs(d - 2.0) < 0.1                        # 恢复向静止距离


def test_bending_optional_absent_matches_m1a(ti_cpu):
    # 不传 bend_pairs → B=0 → 行为与只有距离约束一致 (向后兼容)
    pos = np.array([[0, 0, 0], [1.5, 0, 0]], dtype=np.float32)
    edges = np.array([[0, 1]], dtype=np.int32)
    w = np.array([1.0, 1.0], dtype=np.float32)
    s = ClothSolver(pos, edges, w, substeps=20, damping=0.0, gravity=(0, 0, 0))
    s.rest.from_numpy(np.array([1.0], dtype=np.float32))
    out = s.run(1)
    assert abs(np.linalg.norm(out[0, 0] - out[0, 1]) - 1.0) < 0.05
    assert s.B == 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `$PY -m pytest tests/test_constraints.py::test_bending_restores_rest_distance -v`
Expected: FAIL，`TypeError: __init__() got an unexpected keyword argument 'bend_pairs'`

- [ ] **Step 3: 改 `src/xpbd/solver/cloth.py`**

(a) `__init__` 签名加两个参数（在 `gravity` 之后）：
```python
    def __init__(self, positions0, edges, inv_mass, *, fps=24.0, substeps=20,
                 alpha_dist=1e-7, omega=0.5, damping=0.01, gravity=(0.0, 0.0, -9.8),
                 bend_pairs=None, alpha_bend=1e-4):
```

(b) `__init__` 体内，在距离约束 field 分配/赋值之后追加 bend 的分配（紧接 `self.rest.from_numpy(...)` 之后）：
```python
        self.alpha_bend = float(alpha_bend)
        if bend_pairs is None or len(bend_pairs) == 0:
            self.B = 0
            bp = np.zeros((1, 2), dtype=np.int32)        # 占位 (shape>=1), 不会被用
        else:
            bp = np.ascontiguousarray(bend_pairs, dtype=np.int32)
            self.B = int(bp.shape[0])
        self.bend_pair = ti.Vector.field(2, ti.i32, shape=max(1, self.B))
        self.bend_rest = ti.field(ti.f32, shape=max(1, self.B))
        self.bend_pair.from_numpy(bp)
        if self.B > 0:
            brest = np.linalg.norm(positions0[bp[:, 0]] - positions0[bp[:, 1]], axis=1)
            # 占位补齐到 max(1,B) 长度
            self.bend_rest.from_numpy(brest.astype(np.float32))
        else:
            self.bend_rest.from_numpy(np.ones(1, dtype=np.float32))
```

(c) 在 `apply_dx` 之前加 `solve_bending` kernel：
```python
    @ti.kernel
    def solve_bending(self):
        a = self.alpha_bend / (self.h * self.h)
        for b in self.bend_pair:
            i, j = self.bend_pair[b][0], self.bend_pair[b][1]
            d = self.x[i] - self.x[j]
            ln = d.norm(1e-12)
            n = d / ln
            dl = -(ln - self.bend_rest[b]) / (self.w[i] + self.w[j] + a)
            self.dx[i] += self.w[i] * dl * n
            self.dx[j] += -self.w[j] * dl * n
            self.cnt[i] += 1
            self.cnt[j] += 1
```

(d) `substep` 在 `solve_distance` 与 `apply_dx` 之间插入弯曲（带 `B>0` 守卫）：
```python
    def substep(self):
        self.predict()
        self._clear()
        self.solve_distance()
        if self.B > 0:
            self.solve_bending()
        self.apply_dx()
        self.update_velocity()
```

- [ ] **Step 4: 跑测试确认通过 + M1a 约束回归**

Run: `$PY -m pytest tests/test_constraints.py tests/test_integrator.py -v`
Expected: 全部 passed（3 个新弯曲测试 + M1a 既有约束/积分测试不回归）

- [ ] **Step 5: Commit**

```bash
git add src/xpbd/solver/cloth.py tests/test_constraints.py
git commit -m "feat: 距离式弯曲约束 (solve_bending, bend_pairs 可选)"
```

---

### Task 3: flag 接入弯曲 + 稳定性

**Files:**
- Modify: `src/xpbd/scenes/flag.py`
- Test: `tests/test_solver_stability.py`（追加）

**Interfaces:**
- Consumes: `build_bend_pairs`, `ClothSolver(bend_pairs=...)`
- Produces: flag 求解器带弯曲；带弯曲的小布仍稳定。

- [ ] **Step 1: 写测试（追加到 `tests/test_solver_stability.py`）**

```python
from xpbd.solver.topology import build_bend_pairs


def test_drape_with_bending_is_stable(ti_cpu):
    pos, edges, w = _pinned_small_cloth()
    _, faces, _ = build_grid(16, 10, 1.5, 1.0)
    bend = build_bend_pairs(faces)
    s = ClothSolver(pos, edges, w, substeps=30, bend_pairs=bend)
    out = s.run(40)
    last = out[-1]
    assert np.all(np.isfinite(last))                                   # 无 NaN
    free = w > 0
    assert last[free, 2].mean() < pos[free, 2].mean() - 0.01           # 仍下沉
    rest = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)
    cur = np.linalg.norm(last[edges[:, 0]] - last[edges[:, 1]], axis=1)
    assert (cur / rest).max() < 1.5                                    # 拉伸仍有界
```

- [ ] **Step 2: 跑测试**

Run: `$PY -m pytest tests/test_solver_stability.py -v`
Expected: 3 passed（M1a 的 2 个 + 新的带弯曲 1 个）。若带弯曲后不稳，调 `alpha_bend`（调大=更软更稳）或 substeps，并记进报告——不要放宽断言。

- [ ] **Step 3: 改 `src/xpbd/scenes/flag.py` 接入弯曲**

在 `edges = build_edges(faces)` 之后、构造 solver 之前加：
```python
    from xpbd.solver.topology import build_bend_pairs
    bend_pairs = build_bend_pairs(faces)
```
并把 solver 构造改为传入 bend：
```python
    solver = ClothSolver(pos, edges, inv_mass, fps=fps, substeps=substeps,
                         bend_pairs=bend_pairs)
```
（`build_bend_pairs` 也可提到文件顶部 import；与现有 import 风格一致即可。）

- [ ] **Step 4: 全量回归 + 手动出片**

Run: `$PY -m pytest -m "not slow" -q`
Expected: 全绿（M0 + M1a + M1b，无回归）

Run:
```bash
$PY scripts/run_sim.py flag --frames 60 --res 48
$PY scripts/render.py flag
```
Expected: `out/flag/flag.mp4` 生成；对比 M1a，布更"挺"、褶皱更少软塌（弯曲抗折叠生效）。仍不飘（风是 M1c）。out/ 是 gitignored，不要 git-add。

- [ ] **Step 5: Commit**

```bash
git add src/xpbd/scenes/flag.py tests/test_solver_stability.py
git commit -m "feat: flag 接入距离式弯曲 + 带弯曲稳定性护栏"
```

---

## Self-Review

**1. Spec coverage（对照 M1 spec）：**
- §1 M1b（距离式弯曲）→ Task 1/2/3 ✓
- §2 弯曲=距离约束、累加同 dx、α_bend、substep 顺序 → Task 2 + Global Constraints ✓
- §3 bend_pair/bend_rest field、topology → Task 1/2 ✓
- §6 弯曲不变量（共面零修正 / 弯折恢复）→ Task 2 ✓
- §8 环境（conda taichi）→ Global Constraints + 全 task ✓

**2. Placeholder scan：** 无 TBD/TODO；每步含完整代码（含探针验证过的 build_bend_pairs）与确切命令/期望。✓

**3. Type consistency：** `build_bend_pairs(faces)->[B,2] int32`、`ClothSolver(..., bend_pairs=None, alpha_bend=1e-4)`、`solve_bending()`、`self.B`、substep 守卫 `if self.B>0` 跨 task 一致。✓

**范围说明：** 本计划只覆盖 **M1b**。M1c（风→飘，必须项）、M1d（气动 stretch）各自成计划。M1c 时一并处理延后项（ti.init 归属、mass/size 经 build_scene 透传）。

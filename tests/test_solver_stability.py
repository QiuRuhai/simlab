import numpy as np
from xpbd.scenes.cloth import build_grid
from xpbd.solver.topology import build_edges, build_bend_pairs
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
    # substeps=30: Jacobi 在 substeps=20 时 max_stretch≈1.504（真信号），
    # 提高到 30 后降至 ~1.18，有充足余量（<1.5）。alpha_dist 保持默认 1e-7。
    s = ClothSolver(pos, edges, w, substeps=30)
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
    s = ClothSolver(pos, edges, w, substeps=30)
    out = s.run(40)
    pinned = w == 0
    np.testing.assert_allclose(out[-1][pinned], pos[pinned], atol=1e-5)


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

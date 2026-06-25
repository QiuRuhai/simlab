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

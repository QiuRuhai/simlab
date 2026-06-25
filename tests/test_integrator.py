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

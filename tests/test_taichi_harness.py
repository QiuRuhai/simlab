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

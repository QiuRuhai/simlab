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
    assert np.ptp(p0[:, 2]) > 0.5          # z 方向铺开 (高度~1.0)


def test_flag_hoist_edge_pinned():
    r = build_flag(res=16, frames=6, substeps=10)
    p0, plast = r.positions[0], r.positions[-1]
    # x=min 那条竖边的点全程不动
    xmin = p0[:, 0].min()
    hoist = np.isclose(p0[:, 0], xmin)
    np.testing.assert_allclose(plast[hoist], p0[hoist], atol=1e-5)
    # 非钉点确实动了 (重力)
    assert not np.allclose(plast[~hoist], p0[~hoist], atol=1e-4)

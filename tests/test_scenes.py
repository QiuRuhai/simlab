import numpy as np
from xpbd.scenes.cloth import build_grid
from xpbd.scenes.wave import generate_wave
from xpbd.scenes.presets import PRESETS, build_scene
from xpbd.io.result import SimResult


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


def test_wave_only_z_animates():
    verts, faces, uvs = build_grid(8, 6, 2.0, 1.5)
    r = generate_wave(verts, faces, uvs, name="wave", frames=10, fps=24.0,
                      amplitude=0.2, wavelength=1.0, speed=1.0)
    assert isinstance(r, SimResult)
    assert r.positions.shape == (10, 8 * 6, 3)
    # x,y 全程不变
    np.testing.assert_allclose(r.positions[:, :, 0], np.tile(r.positions[0, :, 0], (r.positions.shape[0], 1)), atol=1e-6)
    np.testing.assert_allclose(r.positions[:, :, 1], np.tile(r.positions[0, :, 1], (r.positions.shape[0], 1)), atol=1e-6)
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

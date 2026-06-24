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

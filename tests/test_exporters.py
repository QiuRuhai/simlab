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

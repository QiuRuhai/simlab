import numpy as np
import pytest
from xpbd.io.result import SimResult


def _mini():
    positions = np.zeros((3, 4, 3), dtype=np.float32)   # F=3, N=4
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)  # M=2
    uvs = np.zeros((4, 2), dtype=np.float32)
    return SimResult(positions=positions, faces=faces, uvs=uvs, fps=24.0, name="t")


def test_counts():
    r = _mini()
    assert r.num_frames == 3
    assert r.num_verts == 4
    assert r.num_faces == 2


def test_forces_dtype():
    r = SimResult(
        positions=np.zeros((2, 4, 3)),          # 传 float64
        faces=np.array([[0, 1, 2], [0, 2, 3]]),  # 传 int64
        uvs=np.zeros((4, 2)),
        fps=24.0, name="t",
    )
    assert r.positions.dtype == np.float32
    assert r.faces.dtype == np.int32
    assert r.uvs.dtype == np.float32


def test_vert_count_mismatch_raises():
    positions = np.zeros((3, 4, 3), dtype=np.float32)  # N=4
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    uvs = np.zeros((5, 2), dtype=np.float32)           # N=5 ≠ 4
    with pytest.raises(ValueError):
        SimResult(positions=positions, faces=faces, uvs=uvs, fps=24.0, name="t")


def test_bad_position_shape_raises():
    with pytest.raises(ValueError):
        SimResult(
            positions=np.zeros((3, 4, 2), dtype=np.float32),  # 末维不是 3
            faces=np.array([[0, 1, 2]], dtype=np.int32),
            uvs=np.zeros((4, 2), dtype=np.float32),
            fps=24.0, name="t",
        )


def test_negative_face_index_raises():
    with pytest.raises(ValueError):
        SimResult(
            positions=np.zeros((2, 4, 3), dtype=np.float32),
            faces=np.array([[-1, 0, 1]], dtype=np.int32),
            uvs=np.zeros((4, 2), dtype=np.float32),
            fps=24.0, name="t",
        )

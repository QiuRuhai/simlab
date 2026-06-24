import numpy as np
from xpbd.io.result import SimResult
from xpbd.io.npz_exporter import save_npz, load_npz
from xpbd.io.usd_exporter import export_usd


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


def test_usd_roundtrip(tmp_path):
    r = _scene()  # F=5, N=6, M=2
    out = export_usd(r, tmp_path / "cache.usd")
    assert out.exists()

    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(out))

    # stage 元数据
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert stage.GetTimeCodesPerSecond() == r.fps
    assert stage.GetStartTimeCode() == 1
    assert stage.GetEndTimeCode() == r.num_frames

    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/cloth"))
    assert mesh

    # 拓扑静态、三角形
    counts = list(mesh.GetFaceVertexCountsAttr().Get())
    assert counts == [3] * r.num_faces
    indices = list(mesh.GetFaceVertexIndicesAttr().Get())
    assert indices == r.faces.flatten().tolist()

    # points 逐帧时间采样
    samples = mesh.GetPointsAttr().GetTimeSamples()
    assert len(samples) == r.num_frames
    pts1 = np.array(mesh.GetPointsAttr().Get(Usd.TimeCode(1)), dtype=np.float32)
    np.testing.assert_allclose(pts1, r.positions[0], rtol=0, atol=1e-6)

    # UV primvar st
    st = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    assert st
    assert st.GetInterpolation() == UsdGeom.Tokens.vertex
    uvs = np.array(st.Get(), dtype=np.float32)
    np.testing.assert_allclose(uvs, r.uvs, rtol=0, atol=1e-6)

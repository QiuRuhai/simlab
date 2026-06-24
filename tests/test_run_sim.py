from pathlib import Path
import run_sim  # 经 conftest.py 加入 sys.path


def test_run_sim_writes_usd_with_timesamples(tmp_path):
    out_dir = tmp_path / "wave"
    usd = run_sim.main(["wave", "--frames", "3", "--out", str(out_dir)])
    assert usd == out_dir / "cache.usd"
    assert usd.exists()

    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(usd))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/cloth"))
    assert len(mesh.GetPointsAttr().GetTimeSamples()) == 3


def test_run_sim_npz_flag(tmp_path):
    out_dir = tmp_path / "wave"
    run_sim.main(["wave", "--frames", "2", "--out", str(out_dir), "--npz"])
    assert (out_dir / "cache.npz").exists()

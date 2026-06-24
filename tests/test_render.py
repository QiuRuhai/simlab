import render  # 经 conftest.py 加入 sys.path


def test_read_fps_from_usd(tmp_path):
    import run_sim
    usd = run_sim.main(["wave", "--frames", "2", "--out", str(tmp_path / "wave")])
    assert render.read_fps_from_usd(usd) == 24.0

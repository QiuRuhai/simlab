import subprocess

import pytest

import run_sim
import render


@pytest.mark.slow
def test_m0_pipeline_end_to_end(tmp_path, monkeypatch):
    # 在 tmp 下跑，避免污染仓库 out/
    monkeypatch.chdir(tmp_path)
    # render.py 用模块级 ROOT 定位 out/ 与 blender/，这里把 ROOT 指到真仓库 blender/，out 指到 tmp
    repo_root = render.ROOT

    # 1) 模拟 → cache.usd（写到 tmp/out/wave）
    out_dir = tmp_path / "out" / "wave"
    usd = run_sim.main(["wave", "--frames", "8", "--out", str(out_dir)])
    assert usd.exists()

    # 2) 渲染：直接调 blender 子进程（复用 render 的常量），输出到 tmp/out/wave
    subprocess.run([
        render.BLENDER, "--background", "--python",
        str(repo_root / "blender" / "blender_render.py"), "--",
        "--usd", str(usd), "--out-dir", str(out_dir), "--res", "240x180",
    ], check=True)
    frames = sorted((out_dir / "frames").glob("*.png"))
    assert len(frames) == 8

    # 3) 帧之间确有差异（几何在动）
    assert frames[0].read_bytes() != frames[-1].read_bytes()

    # 4) 拼帧成 mp4
    subprocess.run([
        render.BLENDER, "--background", "--python",
        str(repo_root / "blender" / "stitch.py"), "--",
        "--frames-dir", str(out_dir / "frames"), "--out-dir", str(out_dir),
        "--name", "wave", "--fps", "24",
    ], check=True)
    mp4 = out_dir / "wave.mp4"
    assert mp4.exists() and mp4.stat().st_size > 0

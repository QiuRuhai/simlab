from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLENDER = shutil.which("blender") or "/opt/homebrew/bin/blender"
BLENDER_DIR = ROOT / "blender"


def read_fps_from_usd(path) -> float:
    from pxr import Usd
    stage = Usd.Stage.Open(str(path))
    return float(stage.GetTimeCodesPerSecond() or 24.0)


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description="渲染 out/<demo>/cache.usd → mp4")
    ap.add_argument("demo")
    ap.add_argument("--res", default="480x360")
    ap.add_argument("--engine", default="BLENDER_EEVEE")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args(argv)

    out_dir = ROOT / "out" / args.demo
    usd = out_dir / "cache.usd"
    if not usd.exists():
        raise SystemExit(f"{usd} not found — 先跑 run_sim.py {args.demo}")

    # 1) 渲 PNG 帧
    subprocess.run([
        BLENDER, "--background", "--python", str(BLENDER_DIR / "blender_render.py"), "--",
        "--usd", str(usd), "--out-dir", str(out_dir),
        "--res", args.res, "--engine", args.engine, "--samples", str(args.samples),
    ], check=True)

    frames_dir = out_dir / "frames"
    if args.no_video:
        print(f"[render] frames at {frames_dir}")
        return frames_dir

    # 2) 拼 mp4：有 ffmpeg 用 ffmpeg，否则用 Blender 自带
    fps = read_fps_from_usd(usd)
    mp4 = out_dir / f"{args.demo}.mp4"
    if shutil.which("ffmpeg"):
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(fps),
            "-pattern_type", "glob", "-i", str(frames_dir / "*.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4),
        ], check=True)
    else:
        subprocess.run([
            BLENDER, "--background", "--python", str(BLENDER_DIR / "stitch.py"), "--",
            "--frames-dir", str(frames_dir), "--out-dir", str(out_dir),
            "--name", args.demo, "--fps", str(fps),
        ], check=True)

    print(f"[render] video at {mp4}")
    return mp4


if __name__ == "__main__":
    main()

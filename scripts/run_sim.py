from __future__ import annotations

import argparse
from pathlib import Path

from xpbd.io.npz_exporter import save_npz
from xpbd.io.usd_exporter import export_usd
from xpbd.scenes.presets import build_scene


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description="跑场景 → out/<scene>/cache.usd")
    ap.add_argument("scene", help="场景名，如 wave")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--res", type=int, default=None, help="长边分辨率")
    ap.add_argument("--substeps", type=int, default=None, help="每帧 substep 数 (求解器场景)")
    ap.add_argument("--npz", action="store_true", help="同时写调试 cache.npz")
    ap.add_argument("--out", type=str, default=None, help="输出目录（默认 out/<scene>）")
    args = ap.parse_args(argv)

    result = build_scene(args.scene, frames=args.frames, res=args.res, substeps=args.substeps)
    out_dir = Path(args.out) if args.out else Path("out") / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)

    usd_path = export_usd(result, out_dir / "cache.usd")
    print(f"[run_sim] wrote {usd_path}  (F={result.num_frames}, N={result.num_verts})")
    if args.npz:
        npz_path = save_npz(result, out_dir / "cache.npz")
        print(f"[run_sim] wrote {npz_path}")
    return usd_path


if __name__ == "__main__":
    main()

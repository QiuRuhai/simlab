"""无头：导入 cache.usd → 套 look → 渲 PNG 帧。在 Blender 内运行。"""
import argparse
import os
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import look  # noqa: E402  同目录


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--res", default="480x360")
    ap.add_argument("--engine", default="BLENDER_EEVEE")
    ap.add_argument("--samples", type=int, default=16)
    return ap.parse_args(argv)


def main():
    args = parse_args()

    # 从空场景开始
    bpy.ops.wm.read_factory_settings(use_empty=True)

    bpy.ops.wm.usd_import(
        filepath=args.usd,
        import_cameras=False,
        import_lights=False,
        import_materials=False,
        read_mesh_uvs=True,
        set_frame_range=True,   # 按 USD 自动设帧范围
    )

    scene = bpy.context.scene
    look.setup(scene, engine=args.engine, samples=args.samples)

    w, h = (int(v) for v in args.res.lower().split("x"))
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.image_settings.file_format = "PNG"

    frames_dir = Path(args.out_dir) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    # 末尾带分隔符 → Blender 自动追加 4 位帧号 0001.png ...
    scene.render.filepath = os.path.join(str(frames_dir), "")

    bpy.ops.render.render(animation=True)
    print(f"[blender_render] frames -> {frames_dir} "
          f"({scene.frame_start}..{scene.frame_end})")


main()

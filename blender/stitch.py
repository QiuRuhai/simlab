"""無頭：把 PNG 幀序列經 VSE 編碼成 mp4（Blender 自帶 FFMPEG）。"""
import argparse
import glob
import os
import sys

import bpy


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--fps", type=float, default=24.0)
    return ap.parse_args(argv)


def main():
    args = parse_args()
    frames = sorted(glob.glob(os.path.join(args.frames_dir, "*.png")))
    if not frames:
        raise SystemExit(f"no png frames in {args.frames_dir}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    se = scene.sequence_editor_create()
    strip = se.strips.new_image(name="frames", filepath=frames[0],
                                channel=1, frame_start=1)
    for fp in frames[1:]:
        strip.elements.append(os.path.basename(fp))

    scene.frame_start = 1
    scene.frame_end = len(frames)
    scene.render.fps = max(1, int(round(args.fps)))
    scene.render.use_sequencer = True
    scene.render.image_settings.media_type = "VIDEO"  # Blender 5.x: unlock FFMPEG in dynamic enum
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"

    stem = os.path.join(args.out_dir, args.name)
    scene.render.filepath = stem
    bpy.ops.render.render(animation=True)

    # FFMPEG 容器輸出會把幀範圍拼進文件名，找出來重命名成 <name>.mp4
    final = stem + ".mp4"
    produced = sorted(glob.glob(stem + "*.mp4"))
    if produced and produced[0] != final:
        os.replace(produced[0], final)
    print(f"[stitch] wrote {final}")


main()

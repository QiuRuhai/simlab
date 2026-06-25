from __future__ import annotations

from xpbd.io.result import SimResult
from xpbd.scenes.cloth import build_grid
from xpbd.scenes.wave import generate_wave

PRESETS: dict[str, dict] = {
    "wave": dict(
        nx=64, ny=48, width=2.0, height=1.5,
        frames=48, fps=24.0,
        amplitude=0.15, wavelength=1.0, speed=1.0,
    ),
}


def build_scene(name: str, *, frames: int | None = None, res: int | None = None,
                substeps: int | None = None) -> SimResult:
    if name == "flag":
        from xpbd.scenes.flag import build_flag
        kw = {}
        if frames is not None:
            kw["frames"] = frames
        if res is not None:
            kw["res"] = res
        if substeps is not None:
            kw["substeps"] = substeps
        return build_flag(**kw)

    if name not in PRESETS:
        raise KeyError(f"unknown scene '{name}'; known: {sorted(PRESETS) + ['flag']}")
    # —— 以下为原 wave 路径，保持不变 ——
    p = dict(PRESETS[name])
    if frames is not None:
        p["frames"] = frames
    if res is not None:
        if p["nx"] >= p["ny"]:
            aspect = p["ny"] / p["nx"]
            p["nx"] = int(res)
            p["ny"] = max(2, round(res * aspect))
        else:
            aspect = p["nx"] / p["ny"]
            p["ny"] = int(res)
            p["nx"] = max(2, round(res * aspect))

    verts, faces, uvs = build_grid(p["nx"], p["ny"], p["width"], p["height"])
    return generate_wave(
        verts, faces, uvs, name=name, frames=p["frames"], fps=p["fps"],
        amplitude=p["amplitude"], wavelength=p["wavelength"], speed=p["speed"],
    )

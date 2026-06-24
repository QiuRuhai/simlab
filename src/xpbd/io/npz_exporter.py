from __future__ import annotations

from pathlib import Path

import numpy as np

from xpbd.io.result import SimResult


def save_npz(result: SimResult, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        positions=result.positions,
        faces=result.faces,
        uvs=result.uvs,
        fps=np.float64(result.fps),
        name=np.array(result.name),
    )
    # np.savez 会自动补 .npz 后缀
    return path if path.suffix == ".npz" else path.with_suffix(".npz")


def load_npz(path) -> SimResult:
    data = np.load(Path(path), allow_pickle=False)
    return SimResult(
        positions=data["positions"],
        faces=data["faces"],
        uvs=data["uvs"],
        fps=float(data["fps"]),
        name=str(data["name"]),
    )

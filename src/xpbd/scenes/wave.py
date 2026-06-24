from __future__ import annotations

import numpy as np

from xpbd.io.result import SimResult


def generate_wave(verts, faces, uvs, *, name, frames, fps,
                  amplitude, wavelength, speed) -> SimResult:
    """假数据：沿 X 行进的正弦波，只动 z。z = A sin(k x - ω t)。"""
    verts = np.asarray(verts, dtype=np.float32)
    x = verts[:, 0]
    k = 2.0 * np.pi / wavelength
    omega = 2.0 * np.pi * speed

    positions = np.empty((frames, verts.shape[0], 3), dtype=np.float32)
    for f in range(frames):
        t = f / fps
        frame = verts.copy()
        frame[:, 2] = amplitude * np.sin(k * x - omega * t)
        positions[f] = frame

    return SimResult(positions=positions, faces=faces, uvs=uvs, fps=fps, name=name)

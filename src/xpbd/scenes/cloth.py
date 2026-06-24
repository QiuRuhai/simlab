from __future__ import annotations

import numpy as np


def build_grid(nx: int, ny: int, width: float, height: float):
    """XY 平面规则网格，z=0，中心在原点。顶点序 idx = j*nx + i。"""
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be >= 2")

    i = np.arange(nx)
    j = np.arange(ny)
    ii, jj = np.meshgrid(i, j)            # 形状 [ny, nx]
    u = ii / (nx - 1)
    v = jj / (ny - 1)

    x = (u - 0.5) * width
    y = (v - 0.5) * height
    z = np.zeros_like(x)
    verts = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
    uvs = np.stack([u, v], axis=-1).reshape(-1, 2).astype(np.float32)

    faces = []
    for jc in range(ny - 1):
        for ic in range(nx - 1):
            v00 = jc * nx + ic
            v10 = jc * nx + ic + 1
            v01 = (jc + 1) * nx + ic
            v11 = (jc + 1) * nx + ic + 1
            faces.append((v00, v10, v11))   # CCW（从 +Z 看）
            faces.append((v00, v11, v01))
    faces = np.array(faces, dtype=np.int32)
    return verts, faces, uvs

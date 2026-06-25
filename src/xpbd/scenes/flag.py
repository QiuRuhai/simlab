from __future__ import annotations

import numpy as np
import taichi as ti

from xpbd.io.result import SimResult
from xpbd.scenes.cloth import build_grid
from xpbd.solver.cloth import ClothSolver
from xpbd.solver.topology import build_edges, build_bend_pairs


def build_flag(*, res: int = 64, frames: int = 250, fps: float = 24.0,
               substeps: int = 30, width: float = 1.5, height: float = 1.0,
               mass: float = 0.2) -> SimResult:
    """旗子场景：竖直平面网格 + hoist 竖边钉住 + 重力垂坠（M1a 无风）。"""
    nx = int(res)
    ny = max(2, round(res * (height / width)))     # 3:2 → ny≈res*2/3

    verts_xy, faces, uvs = build_grid(nx, ny, width, height)
    # 映射进竖直平面: 宽→X, 高→Z, y=0
    pos = np.zeros((len(verts_xy), 3), dtype=np.float32)
    pos[:, 0] = verts_xy[:, 0]
    pos[:, 2] = verts_xy[:, 1]

    n = nx * ny
    inv_mass = np.full(n, n / mass, dtype=np.float32)   # w = 1/m = N/mass
    inv_mass[np.isclose(pos[:, 0], pos[:, 0].min())] = 0.0   # 钉 hoist 竖边

    edges = build_edges(faces)
    bend_pairs = build_bend_pairs(faces)

    ti.init(arch=ti.metal)     # flag 入口负责 init (求解器模块自己不 init)
    solver = ClothSolver(pos, edges, inv_mass, fps=fps, substeps=substeps,
                         bend_pairs=bend_pairs)
    positions = solver.run(frames)

    return SimResult(positions=positions, faces=faces, uvs=uvs, fps=fps, name="flag")

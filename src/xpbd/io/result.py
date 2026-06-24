from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimResult:
    """模拟端 → 渲染端 的唯一数据契约：几何随时间变化，不含任何 look。"""

    positions: np.ndarray  # float32 [F, N, 3]
    faces: np.ndarray      # int32  [M, 3]
    uvs: np.ndarray        # float32 [N, 2]
    fps: float
    name: str

    def __post_init__(self) -> None:
        self.positions = np.ascontiguousarray(self.positions, dtype=np.float32)
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int32)
        self.uvs = np.ascontiguousarray(self.uvs, dtype=np.float32)

        if self.positions.ndim != 3 or self.positions.shape[2] != 3:
            raise ValueError(f"positions must be [F,N,3], got {self.positions.shape}")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"faces must be [M,3], got {self.faces.shape}")
        if self.uvs.ndim != 2 or self.uvs.shape[1] != 2:
            raise ValueError(f"uvs must be [N,2], got {self.uvs.shape}")

        n = self.positions.shape[1]
        if self.uvs.shape[0] != n:
            raise ValueError(f"uvs N={self.uvs.shape[0]} != positions N={n}")
        if self.faces.size and int(self.faces.max()) >= n:
            raise ValueError("face index out of range of vertex count")

    @property
    def num_frames(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_verts(self) -> int:
        return int(self.positions.shape[1])

    @property
    def num_faces(self) -> int:
        return int(self.faces.shape[0])

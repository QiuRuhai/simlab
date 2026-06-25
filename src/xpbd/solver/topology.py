from __future__ import annotations

import numpy as np


def build_edges(faces: np.ndarray) -> np.ndarray:
    """从三角面提取唯一无向边：结构边 + 每个 quad 的剪切对角边。

    返回 int32 [E,2]，每行 i<j，去重并按字典序排序。
    （build_grid 把每个 quad 切成 2 个三角，对角边作为三角边自然包含进来。）
    """
    faces = np.asarray(faces, dtype=np.int64)
    e = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    e = np.sort(e, axis=1)          # 每条边 i<j
    e = np.unique(e, axis=0)        # 去重 + 排序
    return e.astype(np.int32)

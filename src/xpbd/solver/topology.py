from __future__ import annotations

from collections import defaultdict

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


def build_bend_pairs(faces: np.ndarray) -> np.ndarray:
    """每条被恰好 2 个三角共享的内部边 → 它两侧的对顶点 (k,l) 组成一条弯曲距离约束。

    返回 int32 [B,2]（按字典序排序）。开放边界的边（只属 1 个三角）不产生弯曲对。
    """
    faces = np.asarray(faces, dtype=np.int64)
    edge_to_opp: dict[tuple[int, int], list[int]] = defaultdict(list)
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v, opp in ((a, b, c), (b, c, a), (c, a, b)):
            edge_to_opp[(min(u, v), max(u, v))].append(opp)
    pairs = [(o[0], o[1]) for o in edge_to_opp.values() if len(o) == 2]
    if not pairs:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array(sorted(pairs), dtype=np.int32)

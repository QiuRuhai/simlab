import numpy as np
from xpbd.scenes.cloth import build_grid
from xpbd.solver.topology import build_edges


def test_edges_unique_sorted_valid():
    _, faces, _ = build_grid(4, 3, 2.0, 1.5)   # nx=4, ny=3
    edges = build_edges(faces)
    assert edges.dtype == np.int32 and edges.shape[1] == 2
    # 每行 i<j
    assert np.all(edges[:, 0] < edges[:, 1])
    # 无重复
    assert len(np.unique(edges, axis=0)) == len(edges)
    # 索引合法
    assert edges.min() >= 0 and edges.max() < 4 * 3


def test_edge_count_matches_grid_formula():
    nx, ny = 4, 3
    _, faces, _ = build_grid(nx, ny, 2.0, 1.5)
    edges = build_edges(faces)
    # 结构: 水平 (nx-1)*ny + 竖直 nx*(ny-1); 剪切对角: (nx-1)*(ny-1)
    expected = (nx - 1) * ny + nx * (ny - 1) + (nx - 1) * (ny - 1)
    assert len(edges) == expected

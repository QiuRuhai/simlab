from __future__ import annotations

import numpy as np
import taichi as ti


@ti.data_oriented
class ClothSolver:
    """small-steps XPBD 布料求解器（M1a: 距离约束 + 固定；调用方负责 ti.init）。"""

    def __init__(self, positions0, edges, inv_mass, *, fps=24.0, substeps=20,
                 alpha_dist=1e-7, omega=0.5, damping=0.01, gravity=(0.0, 0.0, -9.8),
                 bend_pairs=None, alpha_bend=1e-4):
        positions0 = np.ascontiguousarray(positions0, dtype=np.float32)
        edges = np.ascontiguousarray(edges, dtype=np.int32)
        inv_mass = np.ascontiguousarray(inv_mass, dtype=np.float32)

        self.N = int(positions0.shape[0])
        self.E = int(edges.shape[0])
        self.substeps = int(substeps)
        self.h = (1.0 / fps) / substeps
        self.alpha_dist = float(alpha_dist)
        self.omega = float(omega)
        self.damping = float(damping)

        self.x = ti.Vector.field(3, ti.f32, shape=self.N)
        self.v = ti.Vector.field(3, ti.f32, shape=self.N)
        self.x_prev = ti.Vector.field(3, ti.f32, shape=self.N)
        self.w = ti.field(ti.f32, shape=self.N)
        self.dx = ti.Vector.field(3, ti.f32, shape=self.N)
        self.cnt = ti.field(ti.i32, shape=self.N)
        self.edge = ti.Vector.field(2, ti.i32, shape=self.E)
        self.rest = ti.field(ti.f32, shape=self.E)
        self.g = ti.Vector([float(x) for x in gravity], dt=ti.f32)

        self.x.from_numpy(positions0)
        self.v.fill(0.0)
        self.w.from_numpy(inv_mass)
        self.edge.from_numpy(edges)
        rest = np.linalg.norm(positions0[edges[:, 0]] - positions0[edges[:, 1]], axis=1)
        self.rest.from_numpy(rest.astype(np.float32))

        self.alpha_bend = float(alpha_bend)
        if bend_pairs is None or len(bend_pairs) == 0:
            self.B = 0
            bp = np.zeros((1, 2), dtype=np.int32)        # 占位 (shape>=1), 不会被用
        else:
            bp = np.ascontiguousarray(bend_pairs, dtype=np.int32)
            self.B = int(bp.shape[0])
        self.bend_pair = ti.Vector.field(2, ti.i32, shape=max(1, self.B))
        self.bend_rest = ti.field(ti.f32, shape=max(1, self.B))
        self.bend_pair.from_numpy(bp)
        if self.B > 0:
            brest = np.linalg.norm(positions0[bp[:, 0]] - positions0[bp[:, 1]], axis=1)
            # 占位补齐到 max(1,B) 长度
            self.bend_rest.from_numpy(brest.astype(np.float32))
        else:
            self.bend_rest.from_numpy(np.ones(1, dtype=np.float32))

    @ti.kernel
    def predict(self):
        for i in self.x:
            self.x_prev[i] = self.x[i]
            if self.w[i] > 0:
                self.v[i] += self.h * self.g
                self.x[i] += self.h * self.v[i]

    @ti.kernel
    def update_velocity(self):
        for i in self.x:
            if self.w[i] > 0:
                self.v[i] = (self.x[i] - self.x_prev[i]) / self.h
                self.v[i] *= (1.0 - self.damping)

    @ti.kernel
    def _clear(self):
        for i in self.x:
            self.dx[i] = ti.Vector([0.0, 0.0, 0.0])
            self.cnt[i] = 0

    @ti.kernel
    def solve_distance(self):
        a = self.alpha_dist / (self.h * self.h)
        for e in self.edge:
            i, j = self.edge[e][0], self.edge[e][1]
            d = self.x[i] - self.x[j]
            ln = d.norm(1e-12)
            n = d / ln
            dl = -(ln - self.rest[e]) / (self.w[i] + self.w[j] + a)
            self.dx[i] += self.w[i] * dl * n       # Taichi += 自动原子化
            self.dx[j] += -self.w[j] * dl * n
            self.cnt[i] += 1
            self.cnt[j] += 1

    @ti.kernel
    def solve_bending(self):
        a = self.alpha_bend / (self.h * self.h)
        for b in self.bend_pair:
            i, j = self.bend_pair[b][0], self.bend_pair[b][1]
            d = self.x[i] - self.x[j]
            ln = d.norm(1e-12)
            n = d / ln
            dl = -(ln - self.bend_rest[b]) / (self.w[i] + self.w[j] + a)
            self.dx[i] += self.w[i] * dl * n
            self.dx[j] += -self.w[j] * dl * n
            self.cnt[i] += 1
            self.cnt[j] += 1

    @ti.kernel
    def apply_dx(self):
        for i in self.x:
            if self.w[i] > 0 and self.cnt[i] > 0:
                self.x[i] += self.omega * self.dx[i] / self.cnt[i]

    def substep(self):
        self.predict()
        self._clear()
        self.solve_distance()
        if self.B > 0:
            self.solve_bending()
        self.apply_dx()
        self.update_velocity()

    def run(self, frames: int) -> np.ndarray:
        out = np.empty((frames, self.N, 3), dtype=np.float32)
        for f in range(frames):
            for _ in range(self.substeps):
                self.substep()
            out[f] = self.x.to_numpy()
        return out

from __future__ import annotations

import numpy as np
import taichi as ti


@ti.data_oriented
class ClothSolver:
    """small-steps XPBD 布料求解器（M1a: 距离约束 + 固定；调用方负责 ti.init）。"""

    def __init__(self, positions0, edges, inv_mass, *, fps=24.0, substeps=20,
                 alpha_dist=1e-7, omega=0.5, damping=0.01, gravity=(0.0, 0.0, -9.8)):
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

    def substep(self):
        self.predict()
        self.update_velocity()

    def run(self, frames: int) -> np.ndarray:
        out = np.empty((frames, self.N, 3), dtype=np.float32)
        for f in range(frames):
            for _ in range(self.substeps):
                self.substep()
            out[f] = self.x.to_numpy()
        return out

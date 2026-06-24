from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Sdf, Usd, UsdGeom, Vt

from xpbd.io.result import SimResult


def export_usd(result: SimResult, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(result.fps)
    stage.SetFramesPerSecond(result.fps)
    stage.SetStartTimeCode(1)
    stage.SetEndTimeCode(result.num_frames)

    mesh = UsdGeom.Mesh.Define(stage, "/cloth")
    stage.SetDefaultPrim(mesh.GetPrim())

    # 拓扑静态：写一次
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(
        np.full(result.num_faces, 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(
        result.faces.reshape(-1).astype(np.int32)))

    # points 逐帧时间采样 + 逐帧 extent
    points_attr = mesh.CreatePointsAttr()
    extent_attr = mesh.CreateExtentAttr()
    for f in range(result.num_frames):
        frame_pts = result.positions[f].astype(np.float32)
        tc = Usd.TimeCode(f + 1)
        points_attr.Set(Vt.Vec3fArray.FromNumpy(frame_pts), tc)
        lo = frame_pts.min(axis=0)
        hi = frame_pts.max(axis=0)
        extent_attr.Set(Vt.Vec3fArray.FromNumpy(
            np.stack([lo, hi]).astype(np.float32)), tc)

    # UV 作为 primvar st（顶点插值，静态）
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray.FromNumpy(result.uvs.astype(np.float32)))

    stage.GetRootLayer().Save()
    return path

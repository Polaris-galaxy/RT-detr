"""相机射线与 map 平面求交（用于将检测框投到地面上）.

假设 map 坐标系地面为平面 **z = z_map_ground**（米）.
射线在 **camera光学系** REP-103：X右、Y下、Z前.
"""

from __future__ import annotations

import numpy as np


def ray_dir_optical_from_pixel(u: float, v: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    x = (u - cx) / max(fx, 1e-9)
    y = (v - cy) / max(fy, 1e-9)
    d = np.array([x, y, 1.0], dtype=np.float64)
    n = np.linalg.norm(d)
    return d / max(n, 1e-12)


def intersect_ray_ground_in_map(
    T_map_optical: np.ndarray,
    ray_origin_optical: np.ndarray,
    ray_dir_optical: np.ndarray,
    *,
    z_map_ground: float = 0.0,
) -> tuple[float, float] | None:
    """T_map_optical: 4x4，将光学系点变到 map。射线起点取相机光心 O_optical=[0,0,0].

    返回 (x_map, y_map) 或 None（射线平行于地面或交点在身后）.
    """
    R = T_map_optical[:3, :3].astype(np.float64)
    t = T_map_optical[:3, 3].astype(np.float64)
    O_map = R @ ray_origin_optical + t
    d_map = R @ ray_dir_optical
    if abs(d_map[2]) < 1e-9:
        return None
    s = (float(z_map_ground) - float(O_map[2])) / float(d_map[2])
    if s <= 0:
        return None
    P = O_map + s * d_map
    return (float(P[0]), float(P[1]))


def bottom_center_xyxy(xyxy: np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (float((x1 + x2) * 0.5), float(y2))

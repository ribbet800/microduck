"""Debug markers drawn into the 3D scene.

Works for both the live viewer (`viewer.user_scn`) and offscreen video
(`renderer.scene`): each function appends geoms after whatever is already there.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

ORANGE = (1.0, 0.55, 0.0, 0.9)
GREEN = (0.2, 0.9, 0.3, 0.25)
BLUE = (0.2, 0.5, 1.0, 0.9)
WHITE = (1.0, 1.0, 1.0, 0.6)
RED = (1.0, 0.25, 0.2, 0.9)
YELLOW = (1.0, 0.85, 0.1, 0.9)

_EYE3 = np.eye(3, dtype=np.float64).flatten()


def _add(scn: mujoco.MjvScene):
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    scn.ngeom += 1
    return g


def _connector(scn, kind, width, a, b, rgba):
    g = _add(scn)
    if g is None:
        return
    mujoco.mjv_initGeom(g, kind, np.zeros(3), np.zeros(3), _EYE3, np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, kind, width, np.asarray(a, float), np.asarray(b, float))


def line(scn, a_xy, b_xy, rgba=ORANGE, width=3.0, z=0.01):
    _connector(scn, mujoco.mjtGeom.mjGEOM_LINE, width, [a_xy[0], a_xy[1], z], [b_xy[0], b_xy[1], z], rgba)


def arrow(scn, a_xy, b_xy, rgba=BLUE, width=0.008, z=0.015):
    _connector(scn, mujoco.mjtGeom.mjGEOM_ARROW, width, [a_xy[0], a_xy[1], z], [b_xy[0], b_xy[1], z], rgba)


def disc(scn, xy, radius, rgba=GREEN, z=0.002):
    g = _add(scn)
    if g is not None:
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([radius, radius, 0.001]),
                            np.array([xy[0], xy[1], z]), _EYE3, np.asarray(rgba, np.float32))


def sphere(scn, xyz, radius, rgba):
    g = _add(scn)
    if g is not None:
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]),
                            np.asarray(xyz, float), _EYE3, np.asarray(rgba, np.float32))


def trail(scn, points, rgba=WHITE, radius=0.006):
    """Dots along a path, older ones more transparent."""
    n = len(points)
    for i, (x, y) in enumerate(points):
        c = list(rgba)
        c[3] = 0.15 + 0.5 * (i + 1) / n
        sphere(scn, (x, y, radius), radius, c)


def command_arrow(scn, duck_xy, duck_yaw, cmd, rgba=BLUE):
    """The velocity command being sent: length = forward speed, swung by the turn
    command (roughly where the duck will be heading ~0.5 s from now)."""
    vx, vy, wz = (float(c) for c in cmd)
    heading = duck_yaw + 0.5 * wz
    length = 0.05 + 0.8 * math.hypot(max(vx, 0.0), vy)
    side = math.atan2(vy, max(vx, 1e-3)) if vy else 0.0
    tip = (duck_xy[0] + math.cos(heading + side) * length, duck_xy[1] + math.sin(heading + side) * length)
    arrow(scn, duck_xy, tip, rgba)

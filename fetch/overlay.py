"""Debug markers drawn into the 3D scene, so you can see what the chaser is "thinking".

- a line from the duck to the ball            (what it's aiming at)
- a green disc round the ball = catch radius  (where it counts as "reached")
- a blue arrow from the duck                  (the command it's sending: length = forward
                                               speed, swung by the turn command)
- a fading trail of dots                      (where the duck has been)

Works for both the live viewer (`viewer.user_scn`) and offscreen video
(`renderer.scene`): it appends geoms after whatever is already in the scene.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

LINE_RGBA = np.array([1.0, 0.55, 0.0, 0.9], dtype=np.float32)
CATCH_RGBA = np.array([0.2, 0.9, 0.3, 0.25], dtype=np.float32)
CMD_RGBA = np.array([0.2, 0.5, 1.0, 0.9], dtype=np.float32)
TRAIL_RGBA = np.array([1.0, 1.0, 1.0, 0.6], dtype=np.float32)

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
    mujoco.mjv_initGeom(g, kind, np.zeros(3), np.zeros(3), _EYE3, rgba)
    mujoco.mjv_connector(g, kind, width, np.asarray(a, float), np.asarray(b, float))


def draw(scn: mujoco.MjvScene, duck_xy, duck_yaw, ball_xyz, cmd, catch_radius, trail):
    """Append the debug markers to `scn` (call after the scene is otherwise built)."""
    duck = np.array([duck_xy[0], duck_xy[1], 0.01])
    ball = np.asarray(ball_xyz, dtype=float)

    # Duck -> ball line.
    _connector(scn, mujoco.mjtGeom.mjGEOM_LINE, 3.0, duck, [ball[0], ball[1], 0.01], LINE_RGBA)

    # Catch-radius disc around the ball.
    g = _add(scn)
    if g is not None:
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER,
                            np.array([catch_radius, catch_radius, 0.001]),
                            np.array([ball[0], ball[1], 0.002]), _EYE3, CATCH_RGBA)

    # Command arrow: forward speed sets the length, the turn command bends the
    # direction (where the duck will be heading ~0.5 s from now).
    vx, _, wz = (float(c) for c in cmd)
    heading = duck_yaw + 0.5 * wz
    length = 0.05 + 0.8 * max(vx, 0.0)
    tip = duck + np.array([math.cos(heading) * length, math.sin(heading) * length, 0.0])
    _connector(scn, mujoco.mjtGeom.mjGEOM_ARROW, 0.008, duck + [0, 0, 0.005], tip + [0, 0, 0.005], CMD_RGBA)

    # Trail of where the duck has been (older dots more transparent).
    n = len(trail)
    for i, (x, y) in enumerate(trail):
        g = _add(scn)
        if g is None:
            break
        rgba = TRAIL_RGBA.copy()
        rgba[3] = 0.15 + 0.5 * (i + 1) / n
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.006, 0, 0]),
                            np.array([x, y, 0.006]), _EYE3, rgba)

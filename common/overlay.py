"""Debug markers drawn into the 3D scene.

Works for both the live viewer (`viewer.user_scn`) and offscreen video
(`renderer.scene`): each function appends geoms after whatever is already there.

Markers are drawn flat on the floor with lines, which have a fixed width in
pixels, so they stay readable however far away the camera is. Each colour
means one thing in every game. The floor is a blue checkerboard, so nothing
here is blue.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

# One colour per meaning (the README's marker table and the viewer's legend say the same).
WHITE = (1.0, 1.0, 1.0, 0.9)     # the duck: where it's actually going, where it's been
PINK = (1.0, 0.3, 0.85, 0.95)    # the command sent to the legs
CYAN = (0.3, 1.0, 1.0, 0.9)      # the brain's plan: the route it means to walk
YELLOW = (1.0, 0.85, 0.1, 0.95)  # the ball: where the next kick should send it, where it's been
ORANGE = (1.0, 0.5, 0.0, 0.95)   # ...the same when the aim is "back towards the middle"
GREEN = (0.2, 0.95, 0.3, 0.6)    # goal zones: catch radius, kick spot with the ball on it
RED = (1.0, 0.25, 0.2, 0.5)      # kick spot without the ball on it
FAINT = (1.0, 1.0, 1.0, 0.3)     # the playing area

# Heights above the floor: a few mm apart so overlapping markers don't flicker
# (z-fighting), the more important ones on top.
Z_AREA, Z_ZONE, Z_TRAIL, Z_PLAN, Z_AIM, Z_CMD, Z_MOTION = 0.003, 0.004, 0.006, 0.008, 0.010, 0.012, 0.014
WIDTH = 3.0         # line width (pixels)
HEAD = 0.05         # arrowhead arm length (m)

_EYE3 = np.eye(3, dtype=np.float64).flatten()


def _add(scn: mujoco.MjvScene):
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    scn.ngeom += 1
    return g


def line(scn, a_xy, b_xy, rgba=FAINT, width=WIDTH, z=Z_AREA):
    g = _add(scn)
    if g is None:
        return
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), _EYE3, np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width, np.array([a_xy[0], a_xy[1], z]),
                         np.array([b_xy[0], b_xy[1], z]))


def polyline(scn, points, rgba, width=WIDTH, z=Z_PLAN):
    for a, b in zip(points[:-1], points[1:]):
        line(scn, a, b, rgba, width, z)


def disc(scn, xy, radius, rgba, z=Z_ZONE):
    g = _add(scn)
    if g is not None:
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([radius, radius, 0.001]),
                            np.array([xy[0], xy[1], z]), _EYE3, np.asarray(rgba, np.float32))


def sphere(scn, xyz, radius, rgba):
    g = _add(scn)
    if g is not None:
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]),
                            np.asarray(xyz, float), _EYE3, np.asarray(rgba, np.float32))


def circle(scn, centre, radius, rgba=FAINT, width=2.0, z=Z_AREA, n=48):
    a = np.linspace(0, 2 * math.pi, n + 1)
    polyline(scn, np.c_[centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a)], rgba, width, z)


def head(scn, tip_xy, direction, rgba, width=WIDTH, z=Z_PLAN, size=HEAD):
    """An arrowhead (a ">") at `tip_xy` pointing along `direction`."""
    a = math.atan2(direction[1], direction[0])
    for side in (-1, 1):
        b = a + math.pi + side * math.radians(30)
        line(scn, tip_xy, (tip_xy[0] + size * math.cos(b), tip_xy[1] + size * math.sin(b)), rgba, width, z)


def arrow(scn, points, rgba, width=WIDTH, z=Z_PLAN):
    """A straight or curved arrow along `points`."""
    polyline(scn, points, rgba, width, z)
    d = np.subtract(points[-1], points[-2])
    if np.hypot(*d) > 1e-6:
        head(scn, points[-1], d, rgba, width, z)


def route(scn, points, rgba=CYAN, width=WIDTH, z=Z_PLAN, every=0.25):
    """A planned route: a line through `points`, a small arrowhead every `every`
    metres along it showing which way it goes, and a disc where it ends."""
    pts = np.asarray(points, float)
    polyline(scn, pts, rgba, width, z)
    walked, next_head = 0.0, every
    for a, b in zip(pts[:-1], pts[1:]):
        seg = float(np.linalg.norm(b - a))
        while seg > 1e-9 and next_head <= walked + seg:
            head(scn, a + (b - a) * (next_head - walked) / seg, b - a, rgba, width, z, size=0.6 * HEAD)
            next_head += every
        walked += seg
    disc(scn, pts[-1], 0.025, (*rgba[:3], 0.5 * rgba[3]), z - 0.001)


def trail(scn, points, rgba, width=2.0, z=Z_TRAIL):
    """Where something has been: a line that fades out towards its oldest end."""
    n = len(points)
    for i in range(1, n):
        line(scn, points[i - 1], points[i], (*rgba[:3], rgba[3] * i / n), width, z)


def motion(scn, xy, yaw, v_fwd, v_left, wz, rgba, z, seconds=2.0, stub=0.15, n=12):
    """Where moving at (forward, left, turn) in the duck's frame takes it in
    `seconds`: a curved arrow. If that's shorter than `stub` (standing, or
    turning on the spot) it's a `stub`-long arrow along the heading it would
    end up with instead, so a still duck shows which way it faces."""
    pts, th = [np.array(xy, float)], yaw
    dt = seconds / n
    for _ in range(n):
        c, s = math.cos(th), math.sin(th)
        pts.append(pts[-1] + dt * np.array([c * v_fwd - s * v_left, s * v_fwd + c * v_left]))
        th += wz * dt
    if sum(np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:])) < stub:
        pts = [pts[0], pts[0] + stub * np.array([math.cos(th), math.sin(th)])]
    arrow(scn, pts, rgba, z=z)

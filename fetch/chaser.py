"""The "brain" for Stage 0 of fetch: a hand-written ball chaser.

Pollen's walking policy is the "legs": it turns a velocity command
(forward speed, sideways speed, turn rate) into motor targets. This module
decides WHAT command to send, from where the ball is relative to the duck.

No learning here: it's ~20 lines of geometry. Stage 1 replaces `chase_command`
with a small trained policy that has the same inputs and outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ChaserParams:
    # Command limits — kept inside the ranges the walking policy was trained on
    # (velocity recipe: vx in ±0.4 m/s, wz in ±1.0 rad/s). Going outside them is
    # asking the policy to do something it has never practised.
    max_forward: float = 0.4    # m/s
    # velstand has a big dead zone: from a stand it ignores vx below ~0.25 m/s
    # (and walks at only ~30% of the command once going; Pollen's own bench:
    # 0.4 -> 0.17 m/s). So any forward command is at least this. 0 = ideal legs.
    min_forward: float = 0.3    # m/s
    max_turn: float = 0.9       # rad/s
    turn_gain: float = 1.5      # rad/s of turn per rad of bearing error
    # Only walk forward when roughly facing the ball; beyond this bearing,
    # turn on the spot first.
    walk_cone: float = math.radians(60)
    # Start slowing down this far from the ball (m), measured trunk-to-ball.
    slow_radius: float = 0.35
    # Within this distance (m) the ball counts as "reached".
    catch_radius: float = 0.13


def yaw_from_quat(q) -> float:
    """Heading (rad) from a MuJoCo [w, x, y, z] quaternion."""
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def ball_in_duck_frame(duck_xy, duck_yaw, ball_xy):
    """Ball position relative to the duck: x = ahead, y = to its left (m)."""
    d = np.asarray(ball_xy, dtype=float) - np.asarray(duck_xy, dtype=float)
    c, s = math.cos(duck_yaw), math.sin(duck_yaw)
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])


def chase_command(rel_ball, p: ChaserParams = ChaserParams()):
    """Velocity command [vx, vy, wz] that walks the duck towards the ball.

    rel_ball: ball position in the duck's frame (from `ball_in_duck_frame`).
    Returns (command, distance, bearing).
    """
    dist = float(np.hypot(rel_ball[0], rel_ball[1]))
    bearing = float(math.atan2(rel_ball[1], rel_ball[0]))  # + = ball is to the left

    if dist < p.catch_radius:
        return np.zeros(3, dtype=np.float32), dist, bearing

    # Turn towards the ball, proportionally to how far off we're pointing.
    wz = float(np.clip(p.turn_gain * bearing, -p.max_turn, p.max_turn))

    # Walk forward only when facing it; fade speed in as the bearing closes
    # and fade it out as we arrive, so the duck doesn't trip over the ball.
    # The fade runs from min_forward (not 0) so the command never sits in the
    # walking policy's dead zone, where the duck just stands there.
    facing = max(0.0, 1.0 - abs(bearing) / p.walk_cone)
    arriving = float(np.clip((dist - p.catch_radius) / (p.slow_radius - p.catch_radius), 0.3, 1.0))
    speed = facing * arriving
    vx = p.min_forward + (p.max_forward - p.min_forward) * speed if speed > 0 else 0.0

    return np.array([vx, 0.0, wz], dtype=np.float32), dist, bearing

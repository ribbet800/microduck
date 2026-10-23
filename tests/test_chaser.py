"""Checks the ball-chasing "brain" on its own, with an ideal walker as the legs.

The ideal walker moves exactly as commanded (a unicycle: forward speed + turn
rate). If the chaser can't catch the ball with perfect legs, it won't with
real ones. Runs in a fraction of a second, no MuJoCo needed.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "fetch"), str(REPO / "tests")]
from chaser import ChaserParams, ball_in_duck_frame, chase_command, yaw_from_quat  # noqa: E402
from test_kick_brain import lazy_legs  # noqa: E402


def simulate(ball, yaw0=0.0, seconds=30.0, dt=0.02, p=ChaserParams(), lazy=False):
    pos, yaw, walking = np.zeros(2), yaw0, False
    for step in range(int(seconds / dt)):
        cmd, dist, _ = chase_command(ball_in_duck_frame(pos, yaw, ball), p)
        if dist < p.catch_radius:
            return step * dt
        assert abs(cmd[0]) <= p.max_forward + 1e-6 and abs(cmd[2]) <= p.max_turn + 1e-6
        vx, wz = cmd[0], cmd[2]
        if lazy:
            vx, _, wz, walking = lazy_legs(cmd, walking)
        yaw += wz * dt
        pos += vx * dt * np.array([math.cos(yaw), math.sin(yaw)])
    return None


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("angle_deg", range(-180, 180, 30))
def test_catches_ball_in_any_direction(angle_deg, lazy):
    a = math.radians(angle_deg)
    t = simulate(np.array([math.cos(a), math.sin(a)]), lazy=lazy)
    assert t is not None and t < (25.0 if lazy else 15.0)


def test_ball_straight_ahead_is_quick():
    # 1 m at 0.3-0.4 m/s with slow-down near the ball: well under 8 s.
    assert simulate(np.array([1.0, 0.0])) < 8.0


def test_never_commands_inside_the_dead_zone():
    # Any forward command must be big enough for the real walking policy to act on.
    p = ChaserParams()
    for x in np.linspace(-1.5, 1.5, 31):
        for y in np.linspace(-1.5, 1.5, 31):
            cmd, _, _ = chase_command(np.array([x, y]), p)
            assert cmd[0] == 0.0 or cmd[0] >= p.min_forward


def test_turns_towards_ball_first():
    cmd, _, bearing = chase_command(np.array([0.0, 1.0]))   # ball directly to the left
    assert bearing == pytest.approx(math.pi / 2)
    assert cmd[0] == 0.0 and cmd[2] > 0                   # turn left on the spot


def test_stops_at_ball():
    cmd, dist, _ = chase_command(np.array([0.05, 0.0]))
    assert dist < ChaserParams().catch_radius and not cmd.any()


def test_frames():
    # Duck at (1, 1) facing +y; a ball at (1, 2) is 1 m straight ahead.
    rel = ball_in_duck_frame([1, 1], math.pi / 2, [1, 2])
    assert rel == pytest.approx([1.0, 0.0], abs=1e-9)
    q = [math.cos(0.3), 0, 0, math.sin(0.3)]                 # yaw = 0.6 rad
    assert yaw_from_quat(q) == pytest.approx(0.6)

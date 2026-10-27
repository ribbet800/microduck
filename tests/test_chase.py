"""Checks the steering and the chase brain in the toy world (common/toy.py).

Perfect legs move exactly as commanded; `lazy=True` gives legs like the real
walking policy. If the chaser can't catch the ball with these, it won't with
real ones. Runs in a fraction of a second, no MuJoCo needed.
"""

import math

import numpy as np
import pytest

from chase.brain import ChaseBrain
from common.brain import Action
from common.steer import SteerParams, from_duck_frame, steer_to, to_duck_frame, yaw_from_quat
from common.toy import DT, ToyWorld


def time_to_reach(ball, lazy=False, seconds=30.0, p=SteerParams()):
    """Steer straight at `ball` from the origin; seconds until it's reached, or None."""
    w = ToyWorld(ball_xy=ball, lazy=lazy)
    for step in range(int(seconds / DT)):
        cmd, dist, _ = steer_to(to_duck_frame(w.duck, w.yaw, w.ball), p)
        if dist < p.catch_radius:
            return step * DT
        assert abs(cmd[0]) <= p.max_forward + 1e-6 and abs(cmd[2]) <= p.max_turn + 1e-6
        w.step(Action(cmd=cmd))
    return None


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("angle_deg", range(-180, 180, 30))
def test_catches_ball_in_any_direction(angle_deg, lazy):
    a = math.radians(angle_deg)
    t = time_to_reach((math.cos(a), math.sin(a)), lazy=lazy)
    assert t is not None and t < (25.0 if lazy else 15.0)


def test_ball_straight_ahead_is_quick():
    # 1 m at 0.3-0.4 m/s with slow-down near the ball: well under 8 s.
    assert time_to_reach((1.0, 0.0)) < 8.0


def test_never_commands_inside_the_dead_zone():
    # Any forward command must be big enough for the real walking policy to act on.
    p = SteerParams()
    for x in np.linspace(-1.5, 1.5, 31):
        for y in np.linspace(-1.5, 1.5, 31):
            cmd, _, _ = steer_to(np.array([x, y]), p)
            assert cmd[0] == 0.0 or cmd[0] >= p.min_forward


def test_turns_towards_ball_first():
    cmd, _, bearing = steer_to(np.array([0.0, 1.0]))   # ball directly to the left
    assert bearing == pytest.approx(math.pi / 2)
    assert cmd[0] == 0.0 and cmd[2] > 0                # turn left on the spot


def test_stops_at_ball():
    cmd, dist, _ = steer_to(np.array([0.05, 0.0]))
    assert dist < SteerParams().catch_radius and not cmd.any()


def test_frames():
    # Duck at (1, 1) facing +y; a ball at (1, 2) is 1 m straight ahead.
    rel = to_duck_frame([1, 1], math.pi / 2, [1, 2])
    assert rel == pytest.approx([1.0, 0.0], abs=1e-9)
    assert from_duck_frame([1, 1], math.pi / 2, rel) == pytest.approx([1.0, 2.0], abs=1e-9)
    q = [math.cos(0.3), 0, 0, math.sin(0.3)]                 # yaw = 0.6 rad
    assert yaw_from_quat(q) == pytest.approx(0.6)


# -- the brain, through the same Snapshot -> Action interface the game uses ----------

@pytest.mark.parametrize("lazy", [False, True])
def test_chase_brain_reaches_the_ball_and_recovers_from_a_fall(lazy):
    brain, w = ChaseBrain(), ToyWorld(ball_xy=(-0.8, 0.6), lazy=lazy)
    while w.t < 30:
        w.fallen = 1.0 <= w.t < 2.0          # knocked over for a second early on
        act = brain.update(w.snapshot())
        if w.fallen:
            assert brain.state == "RECOVER" and not act.cmd.any()
        else:
            assert brain.state == "CHASE"
        if np.linalg.norm(w.ball - w.duck) < SteerParams().catch_radius:
            break
        w.step(act)
    else:
        pytest.fail("never reached the ball")
    assert "back up" in brain.last_change


# -- the learned brain's inputs and outputs (chase/brain.py) -------------------------

def test_learned_brain_sees_the_ball_in_its_own_frame():
    from chase.brain import OBS_SIZE, observe
    w = ToyWorld(yaw=math.pi / 2, ball_xy=(0.0, 1.0))         # facing +y, ball 1 m straight ahead
    o = observe(w.snapshot(), np.zeros(3))
    assert o.shape == (OBS_SIZE,) and o.dtype == np.float32
    assert o[:5] == pytest.approx([0.5, 0.0, 0.5, 0.0, 1.0], abs=1e-6)
    assert np.all(np.abs(o) <= 3.0)                           # inside the training env's observation space


def test_learned_brain_commands_stay_in_the_trained_ranges():
    from common.brain import MAX_FORWARD, MAX_TURN, to_command
    assert to_command([-1, 0]) == pytest.approx([0.0, 0.0, 0.0])
    assert to_command([1, 1]) == pytest.approx([MAX_FORWARD, 0.0, MAX_TURN])
    assert to_command([5, -5]) == pytest.approx([MAX_FORWARD, 0.0, -MAX_TURN])   # clipped


def test_training_env_steps():
    pytest.importorskip("gymnasium")                          # only with: uv run --group train pytest
    from chase.train import ChaseEnv
    env = ChaseEnv()
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    total = 0.0
    for _ in range(300):                                      # a whole 30 s episode
        obs, r, terminated, truncated, info = env.step(np.array([1.0, 0.0], np.float32))
        total += r
    assert truncated and np.isfinite(total)

"""Checks the kick-loop brain with an imaginary perfect duck.

The ideal duck moves exactly as commanded (forward, sideways, turn) and has a
"kick" that launches the ball straight ahead at 1 m/s — but only if the ball is
genuinely on the kick spot, like Pollen's blind kick policy. If the brain can't
score with perfect legs, it won't with real ones.

`lazy=True` gives it legs like Pollen's velstand in MuJoCo instead (see
`lazy_legs`): a big dead zone, no sidestep, no walking backwards, ~30% speed.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "kick")]
from brain import KickBrain, KickParams, Snapshot  # noqa: E402

DT = 0.02


def lazy_legs(cmd, walking):
    """What velstand actually does with a command (measured in MuJoCo with BAM,
    2026-10-23): it only starts walking at vx >= ~0.25 m/s or |wz| >= ~0.7 rad/s,
    keeps going down to vx 0.15 once walking, never sidesteps or walks
    backwards, and reaches ~30% of the commanded forward speed.
    Returns (vx, vy, wz achieved, walking)."""
    vx, _, wz = (float(c) for c in cmd)
    if vx >= (0.15 if walking else 0.25):
        return 0.3 * vx, 0.0, 0.8 * wz, True
    if abs(wz) >= 0.7:
        return 0.0, 0.0, 0.8 * wz, True
    return 0.0, 0.0, 0.0, False


class IdealWorld:
    def __init__(self, duck_xy=(-0.4, 0.0), yaw=0.0, ball_xy=(0.0, 0.0), kick_noise=0.0, seed=0, lazy=False):
        self.lazy, self.walking = lazy, False
        self.duck = np.array(duck_xy, float)
        self.yaw = yaw
        self.ball = np.array(ball_xy, float)
        self.ball_v = np.zeros(2)
        self.kick_left_s = 0.0
        self.t = 0.0
        self.rng = np.random.default_rng(seed)
        self.kick_noise = kick_noise
        self.min_ball_gap = np.inf

    def snap(self):
        return Snapshot(self.t, self.duck.copy(), self.yaw, self.ball.copy(),
                        float(np.linalg.norm(self.ball_v)), False, self.kick_left_s > 0)

    def apply(self, act, brain):
        if act.place_ball is not None:
            self.ball = np.array(act.place_ball, float)
            self.ball_v[:] = 0
        if act.kick:
            rel_spot = brain.spot_rel(act.kick)
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            d = self.ball - self.duck
            rel = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
            if np.linalg.norm(rel - rel_spot) < 0.02:     # foot connects
                ang = self.yaw + self.rng.normal(0, self.kick_noise)
                self.ball_v = 1.0 * np.array([math.cos(ang), math.sin(ang)])
            self.kick_left_s = 3.0
        if self.kick_left_s > 0:
            self.kick_left_s -= DT
        else:
            vx, vy, wz = act.cmd
            if self.lazy:
                vx, vy, wz, self.walking = lazy_legs(act.cmd, self.walking)
            self.yaw += wz * DT
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            self.duck += DT * np.array([c * vx - s * vy, s * vx + c * vy])
        # Ball rolls and slows down (~0.4 m/s^2).
        sp = np.linalg.norm(self.ball_v)
        if sp > 0:
            self.ball_v *= max(0.0, sp - 0.4 * DT) / sp
        self.ball += self.ball_v * DT
        self.t += DT
        if brain.state == "APPROACH" and not brain.events["kicks"]:
            self.min_ball_gap = min(self.min_ball_gap, np.linalg.norm(self.duck - self.ball))


def play(world, seconds, params=None, feet=("right",)):
    brain = KickBrain(params or KickParams(), list(feet))
    for _ in range(int(seconds / DT)):
        act = brain.update(world.snap())
        world.apply(act, brain)
    return brain


def test_stance_puts_ball_on_the_spot():
    b = KickBrain(KickParams(), ["right", "left"])
    ball = np.array([0.3, -0.2])
    for foot in ("right", "left"):
        stance, a = b.stance_for(ball, foot)
        yaw = math.atan2(a[1], a[0])
        d = ball - stance
        rel = [math.cos(yaw) * d[0] + math.sin(yaw) * d[1], -math.sin(yaw) * d[0] + math.cos(yaw) * d[1]]
        assert rel == pytest.approx(b.spot_rel(foot), abs=1e-9)


def test_lines_up_and_kicks_without_assist():
    brain = play(IdealWorld(), 25)
    kicks = brain.events["kicks"]
    assert kicks, "never kicked"
    assert not any(k["assisted"] for k in kicks)
    assert all(k["err_m"] <= KickParams().tolerance for k in kicks)


def test_scores_goals_both_ways():
    brain = play(IdealWorld(), 120)
    goals = brain.events["goals"]
    assert len(goals) >= 3, brain.events
    assert {g["goal"] for g in goals} == {1, -1}        # attacked both ends


def test_walks_round_the_ball_when_on_the_wrong_side():
    # Duck starts between the ball and the goal it's attacking.
    w = IdealWorld(duck_xy=(0.5, 0.0), yaw=math.pi, ball_xy=(0.0, 0.0))
    brain = play(w, 30)
    assert brain.events["kicks"], "never kicked"
    assert w.min_ball_gap > 0.15        # never walked through the ball


def test_uses_both_feet_when_available():
    brain = play(IdealWorld(), 120, feet=("left", "right"))
    assert {k["foot"] for k in brain.events["kicks"]} == {"left", "right"}


def play_sloppy(params, seconds=30):
    """A sloppy duck: the first time it starts lining up it ends up 2.5 cm off
    to the side, and it walks in dead straight (no steering, no sidestep), so
    that attempt can never line up exactly."""
    w = IdealWorld()
    brain = KickBrain(params, ["right"])
    shoved = False
    for _ in range(int(seconds / DT)):
        act = brain.update(w.snap())
        if brain.state == "ALIGN":
            if not shoved:
                w.duck += 0.025 * np.array([-math.sin(w.yaw), math.cos(w.yaw)])
                shoved = True
            act.cmd[1] = 0.0
            if act.cmd[0]:
                act.cmd[2] = 0.0
        w.apply(act, brain)
    return brain


def test_assist_nudges_ball_when_duck_cant_get_precise():
    brain = play_sloppy(KickParams(assist=True))
    kicks = brain.events["kicks"]
    assert kicks and kicks[0]["assisted"], brain.events
    assert kicks[0]["err_m"] > KickParams().tolerance


def test_never_kicks_unaligned_without_assist():
    brain = play_sloppy(KickParams())
    assert brain.events["retries"], "should have given up on the sloppy attempt"
    assert all(k["err_m"] <= KickParams().tolerance and not k["assisted"] for k in brain.events["kicks"])


@pytest.mark.parametrize("start", [(-0.45, 0.0, 0.0), (-0.3, 0.3, -1.0), (0.5, 0.0, math.pi), (0.2, -0.4, 2.0)])
def test_scores_with_lazy_legs(start):
    # Legs like the real walking policy: dead zone, no sidestep, no reverse.
    x, y, yaw = start
    brain = play(IdealWorld(duck_xy=(x, y), yaw=yaw, lazy=True), 120, feet=("left", "right"))
    kicks = brain.events["kicks"]
    assert kicks, f"never kicked: {brain.events}"
    assert all(k["err_m"] <= KickParams().tolerance for k in kicks)
    assert brain.events["goals"], brain.events

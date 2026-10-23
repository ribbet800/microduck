"""Checks the kick-loop brain with an imaginary perfect duck.

The ideal duck moves exactly as commanded (forward, sideways, turn) and has a
"kick" that launches the ball straight ahead at 1 m/s — but only if the ball is
genuinely on the kick spot, like Pollen's blind kick policy. If the brain can't
score with perfect legs, it won't with real ones.
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


class IdealWorld:
    def __init__(self, duck_xy=(-0.4, 0.0), yaw=0.0, ball_xy=(0.0, 0.0), kick_noise=0.0, seed=0):
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


def test_assist_nudges_ball_when_duck_cant_get_precise():
    # A sloppy duck: it can only stop within ~3 cm, so it never lines up exactly.
    p = KickParams(tolerance=0.012, assist=True, align_timeout_s=3.0)
    w = IdealWorld()
    brain = KickBrain(p, ["right"])
    for _ in range(int(20 / DT)):
        act = brain.update(w.snap())
        if brain.state == "ALIGN":
            act.cmd = np.zeros(3, dtype=np.float32)       # refuses to shuffle
        w.apply(act, brain)
    # Either it kicked with an assist, or it retried; it must never kick unaligned.
    for k in brain.events["kicks"]:
        assert k["assisted"] or k["err_m"] <= p.tolerance

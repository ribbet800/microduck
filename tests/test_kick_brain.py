"""Checks the kick-loop brain with an imaginary duck.

The ideal duck moves exactly as commanded (forward, sideways, turn) and has a
"kick" that launches the ball straight ahead at 1 m/s — but only if the ball is
genuinely on the kick spot, like Pollen's blind kick policy. If the brain can't
kick well with perfect legs, it won't with real ones.

`lazy=True` gives it legs like Pollen's velstand in MuJoCo instead (see
tests/legs.py): a big dead zone, no sidestep, no walking backwards, ~30% speed.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "kick"), str(REPO / "tests")]
from brain import KickBrain, KickParams  # noqa: E402
from common.brain import Snapshot  # noqa: E402
from legs import lazy_legs  # noqa: E402

DT = 0.02


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
        self.max_ball_r = 0.0

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
        self.max_ball_r = max(self.max_ball_r, float(np.linalg.norm(self.ball)))
        if brain.state == "APPROACH" and not brain.events["kicks"]:
            self.min_ball_gap = min(self.min_ball_gap, np.linalg.norm(self.duck - self.ball))


def play(world, seconds, params=None, feet=("right",), seed=0):
    brain = KickBrain(params or KickParams(), list(feet), rng=np.random.default_rng(seed))
    for _ in range(int(seconds / DT)):
        act = brain.update(world.snap())
        world.apply(act, brain)
    return brain


def finished(brain):
    return [k for k in brain.events["kicks"] if k.get("aim_err_deg") is not None]


def snap_at(duck_xy, ball_xy):
    return Snapshot(0.0, np.array(duck_xy, float), 0.0, np.array(ball_xy, float), 0.0, False, False)


# -- aiming ---------------------------------------------------------------------

def test_aims_are_random_and_roughly_onwards():
    # Inside the arena: duck->ball direction (here +x) give or take aim_spread.
    b = KickBrain(KickParams(), ["right"], rng=np.random.default_rng(1))
    angles = []
    for _ in range(300):
        b.pick_aim(snap_at((-0.5, 0.0), (0.0, 0.0)))
        angles.append(b.aim_deg)
        assert b.aim_mode == "free"
    assert max(abs(a) for a in angles) <= 60 + 1e-9
    assert min(angles) < -45 and max(angles) > 45      # really spread out, both ways


def test_aims_back_home_outside_the_arena():
    b = KickBrain(KickParams(), ["right"], rng=np.random.default_rng(2))
    for _ in range(300):
        # Ball out to the right of the arena, duck behind it pointing further out.
        b.pick_aim(snap_at((1.5, 0.0), (2.0, 0.0)))
        assert b.aim_mode == "home"
        off = abs((b.aim_deg - 180 + 180) % 360 - 180)  # angle from straight home (-x)
        assert off <= 45 + 1e-9


def test_same_seed_same_aims():
    def aims(seed):
        brain = play(IdealWorld(), 60, seed=seed)
        return [k["aim_deg"] for k in brain.events["kicks"]]
    assert aims(7) == aims(7)
    assert aims(7) != aims(8)


def test_stance_puts_ball_on_the_spot():
    b = KickBrain(KickParams(), ["right", "left"])
    ball = np.array([0.3, -0.2])
    b.pick_aim(snap_at((0.0, 0.0), ball))
    for foot in ("right", "left"):
        stance, a = b.stance_for(ball, foot), b.aim
        yaw = math.atan2(a[1], a[0])
        d = ball - stance
        rel = [math.cos(yaw) * d[0] + math.sin(yaw) * d[1], -math.sin(yaw) * d[0] + math.cos(yaw) * d[1]]
        assert rel == pytest.approx(b.spot_rel(foot), abs=1e-9)


# -- playing ----------------------------------------------------------------------

def test_lines_up_and_kicks_without_assist():
    brain = play(IdealWorld(), 25)
    kicks = brain.events["kicks"]
    assert kicks, "never kicked"
    assert not any(k["assisted"] for k in kicks)
    assert all(k["err_m"] <= KickParams().tolerance for k in kicks)


def test_keeps_kicking_and_the_ball_goes_where_aimed():
    brain = play(IdealWorld(), 120, feet=("left", "right"))
    done = finished(brain)
    assert len(done) >= 5, brain.events
    assert all(abs(k["aim_err_deg"]) < 10 for k in done)       # perfect legs, no kick noise
    assert len({round(k["aim_deg"]) for k in done}) == len(done)  # a new direction every time


def test_soft_arena_keeps_the_ball_nearby():
    w = IdealWorld(kick_noise=0.15)
    brain = play(w, 400, feet=("left", "right"), seed=3)
    assert len(brain.events["kicks"]) >= 15
    assert any(k["aim_mode"] == "home" for k in brain.events["kicks"]), "never strayed: test is too short"
    # An ideal kick rolls ~1.25 m, so the ball can end up at most about that far outside.
    assert w.max_ball_r < KickParams().arena_radius + 1.5


def test_walks_round_the_ball_when_on_the_wrong_side():
    # Ball outside the arena, so the aim is back home (-x); the duck starts on the
    # home side of the ball, i.e. right where the ball is about to be kicked.
    w = IdealWorld(duck_xy=(1.6, 0.0), yaw=0.0, ball_xy=(2.0, 0.0))
    brain = play(w, 40)
    assert brain.events["kicks"], "never kicked"
    assert brain.events["kicks"][0]["aim_mode"] == "home"
    assert w.min_ball_gap > 0.12        # never walked through the ball


@pytest.mark.parametrize("foot", ["left", "right"])
def test_picks_the_foot_whose_stance_is_nearer(foot):
    b = KickBrain(KickParams(), ["left", "right"])
    ball = np.zeros(2)
    b.pick_aim(snap_at((-0.5, 0.0), ball))
    stance, a = b.stance_for(ball, foot), b.aim
    b._approach(Snapshot(0.0, stance - 0.3 * a, float(math.atan2(a[1], a[0])), ball, 0.0, False, False))
    assert b.foot == foot


def test_aims_again_when_the_ball_is_knocked():
    w = IdealWorld()
    brain = KickBrain(KickParams(), ["right"])
    for _ in range(int(2 / DT)):
        w.apply(brain.update(w.snap()), brain)
    first = brain.aim.copy()
    w.ball += np.array([0.0, 0.3])           # someone shoves the ball
    for _ in range(3):
        w.apply(brain.update(w.snap()), brain)
    assert brain.aim_from == pytest.approx(w.ball, abs=1e-6)
    assert not np.allclose(brain.aim, first)


def test_long_walk_to_the_ball_is_not_a_timeout():
    # Lazy legs (~0.1 m/s) from 4 m away: ~40 s of walking, then it must still kick.
    w = IdealWorld(duck_xy=(-4.0, 0.0), ball_xy=(0.0, 0.0), lazy=True)
    brain = play(w, 90, feet=("left", "right"))
    assert not brain.events["timeouts"]
    assert brain.events["kicks"]


@pytest.mark.parametrize("start", [(-0.45, 0.0, 0.0), (-0.3, 0.3, -1.0), (0.5, 0.0, math.pi), (0.2, -0.4, 2.0)])
def test_keeps_kicking_with_lazy_legs(start):
    # Legs like the real walking policy: dead zone, no sidestep, no reverse.
    x, y, yaw = start
    brain = play(IdealWorld(duck_xy=(x, y), yaw=yaw, lazy=True), 150, feet=("left", "right"))
    kicks = brain.events["kicks"]
    assert len(kicks) >= 3, f"only {len(kicks)} kicks: {brain.events}"
    assert all(k["err_m"] <= KickParams().tolerance for k in kicks)
    assert all(abs(k["aim_err_deg"]) < 15 for k in finished(brain))


# -- assist -----------------------------------------------------------------------

def play_sloppy(params, seconds=40):
    """A sloppy duck: the first time it lines up, the ball ends up 2.5 cm off
    to the side while it settles (as if it stopped badly), so that attempt can
    never kick from the spot."""
    w = IdealWorld()
    brain = KickBrain(params, ["right"])
    shoved = False
    for _ in range(int(seconds / DT)):
        act = brain.update(w.snap())
        if brain.state == "SETTLE" and not shoved:
            w.ball += 0.025 * np.array([-math.sin(w.yaw), math.cos(w.yaw)])
            shoved = True
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


def test_kick_that_never_starts_does_not_hang():
    # Legs that ignore the kick request: the brain must give up and try again.
    w, brain, reasons = IdealWorld(), KickBrain(KickParams(), ["right"]), set()
    for _ in range(int(25 / DT)):
        act = brain.update(w.snap())
        act.kick = None
        w.apply(act, brain)
        reasons.add(brain.last_change)
    assert any("never started" in r for r in reasons)
    assert not brain.events["kicks"]


def test_falls_go_to_recover_and_back():
    brain = KickBrain(KickParams(), ["right"])
    s = snap_at((-0.4, 0.0), (0.0, 0.0))
    brain.update(s)
    s.fallen, s.t = True, 1.0
    assert not brain.update(s).cmd.any() and brain.state == "RECOVER"
    s.fallen, s.t = False, 3.0
    brain.update(s)
    assert brain.state == "APPROACH" and "back up" in brain.last_change

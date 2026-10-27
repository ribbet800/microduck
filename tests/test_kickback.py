"""Checks the kickback brain in the toy world (common/toy.py).

The toy kick launches the ball straight ahead at 1 m/s, but only if the ball is
genuinely on the kick spot, like Pollen's blind kick policy. If the brain can't
kick well with perfect legs, it won't with real ones. `lazy=True` gives legs
like Pollen's walking policy in MuJoCo instead: a big dead zone, no sidestep,
no walking backwards, ~30% speed.
"""

import math

import numpy as np
import pytest

from common.brain import Action, Snapshot
from common.steer import to_duck_frame
from common.toy import DT, ToyWorld
from kickback.brain import KickbackBrain, KickbackParams, LearnedKickbackBrain

ENDS = ((-0.5, 0.0), (0.5, 0.0))
TOL = KickbackParams().tolerance


def play(world, seconds, brain=None, feet=("left", "right")):
    """Run a kickback brain in `world`. Also returns the closest the duck came to
    the ball while walking to its first kick."""
    brain = brain or KickbackBrain(ENDS, feet=feet)
    first = len(brain.events["kicks"])
    closest = np.inf
    for _ in range(int(seconds / DT)):
        world.step(brain.update(world.snapshot()))
        if brain.state == "APPROACH" and len(brain.events["kicks"]) == first:
            closest = min(closest, float(np.linalg.norm(world.duck - world.ball)))
    return brain, closest


def finished(brain):
    return [k for k in brain.events["kicks"] if k.get("aim_err_deg") is not None]


def snap_at(duck_xy, ball_xy, yaw=0.0):
    return Snapshot(0.0, np.array(duck_xy, float), yaw, np.array(ball_xy, float), 0.0, False)


# -- aiming and geometry -----------------------------------------------------------

@pytest.mark.parametrize("duck_x, first", [(-0.45, +1.0), (0.45, -1.0)])
def test_first_kick_goes_away_from_the_duck(duck_x, first):
    b = KickbackBrain(ENDS)
    b.pick_aim(snap_at((duck_x, 0.0), (0.0, 0.0)))
    assert b.aim == pytest.approx([first, 0.0])


@pytest.mark.parametrize("ball", [(0.7, 0.2), (0.1, 0.2)])
def test_kicks_back_towards_the_line_whatever_the_roll(ball):
    # The last kick went +x. Whether the ball rolled past the right end or stopped
    # short of it (and drifted off the line), the next kick goes back to the left end.
    b = KickbackBrain(ENDS)
    b.events["kicks"].append({"aim": [1.0, 0.0]})
    b.pick_aim(snap_at((0.0, 0.0), ball))
    d = np.array([-0.5, 0.0]) - np.array(ball)
    assert b.aim == pytest.approx(d / np.linalg.norm(d))


def test_stance_puts_ball_on_the_spot():
    b = KickbackBrain(ENDS)
    ball = np.array([0.3, -0.2])
    b.pick_aim(snap_at((0.0, 0.0), ball))
    for foot in ("right", "left"):
        stance, a = b.stance_for(ball, foot), b.aim
        rel = to_duck_frame(stance, math.atan2(a[1], a[0]), ball)
        assert rel == pytest.approx(b.spot_rel(foot), abs=1e-9)


def test_orbit_route_goes_the_short_way_and_clears_the_ball():
    b = KickbackBrain(ENDS, feet=["right"])
    ball = np.zeros(2)
    b.aim, b.aim_from = np.array([1.0, 0.0]), ball      # kick towards +x
    # The duck is in front of the ball, a little to its left (+y): the short way
    # round is anticlockwise, over the left side.
    b._approach(snap_at((0.35, 0.1), ball, yaw=math.pi))
    assert b._orbit_dir == 1.0
    route = b.plan[1:-2]                       # the orbit points (the last two are pre-stance, stance)
    assert all(p[1] >= -1e-9 for p in route)   # stays on the left side
    assert min(np.linalg.norm(p - ball) for p in route) > 0.35
    assert b.plan[-1] == pytest.approx(b.stance_for(ball, "right"))
    # Straight in when it's already on the line behind the ball (a fresh brain: not mid-orbit).
    b = KickbackBrain(ENDS, feet=["right"])
    b.aim, b.aim_from = np.array([1.0, 0.0]), ball
    pre = b.stance_for(ball, "right") - KickbackParams().standoff * b.aim
    b._approach(snap_at(pre - 0.3 * b.aim, ball))
    assert b._orbit_dir == 0.0 and len(b.plan) == 3


@pytest.mark.parametrize("foot", ["left", "right"])
def test_picks_the_foot_whose_stance_is_nearer(foot):
    b = KickbackBrain(ENDS)
    ball = np.zeros(2)
    b.pick_aim(snap_at((-0.5, 0.0), ball))
    stance, a = b.stance_for(ball, foot), b.aim
    b._approach(snap_at(stance - 0.3 * a, ball, yaw=float(math.atan2(a[1], a[0]))))
    assert b.foot == foot


# -- playing ----------------------------------------------------------------------

def test_lines_up_and_kicks():
    brain, _ = play(ToyWorld(duck_xy=(-0.4, 0.0), ball_xy=(0.0, 0.0)), 25, feet=["right"])
    kicks = brain.events["kicks"]
    assert kicks, "never kicked"
    assert all(k["err_m"] <= TOL for k in kicks)


@pytest.mark.parametrize("lazy", [False, True])
def test_goes_back_and_forth_along_the_line(lazy):
    w = ToyWorld(duck_xy=(-0.45, 0.0), ball_xy=(0.0, 0.0), kick_noise=0.05, lazy=lazy)
    brain, _ = play(w, 200 if lazy else 100)
    kicks = brain.events["kicks"]
    assert len(kicks) >= 4, brain.events
    signs = [np.sign(k["aim"][0]) for k in kicks]
    assert all(a != b for a, b in zip(signs, signs[1:])), signs      # alternates
    assert all(abs(k["end_xy"][1]) < 0.3 for k in finished(brain))   # stays near the line
    assert all(abs(k["aim_err_deg"]) < 15 for k in finished(brain))


@pytest.mark.parametrize("lazy", [False, True])
def test_walks_round_the_ball_when_on_the_wrong_side(lazy):
    # The last kick went -x, so the next goes +x; the duck stands on the +x side
    # of the ball, i.e. right where it's about to be kicked. It must orbit round
    # the ball at a distance, not walk into it, and still kick.
    w = ToyWorld(duck_xy=(0.4, 0.05), ball_xy=(0.0, 0.0), lazy=lazy)
    brain = KickbackBrain(ENDS)
    brain.events["kicks"].append({"aim": [-1.0, 0.0]})
    brain, closest = play(w, 60 if lazy else 30, brain)
    assert len(brain.events["kicks"]) > 1, "never kicked"
    assert brain.events["orbits"]
    assert closest > 0.2, closest


@pytest.mark.parametrize("start", [(-0.45, 0.0, 0.0), (-0.3, 0.3, -1.0), (0.5, 0.0, math.pi), (0.2, -0.4, 2.0)])
def test_keeps_kicking_with_lazy_legs(start):
    x, y, yaw = start
    brain, _ = play(ToyWorld(duck_xy=(x, y), yaw=yaw, ball_xy=(0.0, 0.0), lazy=True), 150)
    kicks = brain.events["kicks"]
    assert len(kicks) >= 3, f"only {len(kicks)} kicks: {brain.events}"
    assert all(k["err_m"] <= TOL for k in kicks)
    assert all(abs(k["aim_err_deg"]) < 15 for k in finished(brain))


def test_toy_duck_pushes_the_ball_it_walks_into():
    # Ball on the kick spot doesn't touch the feet; walking on pushes it ahead.
    w = ToyWorld(duck_xy=(0.0, 0.0), ball_xy=(0.09, 0.042))
    w.step(Action())
    assert w.ball == pytest.approx([0.09, 0.042])
    for _ in range(int(1 / DT)):
        w.step(Action(cmd=np.array([0.3, 0.0, 0.0])))
    assert w.ball[0] > w.duck[0] + 0.065 and w.ball[0] > 0.3
    # Turning on the spot with the ball on the kick spot sweeps it with a foot
    # (what knocked it in MuJoCo after retries: plans.md §2).
    w = ToyWorld(duck_xy=(0.0, 0.0), ball_xy=(0.09, 0.042))
    for _ in range(int(1 / DT)):
        w.step(Action(cmd=np.array([0.0, 0.0, -1.0])))
    assert np.linalg.norm(w.ball - [0.09, 0.042]) > 0.005


def test_aims_again_when_the_ball_is_knocked():
    w, brain = ToyWorld(duck_xy=(-0.4, 0.0), ball_xy=(0.0, 0.0)), KickbackBrain(ENDS)
    for _ in range(int(2 / DT)):
        w.step(brain.update(w.snapshot()))
    w.ball += np.array([0.0, 0.3])           # someone shoves the ball
    for _ in range(3):
        w.step(brain.update(w.snapshot()))
    assert brain.aim_from == pytest.approx(w.ball, abs=1e-6)
    assert "knocked" in brain.last_change


def test_long_walk_to_the_ball_is_not_a_timeout():
    # Lazy legs (~0.1 m/s) from 4 m away: ~40 s of walking, then it must still kick.
    brain, _ = play(ToyWorld(duck_xy=(-4.0, 0.0), ball_xy=(0.0, 0.0), lazy=True), 90)
    assert not brain.events["timeouts"]
    assert brain.events["kicks"]


def test_never_kicks_off_the_spot():
    # The first time it lines up, the ball ends up 2.5 cm off to the side while it
    # settles (as if it stopped badly): it must not kick from there.
    w, brain, shoved = ToyWorld(duck_xy=(-0.4, 0.0), ball_xy=(0.0, 0.0)), KickbackBrain(ENDS, feet=["right"]), False
    for _ in range(int(40 / DT)):
        act = brain.update(w.snapshot())
        if brain.state == "SETTLE" and not shoved:
            w.ball += 0.025 * np.array([-math.sin(w.yaw), math.cos(w.yaw)])
            shoved = True
        w.step(act)
    assert shoved
    assert all(k["err_m"] <= TOL for k in brain.events["kicks"])


def test_kick_that_never_starts_does_not_hang():
    # A world that ignores the kick request: the brain must give up and try again.
    w, brain, reasons = ToyWorld(duck_xy=(-0.4, 0.0), ball_xy=(0.0, 0.0)), KickbackBrain(ENDS), set()
    for _ in range(int(25 / DT)):
        act = brain.update(w.snapshot())
        act.skill = None
        w.step(act)
        reasons.add(brain.last_change)
    assert any("never started" in r for r in reasons)
    assert len(brain.events["kicks"]) <= 1      # failed kicks aren't counted (bar one still waiting)


def test_learned_brain_stops_when_lined_up_and_kicks():
    # A "network" that always says full speed ahead: the brain must stop it by itself
    # once the ball is on the spot, then hand over to the usual settle and kick.
    brain = LearnedKickbackBrain(ENDS, net=lambda obs: np.array([1.0, 0.0]), feet=["right"])
    brain.aim, brain.aim_from = np.array([1.0, 0.0]), np.zeros(2)
    stance = brain.stance_for(np.zeros(2), "right")
    w = ToyWorld(duck_xy=stance - [0.1, 0.0], ball_xy=(0.0, 0.0))    # walking in along the aim line
    for _ in range(int(8 / DT)):
        w.step(brain.update(w.snapshot()))
    kicks = brain.events["kicks"]
    assert kicks and kicks[0]["err_m"] <= TOL and not brain.events["knocks"]
    assert w.ball[0] > 1.0                                             # kicked along the aim


def test_falls_go_to_recover_and_back():
    brain = KickbackBrain(ENDS)
    s = snap_at((-0.4, 0.0), (0.0, 0.0))
    brain.update(s)
    s.fallen, s.t = True, 1.0
    assert not brain.update(s).cmd.any() and brain.state == "RECOVER"
    s.fallen, s.t = False, 3.0
    brain.update(s)
    assert brain.state == "APPROACH" and "back up" in brain.last_change

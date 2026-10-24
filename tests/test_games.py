"""Smoke-tests both games (rules, logging, status panel, summary) by running them
on a stand-in for `common.duck.Duck` backed by the toy world. Catches broken
wiring between a game, its brain and the shared loop without starting MuJoCo.
(Drawing needs a real MuJoCo scene, so it isn't covered.)
"""

import numpy as np

from chase.brain import ChaseBrain
from chase.game import Chase
from common.brain import Action
from common.duck import STAND_Z
from common.toy import ToyWorld
from kickback.brain import KickbackBrain
from kickback.game import Kickback


class ToyDuck:
    """The parts of `Duck`'s API that `Game` uses, on top of a ToyWorld."""

    def __init__(self, lazy=True):
        self.w = ToyWorld(lazy=lazy)
        self._start = None

    def reset(self, xy=(0.0, 0.0), yaw=0.0):
        self.w.duck, self.w.yaw = np.array(xy, float), float(yaw)

    def place_ball(self, x, y):
        self.w.ball, self.w.ball_v = np.array([x, y], float), np.zeros(2)

    xy = property(lambda self: self.w.duck.copy())
    z = property(lambda self: STAND_Z)
    yaw = property(lambda self: self.w.yaw)
    ball = property(lambda self: np.array([*self.w.ball, 0.035]))
    ball_speed = property(lambda self: float(np.linalg.norm(self.w.ball_v)))
    skill = property(lambda self: self.w.skill)

    def speeds(self):
        return 0.0, 0.0

    def start_skill(self, name):
        self._start = name

    def stop_skill(self):
        self.w.skill = None

    def step(self, cmd=(0.0, 0.0, 0.0)):
        self.w.step(Action(cmd=np.asarray(cmd, dtype=np.float32), skill=self._start))
        self._start = None


def run_game(game, seconds):
    game.start()
    for _ in range(int(seconds / 0.02)):
        game.tick()
    assert game.status() and game.log["t"]
    return game.summary()


def test_chase_game_runs_and_scores():
    s = run_game(Chase(ToyDuck(), ChaseBrain(), rng=np.random.default_rng(0)), 60)
    assert s["balls_reached"] >= 2 and s["falls"] == 0


def test_kickback_game_runs_and_scores():
    s = run_game(Kickback(ToyDuck(), KickbackBrain(ends=[(-0.5, 0.0), (0.5, 0.0)])), 120)
    assert s["kicks"] >= 2 and s["mean_off_line_m"] is not None

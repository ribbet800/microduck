"""A toy world for brains: a duck and a ball in plain numpy, no MuJoCo.

Fast enough to run thousands of times a second, so it's where brains are tested
(tests/) and where a learned brain can be trained before trying it in MuJoCo.
It speaks the same Snapshot -> Action interface as the real game loop
(common/game.py), so a brain can't tell the difference:

    world = ToyWorld(duck_xy=(-0.4, 0), lazy=True)
    for _ in range(steps):
        world.step(brain.update(world.snapshot()))

Legs: perfect (move exactly as commanded) or `lazy`, like Pollen's walking
policy measured in MuJoCo. Skills: the duck stands still while one runs
(SKILL_SECONDS). Kicks are like Pollen's blind kick policy: the ball flies
straight ahead at 1 m/s only if it's on the kick spot. The ball rolls and slows down.
The duck's feet are a rectangle: walk or turn into the ball and it gets pushed.
"""

from __future__ import annotations

import math

import numpy as np

from common.brain import Action, Snapshot
from common.steer import to_duck_frame

DT = 0.02
SKILL_SECONDS = {"kick_left": 3.0, "kick_right": 3.0}   # how long each skill runs (the real ones: common/duck.py)
# Where each kick skill needs the ball, in the duck's frame (x ahead, y left): what
# Pollen's kick was trained on (microduck_ball_kick_env_cfg.py), and how close is close enough.
KICK_SPOTS = {"kick_left": (0.09, 0.042), "kick_right": (0.09, -0.042)}
KICK_REACH = 0.02
BALL_DECEL = 0.4   # m/s^2
BALL_R = 0.035     # Pollen's 70 mm ball
# What of the duck can touch the ball (it's 7 cm tall), in its frame, measured standing
# in MuJoCo: feet and shins. The kick spot is ~2 cm clear of the toes.
FEET_X, FEET_Y = (-0.033, 0.035), 0.063


def push_ball(duck, yaw, v, wz, ball, ball_v):
    """If the ball overlaps the duck's feet, move it out and give it at least the
    feet's speed there (a dead push, no bounce). Returns (ball, ball_v)."""
    rel = to_duck_frame(duck, yaw, ball)
    near = np.array([np.clip(rel[0], *FEET_X), np.clip(rel[1], -FEET_Y, FEET_Y)])
    gap = rel - near
    dist = float(np.linalg.norm(gap))
    if dist >= BALL_R:
        return ball, ball_v
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    n = rot @ (gap / dist if dist > 1e-9 else np.array([1.0, 0.0]))   # out of the feet, in the world
    ball = ball + (BALL_R - dist) * n
    p = rot @ near
    v_feet = v + wz * np.array([-p[1], p[0]])                         # the feet's velocity at that point
    closing = float(np.dot(ball_v - v_feet, n))
    if closing < 0:
        ball_v = ball_v - closing * n
    return ball, ball_v


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


class ToyWorld:
    def __init__(self, duck_xy=(0.0, 0.0), yaw=0.0, ball_xy=(1.0, 0.0), lazy=False, kick_noise=0.0, seed=0):
        self.duck = np.array(duck_xy, float)
        self.yaw = float(yaw)
        self.ball = np.array(ball_xy, float)
        self.ball_v = np.zeros(2)
        self.lazy, self.walking = lazy, False
        self.kick_noise, self.rng = kick_noise, np.random.default_rng(seed)
        self.fallen = False              # set it to knock the duck over: commands are ignored
        self.skill, self.skill_left = None, 0.0
        self.t = 0.0

    def snapshot(self) -> Snapshot:
        return Snapshot(self.t, self.duck.copy(), self.yaw, self.ball.copy(),
                        float(np.linalg.norm(self.ball_v)), self.fallen, self.skill)

    def step(self, act: Action):
        if act.stop_skill:
            self.skill = None
        if act.skill:
            self.skill, self.skill_left = act.skill, SKILL_SECONDS[act.skill]
            rel = to_duck_frame(self.duck, self.yaw, self.ball)
            spot = KICK_SPOTS.get(act.skill)
            if spot is not None and np.linalg.norm(rel - spot) < KICK_REACH:     # foot connects
                ang = self.yaw + self.rng.normal(0, self.kick_noise)
                self.ball_v = np.array([math.cos(ang), math.sin(ang)])
        v, wz = np.zeros(2), 0.0
        if self.skill:
            self.skill_left -= DT
            if self.skill_left <= 0:
                self.skill = None
        elif not self.fallen:
            vx, vy, wz = (float(c) for c in act.cmd)
            if self.lazy:
                vx, vy, wz, self.walking = lazy_legs(act.cmd, self.walking)
            self.yaw += wz * DT
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            v = np.array([c * vx - s * vy, s * vx + c * vy])
            self.duck += DT * v
        self.ball, self.ball_v = push_ball(self.duck, self.yaw, v, wz, self.ball, self.ball_v)
        sp = np.linalg.norm(self.ball_v)
        if sp > 0:
            self.ball_v *= max(0.0, sp - BALL_DECEL * DT) / sp
        self.ball += self.ball_v * DT
        self.t += DT

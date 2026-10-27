"""Chase's brains: a hand-written ball chaser, and a learned one.

    CHASE     steer at the ball
    RECOVER   fell over: zero command while the walking policy gets up

`ChaseBrain` steers with common/steer.py (~20 lines of geometry).
`LearnedBrain` asks a small network trained by chase/train.py instead; what
it sees (`observe`) and how its output becomes a command (`to_command`, in
common/brain.py) are defined once, so training and playing can't disagree.

Reaching the ball, respawning it and giving up are game rules, not the
brain's business (chase/game.py).
"""

from __future__ import annotations

import math

import numpy as np

from common.brain import DECIDE_EVERY, MAX_FORWARD, MAX_TURN, Action, Brain, Network, Snapshot, to_command
from common.steer import SteerParams, steer_to, to_duck_frame

OBS_SIZE = 7


def observe(s: Snapshot, last_cmd) -> np.ndarray:
    """What the learned brain sees: the ball in the duck's frame (x ahead, y left,
    distance, bearing as sin / cos) and the command it sent last, all roughly in [-1, 1]."""
    x, y = to_duck_frame(s.duck_xy, s.duck_yaw, s.ball_xy)
    dist, bearing = math.hypot(x, y), math.atan2(y, x)
    return np.array([x / 2, y / 2, min(dist, 4.0) / 2, math.sin(bearing), math.cos(bearing),
                     last_cmd[0] / MAX_FORWARD, last_cmd[2] / MAX_TURN], dtype=np.float32)


class ChaseBrain(Brain):
    def __init__(self, params: SteerParams = SteerParams()):
        super().__init__("CHASE")
        self.p = params

    def update(self, s: Snapshot) -> Action:
        if s.fallen:
            if self.state != "RECOVER":
                self.set_state("RECOVER", s.t, "fell over")
            self.plan = []
            return Action()
        if self.state == "RECOVER":
            self.set_state("CHASE", s.t, "back up")
        self.plan = [np.array(s.duck_xy, float), np.array(s.ball_xy, float)]
        return Action(cmd=self.command(s))

    def command(self, s: Snapshot) -> np.ndarray:
        cmd, _, _ = steer_to(to_duck_frame(s.duck_xy, s.duck_yaw, s.ball_xy), self.p)
        return cmd


class LearnedBrain(ChaseBrain):
    """Chases with a network exported by chase/train.py (an ONNX file)."""

    def __init__(self, onnx_path: str):
        super().__init__()
        self.p = {"onnx": str(onnx_path)}
        self.net = Network(onnx_path)
        self.cmd = np.zeros(3, dtype=np.float32)
        self.ticks = 0

    def command(self, s: Snapshot) -> np.ndarray:
        if self.ticks % DECIDE_EVERY == 0:
            self.cmd = to_command(self.net(observe(s, self.cmd)))
        self.ticks += 1
        return self.cmd

"""Chase's brain: a hand-written ball chaser.

    CHASE     steer straight at the ball (common/steer.py: ~20 lines of geometry)
    RECOVER   fell over: zero command while the walking policy gets up

Reaching the ball, respawning it and giving up are game rules, not the
brain's business (chase/game.py). A trained brain can replace this one: any
object with the common/brain.py interface fits.
"""

from __future__ import annotations

import numpy as np

from common.brain import Action, Brain, Snapshot
from common.steer import SteerParams, steer_to, to_duck_frame


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
        cmd, _, _ = steer_to(to_duck_frame(s.duck_xy, s.duck_yaw, s.ball_xy), self.p)
        return Action(cmd=cmd)

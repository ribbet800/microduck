"""What every brain sees, what it says, and the little bit every brain shares.

A brain is the part we write: each 50 Hz step it gets a `Snapshot` of the
world and returns an `Action`. Pollen's policies are the legs and the boot;
the brain only decides which command to send them. No MuJoCo in here, so
brains can be tested against an imaginary duck (see tests/).

Any object with these works as a brain (subclassing `Brain` gives you the
last three for free):
    update(snapshot) -> Action
    state            name of what it's doing now, e.g. "APPROACH"
    last_change      why it last changed state (shown in the viewer)
    events           dict of event lists the brain records (saved to events.json)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Snapshot:
    t: float
    duck_xy: np.ndarray
    duck_yaw: float
    ball_xy: np.ndarray
    ball_speed: float
    fallen: bool
    kicking: bool


@dataclass
class Action:
    cmd: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))  # [vx, vy, wz]
    kick: str | None = None              # "left" / "right": start that kick policy now
    stop_kick: bool = False
    place_ball: tuple | None = None      # teleport the ball here (kick assist only)


class Brain:
    """A named state, when it was entered, and why."""

    def __init__(self, state: str):
        self.state = state
        self.state_t0 = 0.0
        self.last_change = ""
        self.events: dict[str, list] = {}

    def set_state(self, state, t, why=""):
        """Switch state (or restart the current one, if there's a reason to say why)."""
        if state != self.state or why:
            change = f"{self.state} -> {state}" if state != self.state else f"{state} again"
            self.last_change = change + (f"  ({why})" if why else "")
            print(f"[{t:6.1f}s] {self.last_change}")
        self.state, self.state_t0 = state, t

    def update(self, s: Snapshot) -> Action:
        raise NotImplementedError

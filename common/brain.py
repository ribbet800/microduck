"""What every brain sees, what it says, and the little bit every brain shares.

A brain is the part we write: each 50 Hz step it gets a `Snapshot` of the
world and returns an `Action`. Pollen's policies are the legs (walking) and the
skills (e.g. a kick); the brain only decides which command to send and when to
start a skill. No MuJoCo in here, so brains can be tested, and later trained,
against the toy world in common/toy.py.

Any object with these works as a brain, hand-written or a trained network
(subclassing `Brain` gives you the last four for free):
    update(snapshot) -> Action
    state            name of what it's doing now, e.g. "APPROACH"
    last_change      why it last changed state (shown in the viewer)
    events           dict of event lists the brain records (saved to events.json)
    plan             the route it means to walk now: world (x, y) points starting
                     at the duck, or [] (drawn in the viewer)
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
    skill: str | None = None             # the skill running now (e.g. "kick_left"), None while walking


@dataclass
class Action:
    cmd: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))  # [vx, vy, wz] for walking
    skill: str | None = None             # start this skill now (common/duck.py)
    stop_skill: bool = False             # cut a running skill short (e.g. after a fall)


class Brain:
    """A named state, when it was entered, and why."""

    quiet = False                        # True: don't print state changes (training runs millions)

    def __init__(self, state: str):
        self.state = state
        self.state_t0 = 0.0
        self.last_change = ""
        self.events: dict[str, list] = {}
        self.plan: list = []

    def set_state(self, state, t, why=""):
        """Switch state (or restart the current one, if there's a reason to say why)."""
        if state != self.state or why:
            change = f"{self.state} -> {state}" if state != self.state else f"{state} again"
            self.last_change = change + (f"  ({why})" if why else "")
            if not self.quiet:
                print(f"[{t:6.1f}s] {self.last_change}")
        self.state, self.state_t0 = state, t

    def update(self, s: Snapshot) -> Action:
        raise NotImplementedError


# -- learned brains -----------------------------------------------------------------
# A learned brain (chase/train.py, kickback/train.py) decides 10 times a second and
# holds its command in between (5 x 50 Hz ticks), with commands inside the walking
# policy's trained ranges.
DECIDE_EVERY = 5
MAX_FORWARD, MAX_TURN = 0.4, 1.0


def to_command(a) -> np.ndarray:
    """A network's two outputs (each in [-1, 1]) -> [vx, vy, wz]: forward 0 to
    MAX_FORWARD (the duck can't walk backwards or sidestep anyway), turn ±MAX_TURN."""
    a = np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0)
    return np.array([MAX_FORWARD * (a[0] + 1) / 2, 0.0, MAX_TURN * a[1]], dtype=np.float32)


class Network:
    """A trained network exported to ONNX: observation in, the two outputs out."""

    def __init__(self, onnx_path):
        import onnxruntime as ort   # here, so the hand-written brains and the tests don't need it
        self.path = str(onnx_path)
        self.session = ort.InferenceSession(self.path)
        self.input = self.session.get_inputs()[0].name

    def __call__(self, obs) -> np.ndarray:
        return self.session.run(None, {self.input: np.asarray(obs, dtype=np.float32)[None]})[0][0]

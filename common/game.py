"""The loop every game shares: world -> Snapshot -> brain -> Action -> Pollen's policy.

A new game is a subclass of `Game` with a brain (common/brain.py) plus:
    start()          put the duck and ball where they begin
    rules(s)         optional: game rules each step (e.g. "reached the ball: new ball")
    draw(scn)        optional: extra 3D markers (call super().draw(scn) first)
    status()         optional: extra rows for the viewer panel (super().status() + yours)
    log_row()        optional: extra log.csv columns ({**super().log_row(), ...})
    summary()        dict saved as summary.json ({**super().summary(), ...})
    plots(out_dir)   write PNGs, return their names

This class does the plumbing: noticing falls (with hysteresis), building the
Snapshot, carrying out the Action on the duck, stepping, logging every step,
trails, and the bits of drawing / status / summary every game has.
`sim.run_live` / `sim.run_headless` drive it.
"""

from __future__ import annotations

import numpy as np

from common import overlay, sim
from common.brain import Snapshot


class Game:
    camera = {"track": True, "distance": 2.4, "elevation": -45, "azimuth": 135}
    state_colours: dict = {}     # command arrow colour per brain state (default blue)
    trail_len = 200              # dots kept in each trail (one every 0.1 s)

    def __init__(self, duck: sim.Duck, brain):
        self.duck, self.brain = duck, brain
        self.t = 0.0
        self.fallen = False
        self.cmd = np.zeros(3, dtype=np.float32)
        self.duck_trail: list[tuple[float, float]] = []
        self.ball_trail: list[tuple[float, float]] = []
        # The brain's own event lists are shared, not copied, so they fill up here too.
        self.events = {"falls": [], **brain.events}
        self.log: dict[str, list] = {}
        self._vf = self._wz = 0.0

    # -- to fill in --------------------------------------------------------------
    def start(self):
        raise NotImplementedError

    def rules(self, s: Snapshot):
        pass

    def plots(self, out_dir):
        return []

    # -- the loop ----------------------------------------------------------------
    def snapshot(self) -> Snapshot:
        d = self.duck
        return Snapshot(self.t, d.xy, d.yaw, d.ball[:2], d.ball_speed, self.fallen, d.kicking)

    def _check_fallen(self):
        """On the floor below FALLEN_Z, back up above RECOVERED_Z (the gap stops flicker)."""
        d = self.duck
        if not self.fallen and d.z < sim.FALLEN_Z:
            self.fallen = True
            x, y = d.xy
            self.events["falls"].append({"t": self.t, "x": float(x), "y": float(y), "during": self.brain.state})
        elif self.fallen and d.z > sim.RECOVERED_Z:
            self.fallen = False

    def apply(self, act):
        """Do what the brain asked, apart from the walking command."""
        d = self.duck
        if act.stop_kick:
            d.stop_kick()
        if act.place_ball is not None:
            d.place_ball(*act.place_ball)
        if act.kick:
            if not d.can_kick(act.kick):
                raise RuntimeError(f"the brain asked for a {act.kick}-foot kick but that policy isn't loaded")
            d.start_kick(act.kick)

    def tick(self):
        """One 50 Hz control step."""
        self._check_fallen()
        s = self.snapshot()
        act = self.brain.update(s)
        self.rules(s)
        self.apply(act)
        self.cmd = act.cmd
        self.duck.step(act.cmd)
        self.t += sim.CONTROL_DT

        # Smooth achieved velocities over ~0.5 s so plots show the trend,
        # not the step-by-step waddle.
        vf, wz = self.duck.speeds()
        a = sim.CONTROL_DT / 0.5
        self._vf += a * (vf - self._vf)
        self._wz += a * (wz - self._wz)

        for k, v in self.log_row().items():
            self.log.setdefault(k, []).append(v)
        if int(round(self.t / sim.CONTROL_DT)) % 5 == 0:
            x, y = self.duck.xy
            bx, by, _ = self.duck.ball
            self.duck_trail = (self.duck_trail + [(float(x), float(y))])[-self.trail_len:]
            self.ball_trail = (self.ball_trail + [(float(bx), float(by))])[-self.trail_len:]

    # -- what every game shows and saves ----------------------------------------
    def log_row(self) -> dict:
        d = self.duck
        x, y = d.xy
        bx, by, _ = d.ball
        return {"t": self.t, "state": self.brain.state,
                "duck_x": float(x), "duck_y": float(y), "duck_z": d.z, "duck_yaw": d.yaw,
                "ball_x": float(bx), "ball_y": float(by), "ball_speed": d.ball_speed,
                "fallen": int(self.fallen),
                "cmd_vx": float(self.cmd[0]), "cmd_vy": float(self.cmd[1]), "cmd_wz": float(self.cmd[2]),
                "vel_fwd": self._vf, "yaw_rate": self._wz}

    def draw(self, scn):
        d = self.duck
        overlay.command_arrow(scn, d.xy, d.yaw, self.cmd, self.state_colours.get(self.brain.state, overlay.BLUE))
        overlay.trail(scn, self.duck_trail)

    def status(self):
        return [("time", f"{self.t:.1f} s"),
                ("state", self.brain.state),
                ("why", self.brain.last_change or "-"),
                ("falls", str(len(self.events["falls"])))]

    def summary(self):
        return {"sim_seconds": round(self.t, 2), "falls": len(self.events["falls"])}

#!/usr/bin/env python3
"""Fetch, Stage 0: a duck that chases a ball. No training needed.

Pollen's trained walking policy ("the legs") runs exactly as it does in their
`infer_policy.py` rehearsal script — same BAM servo model, same 61-D
observation, same 50 Hz control loop. The only change is who sends the
velocity command: instead of your keyboard, `chaser.chase_command` looks at
where the ball is and steers towards it. When the duck reaches the ball, the
ball jumps somewhere new and the chase starts again.

  Live (watch it in the MuJoCo viewer):
      uv run fetch/chase_ball.py                 # Windows / Linux
      uv run mjpython fetch/chase_ball.py        # macOS needs mjpython for the viewer

  Headless (fast, saves plots + optional video to runs/chase_<time>/):
      uv run fetch/chase_ball.py --headless --duration 90 --video

In the live viewer you can mess with the ball: double-click it to select,
then Ctrl + right-drag to shove it around.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from common import sim  # noqa: E402

sim.setup_gl_for_headless()   # must run before mujoco is imported

from common import overlay  # noqa: E402
from chaser import ChaserParams, ball_in_duck_frame, chase_command  # noqa: E402
import plots  # noqa: E402


class Chase:
    """The game rules: chase, count catches, respawn the ball, cope with falls."""

    camera = {"track": True, "distance": 2.4, "elevation": -45, "azimuth": 135}

    def __init__(self, duck: sim.Duck, params: ChaserParams, give_up_s: float):
        self.duck, self.params, self.give_up_s = duck, params, give_up_s
        self.t = 0.0
        self.fallen = False
        self.ball_t0 = 0.0
        self.cmd = np.zeros(3, dtype=np.float32)
        self.trail: list[tuple[float, float]] = []
        self.events = {"spawns": [], "catches": [], "falls": [], "timeouts": []}
        self.log = {k: [] for k in ("t", "duck_x", "duck_y", "duck_z", "duck_yaw", "ball_x", "ball_y",
                                    "dist", "bearing", "cmd_vx", "cmd_wz", "vel_fwd", "yaw_rate",
                                    "fallen")}
        self._vf = self._wz = 0.0

    def start(self):
        self.duck.reset()
        self._new_ball()

    def _new_ball(self):
        self.events["spawns"].append(self.duck.place_ball_near_duck())
        self.ball_t0 = self.t

    def tick(self):
        d, p = self.duck, self.params
        rel = ball_in_duck_frame(d.xy, d.yaw, d.ball[:2])
        cmd, dist, bearing = chase_command(rel, p)

        # Fallen over? Send zero command: the velstand policy gets itself up.
        if not self.fallen and d.z < sim.FALLEN_Z:
            self.fallen = True
            x, y = d.xy
            self.events["falls"].append({"t": self.t, "x": float(x), "y": float(y)})
            print(f"[{self.t:6.1f}s] fell over — getting back up")
        elif self.fallen and d.z > sim.RECOVERED_Z:
            self.fallen = False
            print(f"[{self.t:6.1f}s] back on its feet")
        if self.fallen:
            cmd = np.zeros(3, dtype=np.float32)

        if dist < p.catch_radius and not self.fallen:
            x, y = d.xy
            took = self.t - self.ball_t0
            self.events["catches"].append({"t": self.t, "took_s": took, "duck_x": float(x), "duck_y": float(y)})
            print(f"[{self.t:6.1f}s] reached ball #{len(self.events['catches'])} in {took:.1f}s")
            self._new_ball()
        elif self.t - self.ball_t0 > self.give_up_s:
            self.events["timeouts"].append({"t": self.t})
            print(f"[{self.t:6.1f}s] gave up on that ball after {self.give_up_s:.0f}s — new ball")
            self._new_ball()

        self.cmd = cmd
        d.step(cmd)
        self.t += sim.CONTROL_DT

        # Smooth achieved velocities over ~0.5 s so the plot shows the trend,
        # not the step-by-step waddle.
        vf, wz = d.speeds()
        a = sim.CONTROL_DT / 0.5
        self._vf += a * (vf - self._vf)
        self._wz += a * (wz - self._wz)

        x, y = d.xy
        b = d.ball
        for k, v in (("t", self.t), ("duck_x", x), ("duck_y", y), ("duck_z", d.z), ("duck_yaw", d.yaw),
                     ("ball_x", b[0]), ("ball_y", b[1]), ("dist", dist), ("bearing", bearing),
                     ("cmd_vx", cmd[0]), ("cmd_wz", cmd[2]), ("vel_fwd", self._vf),
                     ("yaw_rate", self._wz), ("fallen", int(self.fallen))):
            self.log[k].append(float(v))
        if int(round(self.t / sim.CONTROL_DT)) % 5 == 0:
            self.trail = (self.trail + [(float(x), float(y))])[-200:]

    def draw(self, scn):
        d = self.duck
        overlay.line(scn, d.xy, d.ball[:2])
        overlay.disc(scn, d.ball[:2], self.params.catch_radius)
        overlay.command_arrow(scn, d.xy, d.yaw, self.cmd)
        overlay.trail(scn, self.trail)

    def status(self):
        dist = self.log["dist"][-1] if self.log["dist"] else float("nan")
        return [("time", f"{self.t:.1f} s"),
                ("balls reached", str(len(self.events["catches"]))),
                ("falls", str(len(self.events["falls"]))),
                ("distance", f"{dist:.2f} m"),
                ("command", f"fwd {self.cmd[0]:+.2f} m/s  turn {self.cmd[2]:+.2f} rad/s")]

    def summary(self):
        took = [c["took_s"] for c in self.events["catches"]]
        return {"sim_seconds": round(self.t, 2),
                "balls_reached": len(took),
                "mean_time_to_reach_s": round(float(np.mean(took)), 2) if took else None,
                "falls": len(self.events["falls"]),
                "gave_up": len(self.events["timeouts"]),
                "params": vars(self.params)}

    def plots(self, out_dir):
        r = self.params.catch_radius
        plots.topdown(self.log, self.events, str(out_dir / "topdown.png"), r)
        plots.timeseries(self.log, self.events, str(out_dir / "timeseries.png"), r)
        return ["topdown.png", "timeseries.png"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sim.add_common_args(ap)
    ap.add_argument("--give-up", type=float, default=25.0, help="seconds before a ball is respawned")
    ap.add_argument("--max-forward", type=float, default=ChaserParams.max_forward)
    ap.add_argument("--min-forward", type=float, default=ChaserParams.min_forward,
                    help="smallest forward command sent (the walking policy ignores less)")
    ap.add_argument("--max-turn", type=float, default=ChaserParams.max_turn)
    ap.add_argument("--turn-gain", type=float, default=ChaserParams.turn_gain)
    ap.add_argument("--catch-radius", type=float, default=ChaserParams.catch_radius)
    args = ap.parse_args()

    rl_repo = sim.find_rl_repo(args.rl_repo)
    ip = sim.import_infer_policy(rl_repo)
    walk = sim.walking_policy(args.policy, args.policy_revision)
    model, data, bam_ctrl, policy = sim.build_sim(ip, rl_repo, walk, use_bam=not args.no_bam)

    params = ChaserParams(max_forward=args.max_forward, min_forward=args.min_forward, max_turn=args.max_turn,
                          turn_gain=args.turn_gain, catch_radius=args.catch_radius)
    duck = sim.Duck(model, data, bam_ctrl, policy, np.random.default_rng(args.seed))
    chase = Chase(duck, params, give_up_s=args.give_up)
    chase.start()

    print("\nChasing! " + ("(headless)" if args.headless else "Close the viewer window to stop."))
    if args.headless:
        sim.run_headless(duck, chase, args.duration, sim.default_out("chase", args.out), args.video)
    else:
        sim.run_live(duck, chase)


if __name__ == "__main__":
    main()

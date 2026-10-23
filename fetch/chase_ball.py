#!/usr/bin/env python3
"""Fetch, Stage 0: a duck that chases a ball. No training needed.

Pollen's trained walking policy ("the legs") runs exactly as it does in their
`infer_policy.py` rehearsal script — same BAM servo model, same 61-D
observation, same 50 Hz control loop. The only change is who sends the
velocity command: instead of your keyboard, the brain (`chaser.ChaseBrain`)
looks at where the ball is and steers towards it. When the duck reaches the
ball, the ball jumps somewhere new and the chase starts again.

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
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from common import sim  # noqa: E402

sim.setup_gl_for_headless()   # must run before mujoco is imported

from common import overlay  # noqa: E402
from common.brain import Snapshot  # noqa: E402
from common.game import Game  # noqa: E402
from common.steer import SteerParams, to_duck_frame  # noqa: E402
from chaser import ChaseBrain  # noqa: E402
import plots  # noqa: E402


class Chase(Game):
    """The game rules: count catches, respawn the ball, give up on a ball after a while.
    The brain only steers; any brain with the common/brain.py interface fits."""

    def __init__(self, duck: sim.Duck, brain, catch_radius: float, give_up_s: float,
                 rng: np.random.Generator, spawn_r=(0.6, 1.4)):
        super().__init__(duck, brain)
        self.catch_radius, self.give_up_s, self.rng, self.spawn_r = catch_radius, give_up_s, rng, spawn_r
        self.ball_t0 = 0.0
        self.dist = self.bearing = float("nan")
        self.events.update({"spawns": [], "catches": [], "timeouts": []})

    def start(self):
        self.duck.reset()
        self._new_ball()

    def _new_ball(self):
        """Drop the ball at a random spot 0.6-1.4 m from the duck."""
        x, y = self.duck.xy
        r, a = self.rng.uniform(*self.spawn_r), self.rng.uniform(-math.pi, math.pi)
        bx, by = float(x + r * math.cos(a)), float(y + r * math.sin(a))
        self.duck.place_ball(bx, by)
        self.events["spawns"].append((bx, by))
        self.ball_t0 = self.t

    def rules(self, s: Snapshot):
        rel = to_duck_frame(s.duck_xy, s.duck_yaw, s.ball_xy)
        self.dist, self.bearing = float(np.hypot(*rel)), float(math.atan2(rel[1], rel[0]))
        if self.dist < self.catch_radius and not s.fallen:
            took = s.t - self.ball_t0
            self.events["catches"].append({"t": s.t, "took_s": took,
                                           "duck_x": float(s.duck_xy[0]), "duck_y": float(s.duck_xy[1])})
            print(f"[{s.t:6.1f}s] reached ball #{len(self.events['catches'])} in {took:.1f}s")
            self._new_ball()
        elif s.t - self.ball_t0 > self.give_up_s:
            self.events["timeouts"].append({"t": s.t})
            print(f"[{s.t:6.1f}s] gave up on that ball after {self.give_up_s:.0f}s — new ball")
            self._new_ball()

    def log_row(self):
        return {**super().log_row(), "dist": self.dist, "bearing": self.bearing}

    legend = Game.legend + [("green", "catch radius: inside it counts as reached")]

    def draw(self, scn):
        super().draw(scn)
        overlay.disc(scn, self.duck.ball[:2], self.catch_radius, (*overlay.GREEN[:3], 0.3))

    def status(self):
        return super().status() + [
            ("balls reached", str(len(self.events["catches"]))),
            ("distance", f"{self.dist:.2f} m"),
            ("command", f"fwd {self.cmd[0]:+.2f} m/s  turn {self.cmd[2]:+.2f} rad/s")]

    def summary(self):
        took = [c["took_s"] for c in self.events["catches"]]
        return {**super().summary(),
                "balls_reached": len(took),
                "mean_time_to_reach_s": round(float(np.mean(took)), 2) if took else None,
                "gave_up": len(self.events["timeouts"]),
                "params": vars(self.brain.p) if hasattr(self.brain, "p") else {}}

    def plots(self, out_dir):
        r = self.catch_radius
        plots.topdown(self.log, self.events, str(out_dir / "topdown.png"), r)
        plots.timeseries(self.log, self.events, str(out_dir / "timeseries.png"), r)
        return ["topdown.png", "timeseries.png"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sim.add_common_args(ap)
    ap.add_argument("--give-up", type=float, default=25.0, help="seconds before a ball is respawned")
    ap.add_argument("--max-forward", type=float, default=SteerParams.max_forward)
    ap.add_argument("--min-forward", type=float, default=SteerParams.min_forward,
                    help="smallest forward command sent (the walking policy ignores less)")
    ap.add_argument("--max-turn", type=float, default=SteerParams.max_turn)
    ap.add_argument("--turn-gain", type=float, default=SteerParams.turn_gain)
    ap.add_argument("--catch-radius", type=float, default=SteerParams.catch_radius)
    args = ap.parse_args()

    duck = sim.make_duck(args)
    brain = ChaseBrain(SteerParams(max_forward=args.max_forward, min_forward=args.min_forward,
                                   max_turn=args.max_turn, turn_gain=args.turn_gain,
                                   catch_radius=args.catch_radius))
    game = Chase(duck, brain, catch_radius=args.catch_radius, give_up_s=args.give_up,
                 rng=np.random.default_rng(args.seed))
    sim.run(duck, game, args, "chase")


if __name__ == "__main__":
    main()

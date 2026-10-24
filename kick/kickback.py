#!/usr/bin/env python3
"""Kickback: the duck kicks the ball back and forth along a straight line.

The kick loop with one change in the brain (kick/brain.py: KickbackBrain): every
kick goes back along the line the way the last one came, instead of in a random
direction. So the duck kicks the ball, follows it, orbits round it, and kicks it back.

  Live:      uv run kick/kickback.py              (macOS: uv run mjpython kick/kickback.py)
  Headless:  uv run kick/kickback.py --headless --duration 180 --video
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so `uv run kick/kickback.py` finds the packages

from common import overlay, sim  # noqa: E402
from kick.brain import KickbackBrain, KickParams  # noqa: E402
from kick.kick_loop import KickLoop, add_kick_args, aim_arrow, make_kick_duck  # noqa: E402


class Kickback(KickLoop):
    """KickLoop with a line instead of the arena, and a "how straight" score."""

    area_legend = ("faint line", "the line it kicks the ball along")

    def __init__(self, duck, brain: KickbackBrain, follow: bool, ball_roll: float = 1.0):
        super().__init__(duck, brain, follow, ball_roll)
        self.line = brain.ends
        if not follow:
            self.camera = {**self.camera, "distance": 3.0}

    def draw_area(self, scn):
        a, b = self.line
        overlay.line(scn, a, b, overlay.FAINT, width=2.0, z=overlay.Z_AREA)
        for end in (a, b):
            overlay.disc(scn, end, 0.06, (*overlay.FAINT[:3], 0.25), z=overlay.Z_AREA)

    def summary(self):
        # How far from the line each kick left the ball (0 = dead straight back and forth).
        a, b = self.line
        u = (b - a) / np.linalg.norm(b - a)
        off = []
        for k in self.events["kicks"]:
            if "end_xy" in k:
                v = np.asarray(k["end_xy"]) - a
                off.append(abs(float(u[0] * v[1] - u[1] * v[0])))
        return {**super().summary(),
                "mean_off_line_m": round(float(np.mean(off)), 3) if off else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sim.add_common_args(ap)
    add_kick_args(ap)
    ap.add_argument("--length", type=float, default=1.0,
                    help="length of the line (m); about one kick's roll (--ball-roll) works best")
    args = ap.parse_args()

    duck, feet = make_kick_duck(args)
    half = args.length / 2
    brain = KickbackBrain(KickParams(assist=args.assist, tolerance=args.tolerance), feet,
                          ends=[(-half, 0.0), (half, 0.0)])
    sim.run(duck, Kickback(duck, brain, args.follow, aim_arrow(args)), args, "kickback")


if __name__ == "__main__":
    main()

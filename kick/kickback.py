#!/usr/bin/env python3
"""Kickback: the duck kicks the ball back and forth along a straight line.

The kick loop's brain with one change (kick/brain.py: KickbackBrain): every kick
aims at the far end of the line instead of a random direction. So the duck kicks
the ball to one end, follows it, orbits round it, and kicks it back.

  Live:      uv run kick/kickback.py              (macOS: uv run mjpython kick/kickback.py)
  Headless:  uv run kick/kickback.py --headless --duration 180 --video
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
from brain import KickbackBrain, KickParams  # noqa: E402
from kick_loop import KickLoop, add_kick_args, make_kick_duck  # noqa: E402


class Kickback(KickLoop):
    """KickLoop with a line instead of the arena, and a "how straight" score."""

    def __init__(self, duck, brain: KickbackBrain, follow: bool):
        super().__init__(duck, brain, follow)
        if not follow:
            self.camera = {**self.camera, "distance": 3.0}

    legend = KickLoop.legend[:-1] + [("faint line", "the line it kicks the ball along")]

    def draw_area(self, scn):
        a, b = self.brain.ends
        overlay.line(scn, a, b, overlay.FAINT, width=2.0, z=overlay.Z_AREA)
        for end in (a, b):
            overlay.disc(scn, end, 0.06, (*overlay.FAINT[:3], 0.25), z=overlay.Z_AREA)

    def summary(self):
        # How far from the line each kick left the ball (0 = dead straight back and forth).
        a, b = self.brain.ends
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
    sim.run(duck, Kickback(duck, brain, follow=args.follow), args, "kickback")


if __name__ == "__main__":
    main()

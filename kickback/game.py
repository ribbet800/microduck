#!/usr/bin/env python3
"""Kickback: the duck kicks the ball back and forth along a straight line.

Pollen's walking policy is the legs and Pollen's kick policies are skills. The
brain (kickback/brain.py, which explains how) gets behind the ball, lines up
until the ball is exactly in front of a foot, starts the kick, watches where
the ball goes, then goes round it and kicks it back.

  Live:      uv run kickback/game.py            (macOS: uv run mjpython kickback/game.py)
  Headless:  uv run kickback/game.py --headless --duration 180 --video
  Learned:   uv run kickback/game.py --brain runs/kickback_learned/brain.onnx   (kickback/train.py)
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so `uv run kickback/game.py` finds the packages

from common import overlay, run  # noqa: E402
from common.duck import Duck, hub_policy  # noqa: E402
from common.game import Game  # noqa: E402
from kickback import plots  # noqa: E402
from kickback.brain import WHIFF_SPEED, KickbackBrain, LearnedKickbackBrain  # noqa: E402

KICK_SECONDS = 3.0   # how long a kick skill runs before handing back to walking (Pollen's default)


class Kickback(Game):
    """The brain does nearly everything here; the game draws it and keeps score."""

    name = "kickback"
    trail_len = 150
    camera = {"track": False, "lookat": [0.0, 0.0, 0.0], "distance": 3.0, "elevation": -60, "azimuth": 90}
    legend = Game.legend + [("yellow", "where the next kick should send the ball, ball trail"),
                            ("red / green", "kick spot in front of the foot: green = ball on it"),
                            ("faint line", "the line it kicks the ball along")]

    def __init__(self, duck: Duck, brain: KickbackBrain, aim_arrow: float = 1.0):
        super().__init__(duck, brain)
        self.aim_arrow = aim_arrow       # how long to draw the aim arrow (m): about one kick's roll

    def start(self):
        self.duck.reset(xy=(-0.45, 0.0), yaw=0.0)
        self.duck.place_ball(0.0, 0.0)

    def log_row(self):
        b = self.brain
        lined_up = b.state in ("ALIGN", "SETTLE")
        return {**super().log_row(), "aim_deg": b.aim_deg,
                "align_err": b.align_err if lined_up else float("nan"),
                "heading_err": b.heading_err if lined_up else float("nan")}

    # -- drawing ------------------------------------------------------------------
    def draw(self, scn):
        super().draw(scn)
        d, b, p = self.duck, self.brain, self.brain.p
        a, e = b.ends
        overlay.line(scn, a, e, overlay.FAINT, width=2.0, z=overlay.Z_AREA)
        for end in (a, e):
            overlay.disc(scn, end, 0.06, (*overlay.FAINT[:3], 0.25), z=overlay.Z_AREA)
        overlay.trail(scn, self.ball_trail, overlay.YELLOW)
        # Where the next kick should send the ball, from where it was aimed (so
        # after a kick the ball's trail can be compared with it).
        if b.aim is not None:
            start = b.aim_from + 0.04 * b.aim                        # from the ball's edge
            overlay.arrow(scn, [start, b.aim_from + self.aim_arrow * b.aim], overlay.YELLOW,
                          width=4.0, z=overlay.Z_AIM)
        # The kick spot in front of the duck's foot: green when the ball is on it.
        spot = b.spot_world(d.xy, d.yaw)
        on_spot = np.linalg.norm(spot - d.ball[:2]) < p.tolerance
        overlay.disc(scn, spot, p.tolerance, overlay.GREEN if on_spot else overlay.RED)

    def status(self):
        b = self.brain
        kicks = b.events["kicks"]
        err = f"{b.align_err * 1000:.0f} mm, {np.degrees(b.heading_err):+.0f}°" if b.state in ("ALIGN", "SETTLE") else "-"
        done = [k for k in kicks if k.get("aim_err_deg") is not None]
        last = (f"aimed {done[-1]['aim_deg']:+.0f}°, went {done[-1]['actual_deg']:+.0f}°, "
                f"{done[-1]['travel_m']:.2f} m") if done else "-"
        return super().status() + [
            ("foot", b.foot),
            ("next aim", f"{b.aim_deg:+.0f}°" if b.aim is not None else "-"),
            ("kicks", str(len(kicks))),
            ("last kick", last),
            ("line-up error", err)]

    # -- outputs ------------------------------------------------------------------
    def summary(self):
        b = self.brain
        kicks = b.events["kicks"]
        done = [k for k in kicks if "peak_ball_speed" in k]
        speeds = [k["peak_ball_speed"] for k in done]
        aim_errs = [abs(k["aim_err_deg"]) for k in done if k.get("aim_err_deg") is not None]
        # How far from the line each kick left the ball (0 = dead straight back and forth).
        a, e = b.ends
        u = (e - a) / np.linalg.norm(e - a)
        off = [abs(float(u[0] * (k["end_xy"][1] - a[1]) - u[1] * (k["end_xy"][0] - a[0]))) for k in done]
        return {**super().summary(),
                "kicks": len(kicks),
                "kicks_per_min": round(60 * len(kicks) / max(self.t, 1e-6), 2),
                "whiffs": sum(1 for s in speeds if s < WHIFF_SPEED),
                "mean_ball_speed_m_s": round(float(np.mean(speeds)), 2) if speeds else None,
                "mean_travel_m": round(float(np.mean([k["travel_m"] for k in done])), 2) if done else None,
                "mean_abs_aim_error_deg": round(float(np.mean(aim_errs)), 1) if aim_errs else None,
                "mean_lineup_error_mm": round(1000 * float(np.mean([k["err_m"] for k in kicks])), 1) if kicks else None,
                "mean_off_line_m": round(float(np.mean(off)), 3) if off else None,
                "line_up_retries": len(b.events["retries"]),
                "knocks": len(b.events["knocks"]),
                "ball_crept": len(b.events["crept"]),
                "brain": type(b).__name__ + (f" ({b.net.path})" if hasattr(b, "net") else ""),
                "timeouts": len(b.events["timeouts"]),
                "params": dataclasses.asdict(b.p)}

    def plots(self, out_dir):
        b = self.brain
        plots.topdown(self.log, self.events, b.ends, str(out_dir / "topdown.png"))
        plots.alignment(self.events, b.p, str(out_dir / "alignment.png"))
        plots.aiming(self.events, str(out_dir / "aiming.png"))
        plots.timeline(self.log, self.events, b.p, str(out_dir / "timeline.png"))
        return ["topdown.png", "alignment.png", "aiming.png", "timeline.png"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    run.add_args(ap)
    ap.add_argument("--length", type=float, default=1.0,
                    help="length of the line (m); about one kick's roll (--ball-roll) works best")
    for foot in ("left", "right"):
        ap.add_argument(f"--kick-{foot}", type=str, default=None,
                        help=f"{foot}-foot kick policy ONNX (default: ball_kick_{foot}.onnx from the Hub)")
    ap.add_argument("--brain", type=str, default=None,
                    help="a learned line-up (ONNX from kickback/train.py) instead of the hand-written one")
    args = ap.parse_args()

    train_hint = ("or train one in microduck_rl: `uv run train Mjlab-BallKick-Flat-MicroDuck` "
                  "(KICK_FOOT at the top of microduck_ball_kick_env_cfg.py picks the foot), "
                  "then export it, and pass --kick-left / --kick-right")
    skills = {f"kick_{foot}": (getattr(args, f"kick_{foot}")
                               or hub_policy(f"ball_kick_{foot}.onnx", args.policy_revision, train_hint), KICK_SECONDS)
              for foot in ("left", "right")}
    duck = run.duck_from_args(args, skills)
    half = args.length / 2
    ends = [(-half, 0.0), (half, 0.0)]
    brain = LearnedKickbackBrain(ends, args.brain) if args.brain else KickbackBrain(ends)
    run.play(Kickback(duck, brain, aim_arrow=args.ball_roll or 2.0), args)   # Pollen's ball (0) never stops: draw 2 m


if __name__ == "__main__":
    main()

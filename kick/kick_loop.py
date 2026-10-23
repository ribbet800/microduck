#!/usr/bin/env python3
"""Kick loop: the duck walks up to the ball, kicks it, follows it, kicks it again.

Pollen's walking policy is the legs and Pollen's kick policy is the boot. The
brain (kick/brain.py) picks a random direction for each kick (roughly onwards
from where the duck is, and back towards the middle once the ball strays
outside the arena), walks round behind the ball, walks in along that line until
the ball is exactly in front of a foot, stands still, swaps in the kick policy,
watches where the ball goes, and does it all again.

  Live:      uv run kick/kick_loop.py            (macOS: uv run mjpython kick/kick_loop.py)
  Headless:  uv run kick/kick_loop.py --headless --duration 180 --video

The kick policy is blind, so lining up is the hard part. If the duck keeps
failing to line up, try --assist: when it gets close, the ball is nudged onto
the exact spot (Pollen's own demo always does this).
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
from common.game import Game  # noqa: E402
from brain import WHIFF_SPEED, KickBrain, KickParams  # noqa: E402
import kick_plots  # noqa: E402


class KickLoop(Game):
    """The brain does nearly everything here; the game draws it and keeps score."""

    trail_len = 150

    def __init__(self, duck: sim.Duck, brain: KickBrain, follow: bool):
        super().__init__(duck, brain)
        self.camera = ({"track": True, "distance": 1.8, "elevation": -40, "azimuth": 135} if follow else
                       {"track": False, "lookat": [0.0, 0.0, 0.0], "distance": 4.2, "elevation": -60, "azimuth": 90})

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
    legend = Game.legend + [("yellow", "where the next kick should send the ball (orange: home), ball trail"),
                            ("red / green", "kick spot in front of the foot: green = ball on it"),
                            ("faint circle", "soft arena: outside it, kicks aim back home")]

    def draw(self, scn):
        super().draw(scn)
        d, b, p = self.duck, self.brain, self.brain.p
        self.draw_area(scn)
        overlay.trail(scn, self.ball_trail, overlay.YELLOW)
        # Where the next kick should send the ball, from where it was aimed (so
        # after a kick the ball's trail can be compared with it), about as far
        # as a kick rolls.
        if b.aim is not None:
            start = b.aim_from + 0.04 * b.aim                        # from the ball's edge
            overlay.arrow(scn, [start, b.aim_from + self.ball_roll * b.aim],
                          overlay.ORANGE if b.aim_mode == "home" else overlay.YELLOW, width=4.0, z=overlay.Z_AIM)
        # The kick spot in front of the duck's foot: green when the ball is on it.
        spot = b.spot_world(d.xy, d.yaw)
        on_spot = np.linalg.norm(spot - d.ball[:2]) < p.tolerance
        overlay.disc(scn, spot, p.tolerance, overlay.GREEN if on_spot else overlay.RED)

    def draw_area(self, scn):
        """The soft arena: outside this circle, kicks aim back towards the middle."""
        overlay.circle(scn, (0.0, 0.0), self.brain.p.arena_radius)

    def status(self):
        b = self.brain
        kicks = b.events["kicks"]
        assisted = sum(k["assisted"] for k in kicks)
        err = f"{b.align_err * 1000:.0f} mm, {np.degrees(b.heading_err):+.0f}°" if b.state in ("ALIGN", "SETTLE") else "-"
        done = [k for k in kicks if k.get("aim_err_deg") is not None]
        last = (f"aimed {done[-1]['aim_deg']:+.0f}°, went {done[-1]['actual_deg']:+.0f}°, "
                f"{done[-1]['travel_m']:.2f} m") if done else "-"
        return super().status() + [
            ("foot", b.foot),
            ("next aim", f"{b.aim_deg:+.0f}°" + (" (home)" if b.aim_mode == "home" else "") if b.aim is not None else "-"),
            ("kicks", f"{len(kicks)}" + (f" ({assisted} assisted)" if assisted else "")),
            ("last kick", last),
            ("line-up error", err)]

    # -- outputs ------------------------------------------------------------------
    def summary(self):
        b = self.brain
        kicks = b.events["kicks"]
        done = [k for k in kicks if "peak_ball_speed" in k]
        speeds = [k["peak_ball_speed"] for k in done]
        aim_errs = [abs(k["aim_err_deg"]) for k in done if k.get("aim_err_deg") is not None]
        return {**super().summary(),
                "kicks": len(kicks),
                "kicks_per_min": round(60 * len(kicks) / max(self.t, 1e-6), 2),
                "assisted_kicks": sum(k["assisted"] for k in kicks),
                "whiffs": sum(1 for s in speeds if s < WHIFF_SPEED),
                "mean_ball_speed_m_s": round(float(np.mean(speeds)), 2) if speeds else None,
                "mean_travel_m": round(float(np.mean([k["travel_m"] for k in done])), 2) if done else None,
                "mean_abs_aim_error_deg": round(float(np.mean(aim_errs)), 1) if aim_errs else None,
                "mean_lineup_error_mm": round(1000 * float(np.mean([k["err_m"] for k in kicks])), 1) if kicks else None,
                "home_aims": sum(1 for k in kicks if k["aim_mode"] == "home"),
                "line_up_retries": len(b.events["retries"]),
                "timeouts": len(b.events["timeouts"]),
                "params": vars(b.p)}

    def plots(self, out_dir):
        p = self.brain.p
        kick_plots.topdown(self.log, self.events, p, str(out_dir / "topdown.png"))
        kick_plots.alignment(self.events, p, str(out_dir / "alignment.png"))
        kick_plots.aiming(self.events, str(out_dir / "aiming.png"))
        kick_plots.timeline(self.log, self.events, p, str(out_dir / "timeline.png"))
        return ["topdown.png", "alignment.png", "aiming.png", "timeline.png"]


def add_kick_args(ap):
    """Options every kick game shares (on top of sim.add_common_args)."""
    ap.add_argument("--kick-left", type=str, default=None, help="left-foot kick policy ONNX (default: from the Hub)")
    ap.add_argument("--kick-right", type=str, default=None, help="right-foot kick policy ONNX (default: from the Hub)")
    ap.add_argument("--feet", choices=["both", "left", "right"], default="both")
    ap.add_argument("--assist", action="store_true", help="nudge the ball onto the kick spot when close")
    ap.add_argument("--tolerance", type=float, default=KickParams.tolerance, help="line-up tolerance (m)")
    ap.add_argument("--follow", action="store_true", help="camera follows the duck instead of showing the whole area")


def make_kick_duck(args):
    """The duck with the walking policy plus the kick policies for `--feet`. Returns (duck, feet)."""
    train_hint = ("or train one in microduck_rl: `uv run train Mjlab-BallKick-Flat-MicroDuck` "
                  "(KICK_FOOT at the top of microduck_ball_kick_env_cfg.py picks the foot), "
                  "then export it, and pass --kick-right / --kick-left")
    feet = ["left", "right"] if args.feet == "both" else [args.feet]
    paths = {}
    for foot in feet:
        given = getattr(args, f"kick_{foot}")
        paths[foot] = given or sim.hub_policy(["kick", foot], f"{foot}-foot kick", args.policy_revision,
                                              exact=f"ball_kick_{foot}.onnx", how_to_train=train_hint)
    return sim.make_duck(args, kick_left=paths.get("left"), kick_right=paths.get("right")), feet


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sim.add_common_args(ap)
    add_kick_args(ap)
    ap.add_argument("--aim-spread", type=float, default=math.degrees(KickParams.aim_spread),
                    help="random aim: up to this many degrees either side of the duck->ball direction")
    ap.add_argument("--arena", type=float, default=KickParams.arena_radius,
                    help="soft arena radius (m): outside it, kicks aim back towards the middle")
    args = ap.parse_args()

    duck, feet = make_kick_duck(args)
    params = KickParams(assist=args.assist, tolerance=args.tolerance,
                        aim_spread=math.radians(args.aim_spread), arena_radius=args.arena)
    brain = KickBrain(params, feet, rng=np.random.default_rng(args.seed))
    sim.run(duck, KickLoop(duck, brain, follow=args.follow), args, "kick")


if __name__ == "__main__":
    main()

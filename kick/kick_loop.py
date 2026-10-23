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
from brain import KickBrain, KickParams, Snapshot  # noqa: E402
import kick_plots  # noqa: E402

STATE_COLOURS = {"APPROACH": overlay.BLUE, "ALIGN": overlay.YELLOW, "SETTLE": overlay.YELLOW,
                 "KICK": overlay.RED, "WATCH": overlay.WHITE, "RECOVER": overlay.RED}


class KickLoop:
    def __init__(self, duck: sim.Duck, brain: KickBrain, follow: bool):
        self.duck, self.brain = duck, brain
        self.t = 0.0
        self.cmd = np.zeros(3, dtype=np.float32)
        self.fallen = False
        self.duck_trail: list = []
        self.ball_trail: list = []
        self.camera = ({"track": True, "distance": 1.8, "elevation": -40, "azimuth": 135} if follow else
                       {"track": False, "lookat": [0.0, 0.0, 0.0], "distance": 4.2, "elevation": -60, "azimuth": 90})
        self.log = {k: [] for k in ("t", "state", "duck_x", "duck_y", "duck_z", "duck_yaw", "ball_x", "ball_y",
                                    "ball_speed", "aim_deg", "align_err", "heading_err",
                                    "cmd_vx", "cmd_vy", "cmd_wz")}

    @property
    def events(self):
        return self.brain.events

    def start(self):
        self.duck.reset(xy=(-0.45, 0.0), yaw=0.0)
        self.duck.place_ball(0.0, 0.0)

    def tick(self):
        d, b = self.duck, self.brain
        if not self.fallen and d.z < sim.FALLEN_Z:
            self.fallen = True
        elif self.fallen and d.z > sim.RECOVERED_Z:
            self.fallen = False

        snap = Snapshot(self.t, d.xy, d.yaw, d.ball[:2], d.ball_speed, self.fallen, d.kicking)
        act = b.update(snap)
        if act.stop_kick:
            d.stop_kick()
        if act.place_ball is not None:
            d.place_ball(*act.place_ball)
        if act.kick:
            if d.can_kick(act.kick):
                d.start_kick(act.kick)
            else:
                print(f"no {act.kick}-foot kick policy loaded")
        self.cmd = act.cmd
        d.step(act.cmd)
        self.t += sim.CONTROL_DT

        x, y = d.xy
        bx, by, _ = d.ball
        lined_up = b.state in ("ALIGN", "SETTLE")
        for k, v in (("t", self.t), ("state", b.state), ("duck_x", x), ("duck_y", y), ("duck_z", d.z),
                     ("duck_yaw", d.yaw), ("ball_x", bx), ("ball_y", by), ("ball_speed", d.ball_speed),
                     ("aim_deg", b.aim_deg),
                     ("align_err", b.align_err if lined_up else float("nan")),
                     ("heading_err", b.heading_err if lined_up else float("nan")),
                     ("cmd_vx", act.cmd[0]), ("cmd_vy", act.cmd[1]), ("cmd_wz", act.cmd[2])):
            self.log[k].append(v if k == "state" else float(v))
        if int(round(self.t / sim.CONTROL_DT)) % 5 == 0:
            self.duck_trail = (self.duck_trail + [(float(x), float(y))])[-150:]
            self.ball_trail = (self.ball_trail + [(float(bx), float(by))])[-150:]

    # -- drawing ------------------------------------------------------------------
    def draw(self, scn):
        d, b, p = self.duck, self.brain, self.brain.p
        # The soft arena: outside this circle, kicks aim back towards the middle.
        n = 48
        for i in range(n):
            a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
            overlay.line(scn, (p.arena_radius * math.cos(a0), p.arena_radius * math.sin(a0)),
                         (p.arena_radius * math.cos(a1), p.arena_radius * math.sin(a1)),
                         (1, 1, 1, 0.3), width=2.0, z=0.003)
        ball = d.ball[:2]
        # Where it's aiming the next kick.
        if b.aim is not None:
            colour = (1.0, 0.55, 0.0, 0.7) if b.aim_mode == "home" else (1.0, 0.85, 0.1, 0.7)
            overlay.line(scn, ball, ball + 0.6 * b.aim, colour, width=2.0)
            # Where it's walking to right now, and the kick stance (ghost marker + facing arrow).
            if b.state == "APPROACH":
                overlay.sphere(scn, (b.target[0], b.target[1], 0.01), 0.012, overlay.BLUE)
            overlay.sphere(scn, (b.stance[0], b.stance[1], 0.01), 0.01, (1, 1, 1, 0.7))
            overlay.arrow(scn, b.stance, b.stance + 0.12 * b.aim, (1, 1, 1, 0.5), width=0.005, z=0.01)
        # The kick spot in front of the duck's foot: green when the ball is on it.
        spot = b.spot_world(Snapshot(self.t, d.xy, d.yaw, ball, 0, False, False))
        on_spot = np.linalg.norm(spot - ball) < p.tolerance
        overlay.disc(scn, spot, p.tolerance, (0.2, 0.95, 0.3, 0.6) if on_spot else (1.0, 0.3, 0.2, 0.45), z=0.004)
        overlay.command_arrow(scn, d.xy, d.yaw, self.cmd, STATE_COLOURS.get(b.state, overlay.BLUE))
        overlay.trail(scn, self.duck_trail)
        overlay.trail(scn, self.ball_trail, (1.0, 0.55, 0.0, 0.7), radius=0.005)

    def status(self):
        b = self.brain
        kicks = b.events["kicks"]
        assisted = sum(k["assisted"] for k in kicks)
        err = f"{b.align_err * 1000:.0f} mm, {np.degrees(b.heading_err):+.0f}°" if b.state in ("ALIGN", "SETTLE") else "-"
        done = [k for k in kicks if k.get("aim_err_deg") is not None]
        last = (f"aimed {done[-1]['aim_deg']:+.0f}°, went {done[-1]['actual_deg']:+.0f}°, "
                f"{done[-1]['travel_m']:.2f} m") if done else "-"
        return [("time", f"{self.t:.1f} s"),
                ("state", f"{b.state} ({b.foot} foot)"),
                ("next aim", f"{b.aim_deg:+.0f}°" + (" (home)" if b.aim_mode == "home" else "") if b.aim is not None else "-"),
                ("kicks", f"{len(kicks)}" + (f" ({assisted} assisted)" if assisted else "")),
                ("last kick", last),
                ("falls", str(len(b.events["falls"]))),
                ("line-up error", err)]

    # -- outputs ------------------------------------------------------------------
    def summary(self):
        b = self.brain
        kicks = b.events["kicks"]
        done = [k for k in kicks if "peak_ball_speed" in k]
        speeds = [k["peak_ball_speed"] for k in done]
        aim_errs = [abs(k["aim_err_deg"]) for k in done if k.get("aim_err_deg") is not None]
        return {"sim_seconds": round(self.t, 2),
                "kicks": len(kicks),
                "kicks_per_min": round(60 * len(kicks) / max(self.t, 1e-6), 2),
                "assisted_kicks": sum(k["assisted"] for k in kicks),
                "whiffs": sum(1 for s in speeds if s < 0.2),
                "mean_ball_speed_m_s": round(float(np.mean(speeds)), 2) if speeds else None,
                "mean_travel_m": round(float(np.mean([k["travel_m"] for k in done])), 2) if done else None,
                "mean_abs_aim_error_deg": round(float(np.mean(aim_errs)), 1) if aim_errs else None,
                "mean_lineup_error_mm": round(1000 * float(np.mean([k["err_m"] for k in kicks])), 1) if kicks else None,
                "home_aims": sum(1 for k in kicks if k["aim_mode"] == "home"),
                "falls": len(b.events["falls"]),
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sim.add_common_args(ap)
    ap.add_argument("--kick-left", type=str, default=None, help="left-foot kick policy ONNX (default: from the Hub)")
    ap.add_argument("--kick-right", type=str, default=None, help="right-foot kick policy ONNX (default: from the Hub)")
    ap.add_argument("--feet", choices=["both", "left", "right"], default="both")
    ap.add_argument("--assist", action="store_true", help="nudge the ball onto the kick spot when close")
    ap.add_argument("--tolerance", type=float, default=KickParams.tolerance, help="line-up tolerance (m)")
    ap.add_argument("--aim-spread", type=float, default=math.degrees(KickParams.aim_spread),
                    help="random aim: up to this many degrees either side of straight on")
    ap.add_argument("--arena", type=float, default=KickParams.arena_radius,
                    help="soft arena radius (m): outside it, kicks aim back towards the middle")
    ap.add_argument("--follow", action="store_true", help="camera follows the duck instead of showing the arena")
    args = ap.parse_args()

    rl_repo = sim.find_rl_repo(args.rl_repo)
    ip = sim.import_infer_policy(rl_repo)
    walk = sim.walking_policy(args.policy, args.policy_revision)

    train_hint = ("or train one in microduck_rl: `uv run train Mjlab-BallKick-Flat-MicroDuck` "
                  "(KICK_FOOT at the top of microduck_ball_kick_env_cfg.py picks the foot), "
                  "then export it, and pass --kick-right / --kick-left")
    feet = ["left", "right"] if args.feet == "both" else [args.feet]
    paths = {}
    for foot in feet:
        given = getattr(args, f"kick_{foot}")
        paths[foot] = given or sim.hub_policy(["kick", foot], f"{foot}-foot kick", args.policy_revision,
                                              exact=f"ball_kick_{foot}.onnx", how_to_train=train_hint)

    model, data, bam_ctrl, policy = sim.build_sim(ip, rl_repo, walk, use_bam=not args.no_bam,
                                                  kick_left=paths.get("left"), kick_right=paths.get("right"))
    duck = sim.Duck(model, data, bam_ctrl, policy, np.random.default_rng(args.seed))
    params = KickParams(assist=args.assist, tolerance=args.tolerance,
                        aim_spread=math.radians(args.aim_spread), arena_radius=args.arena)
    brain = KickBrain(params, feet, rng=np.random.default_rng(args.seed))
    game = KickLoop(duck, brain, follow=args.follow)
    game.start()

    print("\nKicking about! " + ("(headless)" if args.headless else "Close the viewer window to stop."))
    if args.headless:
        sim.run_headless(duck, game, args.duration, sim.default_out("kick", args.out), args.video)
    else:
        sim.run_live(duck, game)


if __name__ == "__main__":
    main()

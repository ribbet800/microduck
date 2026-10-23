#!/usr/bin/env python3
"""Fetch, Stage 0: a duck that chases a ball. No training needed.

Pollen's trained walking policy ("the legs") runs exactly as it does in their
`infer_policy.py` rehearsal script — same BAM servo model, same 61-D
observation, same 50 Hz control loop. The only change is who sends the
velocity command: instead of your keyboard, `chaser.chase_command` looks at
where the ball is and steers towards it. When the duck reaches the ball, the
ball jumps somewhere new and the chase starts again.

Two ways to run it:

  Live (watch it in the MuJoCo viewer):
      uv run fetch/chase_ball.py                 # Linux / Windows
      uv run mjpython fetch/chase_ball.py        # macOS needs mjpython for the viewer

  Headless (fast, saves plots + optional video to runs/chase_<time>/):
      uv run fetch/chase_ball.py --headless --duration 90 --video

In the live viewer you can mess with the ball: double-click it to select,
then Ctrl + right-drag to shove it around.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

# Offscreen rendering (headless video) on Linux without a display needs EGL, and
# MuJoCo picks its GL backend when first imported — so set it before any import.
if "--headless" in sys.argv and "MUJOCO_GL" not in os.environ and sys.platform.startswith("linux"):
    os.environ["MUJOCO_GL"] = "egl"

from chaser import ChaserParams, ball_in_duck_frame, chase_command, yaw_from_quat  # noqa: E402
import overlay  # noqa: E402

DEFAULT_POLICY_REPO = "pollen-robotics/microduck-policies"
DEFAULT_POLICY_FILE = "velstand.onnx"   # walks, stands at zero command, gets up after falls

CONTROL_DT = 0.02      # 50 Hz policy, as in training
DECIMATION = 4         # 4 x 5 ms physics steps per policy step
STAND_Z = 0.125        # trunk height used by infer_policy.py's reset
FALLEN_Z = 0.075       # below this the duck is on the floor
RECOVERED_Z = 0.105


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def find_rl_repo(cli_path: str | None) -> Path:
    """Locate a checkout of pollen-robotics/microduck_rl (for the robot model
    and the policy-runner code we reuse)."""
    candidates = [cli_path, os.environ.get("MICRODUCK_RL"),
                  REPO / "third_party" / "microduck_rl", REPO.parent / "microduck_rl"]
    for c in candidates:
        if c and (Path(c) / "scripts" / "infer_policy.py").exists():
            return Path(c).resolve()
    sys.exit(
        "Couldn't find Pollen's microduck_rl repo. Clone it with\n"
        "    git clone https://github.com/pollen-robotics/microduck_rl third_party/microduck_rl\n"
        "or pass --rl-repo /path/to/microduck_rl (or set MICRODUCK_RL)."
    )


def get_policy(path: str | None, revision: str | None) -> str:
    if path:
        return path
    try:
        from huggingface_hub import hf_hub_download
        print(f"Downloading {DEFAULT_POLICY_FILE} from {DEFAULT_POLICY_REPO} (cached after the first time)...")
        return hf_hub_download(DEFAULT_POLICY_REPO, DEFAULT_POLICY_FILE, revision=revision)
    except Exception as e:  # noqa: BLE001
        sys.exit(
            f"Couldn't download the walking policy ({e}).\n\n"
            f"Any of these works — then pass --policy path/to/file.onnx:\n"
            f"  * {DEFAULT_POLICY_FILE} from https://huggingface.co/{DEFAULT_POLICY_REPO}\n"
            f"    (if the repo is gated, run `uv run huggingface-cli login` first)\n"
            f"  * alpha_walking.onnx from Pollen's official policy set (walks, but\n"
            f"    doesn't get up after a fall)\n"
            f"  * your own: in microduck_rl, `uv run train Mjlab-VelStand-Flat-MicroDuck`\n"
            f"    then `uv run scripts/export.py Mjlab-VelStand-Flat-MicroDuck --wandb-run-path ...`"
        )


def build_sim(ip, rl_repo: Path, policy_path: str, use_bam: bool):
    """Load the duck + ball scene and Pollen's PolicyInference runner."""
    import mujoco
    import onnxruntime as ort

    xml = str(rl_repo / ip.MICRODUCK_BALL_XML)   # groundcontact duck + 70 mm ball
    bam_ctrl = None
    if use_bam:
        bam_model = ip.load_bam_model(ip.BAM_KP_FW, 7.4, 0.0)
        model, data, bam_ctrl, _ = ip.load_mujoco_with_bam(xml, bam_model, 0.005, 0.1, ip.BAM_VIN_MIN)
    else:
        model = mujoco.MjModel.from_xml_path(xml)
        model.opt.timestep = 0.005
        data = mujoco.MjData(model)

    obs_dim = ort.InferenceSession(policy_path).get_inputs()[0].shape[-1]
    policy = ip.PolicyInference(model, data, walking_onnx_path=policy_path, bam_ctrl=bam_ctrl,
                                new_cmd_obs=(obs_dim == 61), use_projected_gravity=True)
    return model, data, bam_ctrl, policy


class Duck:
    """Small helper around the sim: reset, read state, step, move the ball."""

    def __init__(self, model, data, bam_ctrl, policy, rng):
        import mujoco
        self.mj = mujoco
        self.model, self.data, self.bam_ctrl, self.policy, self.rng = model, data, bam_ctrl, policy, rng
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self.qadr = int(model.jnt_qposadr[jid])
        self.vadr = int(model.jnt_dofadr[jid])
        self.trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.ball_q = policy.ball_qpos_adr
        self.ball_v = policy.ball_qvel_adr

    def reset(self):
        d, p = self.data, self.policy
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.qpos[self.qadr:self.qadr + 7] = [0, 0, STAND_Z, 1, 0, 0, 0]
        for i, qi in enumerate(p.joint_qpos_indices):
            d.qpos[qi] = p.default_pose[i]
        self.place_ball()
        if self.bam_ctrl is not None:
            self.bam_ctrl.reset(d.qpos)
        p.set_position_targets(p.default_pose)
        p.last_action[:] = 0.0
        self.mj.mj_forward(self.model, d)

    def place_ball(self, min_r=0.6, max_r=1.4):
        """Drop the ball at a random spot 0.6–1.4 m from the duck."""
        x, y = self.xy
        r = self.rng.uniform(min_r, max_r)
        a = self.rng.uniform(-math.pi, math.pi)
        bx, by = x + r * math.cos(a), y + r * math.sin(a)
        self.data.qpos[self.ball_q:self.ball_q + 7] = [bx, by, 0.035, 1, 0, 0, 0]
        self.data.qvel[self.ball_v:self.ball_v + 6] = 0.0
        return bx, by

    @property
    def xy(self):
        return self.data.qpos[self.qadr:self.qadr + 2].copy()

    @property
    def z(self):
        return float(self.data.qpos[self.qadr + 2])

    @property
    def yaw(self):
        return yaw_from_quat(self.data.qpos[self.qadr + 3:self.qadr + 7])

    @property
    def ball(self):
        return self.data.qpos[self.ball_q:self.ball_q + 3].copy()

    def speeds(self):
        """(forward speed in the duck's heading frame, yaw rate)."""
        v = self.data.qvel[self.vadr:self.vadr + 2]
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return float(c * v[0] + s * v[1]), float(self.data.qvel[self.vadr + 5])

    def step(self, cmd):
        """One 50 Hz control step: command -> policy -> 4 physics steps."""
        p = self.policy
        p.vel_cmd = np.asarray(cmd, dtype=np.float32)   # set directly (set_vel_cmd prints every call)
        p._update_command()
        p.apply_action(p.infer())
        for _ in range(DECIMATION):
            if self.bam_ctrl is not None:
                self.bam_ctrl.update()
            self.mj.mj_step(self.model, self.data)


# ---------------------------------------------------------------------------
# The chase loop (shared by live and headless)
# ---------------------------------------------------------------------------

class Chase:
    def __init__(self, duck: Duck, params: ChaserParams, give_up_s: float):
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
        b = self.duck.ball
        self.events["spawns"].append((float(b[0]), float(b[1])))

    def _new_ball(self):
        bx, by = self.duck.place_ball()
        self.events["spawns"].append((bx, by))
        self.ball_t0 = self.t

    def tick(self):
        d, p = self.duck, self.params
        rel = ball_in_duck_frame(d.xy, d.yaw, d.ball[:2])
        cmd, dist, bearing = chase_command(rel, p)

        # Fallen over? Send zero command: the velstand policy gets itself up.
        if not self.fallen and d.z < FALLEN_Z:
            self.fallen = True
            x, y = d.xy
            self.events["falls"].append({"t": self.t, "x": float(x), "y": float(y)})
            print(f"[{self.t:6.1f}s] fell over — getting back up")
        elif self.fallen and d.z > RECOVERED_Z:
            self.fallen = False
            print(f"[{self.t:6.1f}s] back on its feet")
        if self.fallen:
            cmd = np.zeros(3, dtype=np.float32)

        # Reached the ball -> new ball.
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
        self.t += CONTROL_DT

        # Smooth achieved velocities over ~0.5 s so the plot shows the trend
        # rather than the step-by-step waddle.
        vf, wz = d.speeds()
        a = CONTROL_DT / 0.5
        self._vf += a * (vf - self._vf)
        self._wz += a * (wz - self._wz)

        x, y = d.xy
        b = d.ball
        for k, v in (("t", self.t), ("duck_x", x), ("duck_y", y), ("duck_z", d.z), ("duck_yaw", d.yaw),
                     ("ball_x", b[0]), ("ball_y", b[1]), ("dist", dist), ("bearing", bearing),
                     ("cmd_vx", cmd[0]), ("cmd_wz", cmd[2]), ("vel_fwd", self._vf),
                     ("yaw_rate", self._wz), ("fallen", int(self.fallen))):
            self.log[k].append(float(v))
        if int(round(self.t / CONTROL_DT)) % 5 == 0:
            self.trail.append((float(x), float(y)))
            self.trail = self.trail[-200:]

    def draw(self, scn):
        d = self.duck
        overlay.draw(scn, d.xy, d.yaw, d.ball, self.cmd, self.params.catch_radius, self.trail)

    def status(self):
        dist = self.log["dist"][-1] if self.log["dist"] else float("nan")
        return (f"time {self.t:5.1f}s   balls reached {len(self.events['catches'])}   "
                f"falls {len(self.events['falls'])}",
                f"distance {dist:4.2f} m   command fwd {self.cmd[0]:+.2f} m/s  turn {self.cmd[2]:+.2f} rad/s")


# ---------------------------------------------------------------------------
# Live and headless runners
# ---------------------------------------------------------------------------

def run_live(chase: Chase):
    import mujoco
    import mujoco.viewer

    duck = chase.duck
    with mujoco.viewer.launch_passive(duck.model, duck.data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = duck.trunk_id
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 2.4, -45, 135
        last_print = 0.0
        while viewer.is_running():
            t0 = time.time()
            with viewer.lock():
                chase.tick()
                viewer.user_scn.ngeom = 0
                chase.draw(viewer.user_scn)
            line1, line2 = chase.status()
            if hasattr(viewer, "set_texts"):
                viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, line1, line2))
            elif chase.t - last_print > 2.0:
                print(line1, "|", line2)
                last_print = chase.t
            viewer.sync()
            time.sleep(max(0.0, CONTROL_DT - (time.time() - t0)))


def run_headless(chase: Chase, duration: float, out_dir: Path, video: bool, fps: int = 25):
    import mujoco

    duck = chase.duck
    frames, renderer, cam = [], None, None
    if video:
        renderer = mujoco.Renderer(duck.model, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = duck.trunk_id
        cam.distance, cam.elevation, cam.azimuth = 2.4, -45, 135
    every = max(1, round(1.0 / (fps * CONTROL_DT)))

    n_steps = int(duration / CONTROL_DT)
    t0 = time.time()
    for i in range(n_steps):
        chase.tick()
        if renderer is not None and i % every == 0:
            renderer.update_scene(duck.data, camera=cam)
            chase.draw(renderer.scene)
            frames.append(renderer.render().copy())
    wall = time.time() - t0
    print(f"Simulated {duration:.0f}s in {wall:.0f}s of wall time ({duration / max(wall, 1e-6):.1f}x real time)")

    save_outputs(chase, out_dir, frames, fps)


def save_outputs(chase: Chase, out_dir: Path, frames, fps):
    import plots

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "log.csv", "w", newline="") as f:
        w = csv.writer(f)
        keys = list(chase.log)
        w.writerow(keys)
        w.writerows(zip(*(chase.log[k] for k in keys)))

    took = [c["took_s"] for c in chase.events["catches"]]
    summary = {
        "sim_seconds": round(chase.t, 2),
        "balls_reached": len(took),
        "mean_time_to_reach_s": round(float(np.mean(took)), 2) if took else None,
        "falls": len(chase.events["falls"]),
        "gave_up": len(chase.events["timeouts"]),
        "params": vars(chase.params),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "events.json").write_text(json.dumps(chase.events, indent=2))

    r = chase.params.catch_radius
    plots.topdown(chase.log, chase.events, str(out_dir / "topdown.png"), r)
    plots.timeseries(chase.log, chase.events, str(out_dir / "timeseries.png"), r)

    if frames:
        import imageio.v2 as imageio
        imageio.mimwrite(out_dir / "chase.mp4", frames, fps=fps, quality=7, macro_block_size=8)

    print(f"\nSaved to {out_dir}/")
    for name in ("summary.json", "topdown.png", "timeseries.png", "log.csv") + (("chase.mp4",) if frames else ()):
        print(f"  {name}")
    print(json.dumps({k: v for k, v in summary.items() if k != "params"}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true", help="no viewer: run fast and save plots/logs")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds of sim time (headless)")
    ap.add_argument("--video", action="store_true", help="headless: also record chase.mp4")
    ap.add_argument("--out", type=str, default=None, help="output folder (default runs/chase_<time>)")
    ap.add_argument("--policy", type=str, default=None,
                    help=f"walking policy ONNX (default: download {DEFAULT_POLICY_FILE} from {DEFAULT_POLICY_REPO})")
    ap.add_argument("--policy-revision", type=str, default=None, help="Hub revision/tag, e.g. v8")
    ap.add_argument("--rl-repo", type=str, default=None, help="path to a microduck_rl checkout")
    ap.add_argument("--no-bam", action="store_true", help="simple PD servos instead of the BAM model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--give-up", type=float, default=25.0, help="seconds before a ball is respawned")
    ap.add_argument("--max-forward", type=float, default=ChaserParams.max_forward)
    ap.add_argument("--max-turn", type=float, default=ChaserParams.max_turn)
    ap.add_argument("--turn-gain", type=float, default=ChaserParams.turn_gain)
    ap.add_argument("--catch-radius", type=float, default=ChaserParams.catch_radius)
    args = ap.parse_args()

    rl_repo = find_rl_repo(args.rl_repo)
    sys.path.insert(0, str(rl_repo / "scripts"))
    if sys.platform == "win32":
        # Pollen's script imports termios/tty (Mac/Linux only) for its keyboard
        # reader, which the chase never uses. Empty stand-ins let it import on Windows.
        import types
        for name in ("termios", "tty"):
            sys.modules.setdefault(name, types.ModuleType(name))
    import infer_policy as ip   # Pollen's runner: BAM servos, 61-D obs, PolicyInference

    policy_path = get_policy(args.policy, args.policy_revision)
    model, data, bam_ctrl, policy = build_sim(ip, rl_repo, policy_path, use_bam=not args.no_bam)

    params = ChaserParams(max_forward=args.max_forward, max_turn=args.max_turn,
                          turn_gain=args.turn_gain, catch_radius=args.catch_radius)
    duck = Duck(model, data, bam_ctrl, policy, np.random.default_rng(args.seed))
    chase = Chase(duck, params, give_up_s=args.give_up)
    chase.start()

    print("\nChasing! " + ("(headless)" if args.headless else "Close the viewer window to stop."))
    if args.headless:
        out = Path(args.out) if args.out else REPO / "runs" / time.strftime("chase_%Y%m%d-%H%M%S")
        run_headless(chase, args.duration, out, args.video)
    else:
        run_live(chase)


if __name__ == "__main__":
    main()

"""Shared simulation plumbing for every game (fetch, kick loop, ...).

Pollen's `microduck_rl/scripts/infer_policy.py` does the heavy lifting: it
loads the duck with the BAM servo model and runs ONNX policies with the same
61-D observation used in training. This module wraps it so a game only has to
say what command to send each step.

The runners here drive a game (common/game.py: `Game`), which has:
    t                      sim time (s)
    start()                put the duck and ball where they begin
    tick()                 one 50 Hz control step
    draw(scn)              append debug markers to a MuJoCo scene
    status()               list of (label, value) strings for the on-screen panel
    camera                 dict: {"track": True, "distance", "elevation", "azimuth"}
                           or {"track": False, "lookat", "distance", ...}
    log, events            dict of lists / dict of event lists, saved after headless runs
    summary()              dict saved as summary.json
    plots(out_dir)         write any PNGs
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

POLICY_REPO = "pollen-robotics/microduck-policies"

CONTROL_DT = 0.02      # 50 Hz policy, as in training
DECIMATION = 4         # 4 x 5 ms physics steps per policy step
STAND_Z = 0.125        # trunk height used by infer_policy.py's reset
FALLEN_Z = 0.075       # below this the duck is on the floor
RECOVERED_Z = 0.105


def setup_gl_for_headless():
    """Offscreen rendering on Linux without a display needs EGL, and MuJoCo picks
    its GL backend when first imported — so call this before importing mujoco."""
    if "--headless" in sys.argv and "MUJOCO_GL" not in os.environ and sys.platform.startswith("linux"):
        os.environ["MUJOCO_GL"] = "egl"


# ---------------------------------------------------------------------------
# Finding Pollen's code and policies
# ---------------------------------------------------------------------------

def find_rl_repo(cli_path: str | None = None) -> Path:
    """Locate a checkout of pollen-robotics/microduck_rl."""
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


def import_infer_policy(rl_repo: Path):
    """Import Pollen's infer_policy.py as a module."""
    sys.path.insert(0, str(rl_repo / "scripts"))
    if sys.platform == "win32":
        # It imports termios/tty (Mac/Linux only) for its keyboard reader, which
        # we never use. Empty stand-ins let it import on Windows.
        for name in ("termios", "tty"):
            sys.modules.setdefault(name, types.ModuleType(name))
        # It also prints arrows (→), which crash a cp1252 stdout (output piped to a file).
        sys.stdout.reconfigure(errors="replace")
    import infer_policy
    return infer_policy


def hub_policy(keywords: list[str], what: str, revision: str | None = None,
               exact: str | None = None, how_to_train: str = "") -> str:
    """Download a policy from Pollen's Hub repo: `exact` filename if it exists,
    else the first .onnx whose name contains all `keywords`."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        files = list_repo_files(POLICY_REPO, revision=revision)
        onnx = sorted(f for f in files if f.endswith(".onnx"))
        if exact and exact in onnx:
            name = exact
        else:
            matches = [f for f in onnx if all(k in f.lower() for k in keywords)]
            if not matches:
                raise FileNotFoundError(f"no {what} policy among {onnx}")
            name = min(matches, key=len)
        print(f"Using {what} policy {name} from {POLICY_REPO} (downloaded once, then cached)")
        return hf_hub_download(POLICY_REPO, name, revision=revision)
    except Exception as e:  # noqa: BLE001
        sys.exit(
            f"Couldn't get the {what} policy from https://huggingface.co/{POLICY_REPO} ({e}).\n\n"
            f"Download one yourself and pass its path on the command line"
            f"{' — ' + how_to_train if how_to_train else '.'}\n"
            f"If the Hub repo is gated, `uv run huggingface-cli login` first."
        )


def walking_policy(path: str | None, revision: str | None) -> str:
    return path or hub_policy(
        ["velstand"], "walking", revision, exact="velstand.onnx",
        how_to_train=("velstand.onnx (walks AND gets up after falls) is best; alpha_walking.onnx also works. "
                      "Or train your own in microduck_rl: `uv run train Mjlab-VelStand-Flat-MicroDuck`, "
                      "then `uv run scripts/export.py ...`."))


def build_sim(ip, rl_repo: Path, walking_path: str, use_bam: bool = True,
              kick_left: str | None = None, kick_right: str | None = None, kick_duration: float = 3.0):
    """Load the duck + ball scene and Pollen's PolicyInference runner."""
    import mujoco
    import onnxruntime as ort

    xml = str(rl_repo / ip.MICRODUCK_BALL_XML)   # groundcontact duck + 70 mm / 15 g ball
    bam_ctrl = None
    if use_bam:
        bam_model = ip.load_bam_model(ip.BAM_KP_FW, 7.4, 0.0)
        model, data, bam_ctrl, _ = ip.load_mujoco_with_bam(xml, bam_model, 0.005, 0.1, ip.BAM_VIN_MIN)
    else:
        model = mujoco.MjModel.from_xml_path(xml)
        model.opt.timestep = 0.005
        data = mujoco.MjData(model)

    obs_dim = ort.InferenceSession(walking_path).get_inputs()[0].shape[-1]
    if (kick_left or kick_right) and obs_dim != 61:
        sys.exit(f"Kick policies need a 61-D walking policy to hand back to; this one takes {obs_dim} inputs.")
    policy = ip.PolicyInference(model, data, walking_onnx_path=walking_path, bam_ctrl=bam_ctrl,
                                new_cmd_obs=(obs_dim == 61), use_projected_gravity=True,
                                kick_left_onnx_path=kick_left, kick_right_onnx_path=kick_right,
                                kick_duration=kick_duration)
    return model, data, bam_ctrl, policy


# ---------------------------------------------------------------------------
# The duck
# ---------------------------------------------------------------------------

def yaw_from_quat(q) -> float:
    """Heading (rad) from a MuJoCo [w, x, y, z] quaternion."""
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class Duck:
    """Thin wrapper round the sim: reset, read state, step, move the ball, kick."""

    def __init__(self, model, data, bam_ctrl, policy):
        import mujoco
        self.mj = mujoco
        self.model, self.data, self.bam_ctrl, self.policy = model, data, bam_ctrl, policy
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self.qadr = int(model.jnt_qposadr[jid])
        self.vadr = int(model.jnt_dofadr[jid])
        self.trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.ball_q = policy.ball_qpos_adr
        self.ball_v = policy.ball_qvel_adr

    # -- reset / ball ---------------------------------------------------------
    def reset(self, xy=(0.0, 0.0), yaw=0.0):
        d, p = self.data, self.policy
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.qpos[self.qadr:self.qadr + 7] = [xy[0], xy[1], STAND_Z,
                                            math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        for i, qi in enumerate(p.joint_qpos_indices):
            d.qpos[qi] = p.default_pose[i]
        d.qpos[self.ball_q:self.ball_q + 7] = [xy[0] + 5.0, xy[1], 0.035, 1, 0, 0, 0]   # out of the way
        if self.bam_ctrl is not None:
            self.bam_ctrl.reset(d.qpos)
        p.set_position_targets(p.default_pose)
        p.last_action[:] = 0.0
        self.mj.mj_forward(self.model, d)

    def place_ball(self, x, y):
        self.data.qpos[self.ball_q:self.ball_q + 7] = [x, y, 0.035, 1, 0, 0, 0]
        self.data.qvel[self.ball_v:self.ball_v + 6] = 0.0

    # -- state ------------------------------------------------------------------
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

    @property
    def ball_speed(self):
        return float(np.linalg.norm(self.data.qvel[self.ball_v:self.ball_v + 2]))

    def speeds(self):
        """(forward speed in the duck's heading frame, yaw rate)."""
        v = self.data.qvel[self.vadr:self.vadr + 2]
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return float(c * v[0] + s * v[1]), float(self.data.qvel[self.vadr + 5])

    # -- kicking ------------------------------------------------------------------
    def can_kick(self, foot: str) -> bool:
        return f"kick_{foot}" in self.policy.behavior_sessions

    @property
    def kicking(self) -> bool:
        return self.policy.behavior_mode is not None

    def start_kick(self, foot: str):
        """Swap in the kick policy WITHOUT moving the ball. (Pollen's own
        trigger_behavior teleports the ball into the perfect spot; here the duck
        has to line itself up.) The runner hands back to walking afterwards."""
        p, name = self.policy, f"kick_{foot}"
        p.behavior_mode = name
        p.behavior_time_left = p.behavior_durations[name]
        p.vel_cmd = np.zeros(3, dtype=np.float32)
        p.current_policy = name
        p.ort_session = p.behavior_sessions[name]
        p._update_command()

    def stop_kick(self):
        if self.kicking:
            self.policy._end_behavior()

    # -- stepping -------------------------------------------------------------------
    def step(self, cmd=(0.0, 0.0, 0.0)):
        """One 50 Hz control step: command -> policy -> 4 physics steps.
        While a kick is running the command is ignored (kicks use an all-zero one)."""
        p = self.policy
        if not self.kicking:
            p.vel_cmd = np.asarray(cmd, dtype=np.float32)   # set directly: set_vel_cmd prints every call
            p._update_command()
        p.apply_action(p.infer())
        p.update_behavior(CONTROL_DT)
        for _ in range(DECIMATION):
            if self.bam_ctrl is not None:
                self.bam_ctrl.update()
            self.mj.mj_step(self.model, self.data)


# ---------------------------------------------------------------------------
# Running a game: live viewer or headless
# ---------------------------------------------------------------------------

def _apply_camera(cam, spec, trunk_id):
    import mujoco
    if spec.get("track", True):
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = trunk_id
    else:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = spec["lookat"]
    cam.distance = spec.get("distance", 2.4)
    cam.elevation = spec.get("elevation", -45)
    cam.azimuth = spec.get("azimuth", 135)


def run_live(duck: Duck, game):
    import mujoco
    import mujoco.viewer

    with mujoco.viewer.launch_passive(duck.model, duck.data, show_left_ui=False, show_right_ui=False) as viewer:
        _apply_camera(viewer.cam, game.camera, duck.trunk_id)
        while viewer.is_running():
            t0 = time.time()
            with viewer.lock():
                game.tick()
                viewer.user_scn.ngeom = 0
                game.draw(viewer.user_scn)
            rows = game.status()
            viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                              "\n".join(r[0] for r in rows), "\n".join(r[1] for r in rows)))
            viewer.sync()
            time.sleep(max(0.0, CONTROL_DT - (time.time() - t0)))


def run_headless(duck: Duck, game, duration: float, out_dir: Path, video: bool, fps: int = 25):
    import mujoco

    frames, renderer, cam = [], None, None
    if video:
        renderer = mujoco.Renderer(duck.model, height=480, width=640)
        cam = mujoco.MjvCamera()
        _apply_camera(cam, game.camera, duck.trunk_id)
    every = max(1, round(1.0 / (fps * CONTROL_DT)))

    n_steps = int(duration / CONTROL_DT)
    t0 = time.time()
    for i in range(n_steps):
        game.tick()
        if renderer is not None and i % every == 0:
            renderer.update_scene(duck.data, camera=cam)
            game.draw(renderer.scene)
            frames.append(renderer.render().copy())
    wall = time.time() - t0
    print(f"Simulated {duration:.0f}s in {wall:.0f}s of wall time ({duration / max(wall, 1e-6):.1f}x real time)")
    save_outputs(game, out_dir, frames, fps)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save_outputs(game, out_dir: Path, frames=(), fps=25):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "log.csv", "w", newline="") as f:
        w = csv.writer(f)
        keys = list(game.log)
        w.writerow(keys)
        w.writerows(zip(*(game.log[k] for k in keys)))
    summary = game.summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_jsonable))
    (out_dir / "events.json").write_text(json.dumps(game.events, indent=2, default=_jsonable))
    pngs = game.plots(out_dir) or []
    if frames:
        import imageio.v2 as imageio
        imageio.mimwrite(out_dir / "video.mp4", frames, fps=fps, quality=7, macro_block_size=8)

    print(f"\nSaved to {out_dir}/")
    for name in ["summary.json", *pngs, "log.csv", "events.json"] + (["video.mp4"] if frames else []):
        print(f"  {name}")
    print(json.dumps({k: v for k, v in summary.items() if k != "params"}, indent=2, default=_jsonable))


def add_common_args(ap):
    ap.add_argument("--headless", action="store_true", help="no viewer: run fast and save plots/logs")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds of sim time (headless)")
    ap.add_argument("--video", action="store_true", help="headless: also record video.mp4")
    ap.add_argument("--out", type=str, default=None, help="output folder (default runs/<game>_<time>)")
    ap.add_argument("--policy", type=str, default=None,
                    help=f"walking policy ONNX (default: download velstand.onnx from {POLICY_REPO})")
    ap.add_argument("--policy-revision", type=str, default=None, help="Hub revision/tag, e.g. v8")
    ap.add_argument("--rl-repo", type=str, default=None, help="path to a microduck_rl checkout")
    ap.add_argument("--no-bam", action="store_true", help="simple PD servos instead of the BAM model")
    ap.add_argument("--seed", type=int, default=0)


def default_out(game_name: str, cli_out: str | None) -> Path:
    return Path(cli_out) if cli_out else REPO / "runs" / time.strftime(f"{game_name}_%Y%m%d-%H%M%S")


def make_duck(args, kick_left: str | None = None, kick_right: str | None = None) -> Duck:
    """Everything up to a duck standing in the ball scene, from the common CLI args."""
    rl_repo = find_rl_repo(args.rl_repo)
    ip = import_infer_policy(rl_repo)
    walk = walking_policy(args.policy, args.policy_revision)
    model, data, bam_ctrl, policy = build_sim(ip, rl_repo, walk, use_bam=not args.no_bam,
                                              kick_left=kick_left, kick_right=kick_right)
    return Duck(model, data, bam_ctrl, policy)


def run(duck: Duck, game, args, game_name: str):
    """Start the game, then watch it live or run it headless and save the outputs."""
    game.start()
    print(f"\nRunning {game_name}: " + ("headless" if args.headless else "close the viewer window to stop."))
    if args.headless:
        run_headless(duck, game, args.duration, default_out(game_name, args.out), args.video)
    else:
        run_live(duck, game)

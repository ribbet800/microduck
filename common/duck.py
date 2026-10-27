"""The simulated duck: Pollen's scene, servo model and policies, behind a small API.

Pollen's `microduck_rl/scripts/infer_policy.py` does the heavy lifting: it
loads the duck with the BAM servo model and runs ONNX policies with the 61-D
observation every Microduck policy shares. `Duck` wraps it so the rest of this
repo only has to say, every 50 Hz step, what walking command to send and
whether to start a skill.

Skills are Pollen-style episodic policies (e.g. `ball_kick_left.onnx`): a
trained ONNX that takes over from walking for a fixed time, then hands back.
Load any number of them by name (`load_duck(..., skills={"kick_left": (path, 3.0)})`)
and a brain starts one with `Action(skill="kick_left")`. A new trick trained in
microduck_rl plugs in the same way.
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from common.steer import yaw_from_quat

REPO = Path(__file__).resolve().parents[1]
POLICY_REPO = "pollen-robotics/microduck-policies"

CONTROL_DT = 0.02      # 50 Hz policy, as in training
DECIMATION = 4         # 4 x 5 ms physics steps per policy step
STAND_Z = 0.125        # trunk height used by infer_policy.py's reset
FALLEN_Z = 0.075       # below this the duck is on the floor
RECOVERED_Z = 0.105
OBS_DIM = 61           # the observation every Microduck policy shares (AGENTS.md in microduck_rl)


# ---------------------------------------------------------------------------
# Finding Pollen's code and policies
# ---------------------------------------------------------------------------

def find_rl_repo(cli_path: str | None = None) -> Path:
    """Locate a checkout of pollen-robotics/microduck_rl: --rl-repo, else third_party/."""
    path = Path(cli_path) if cli_path else REPO / "third_party" / "microduck_rl"
    if (path / "scripts" / "infer_policy.py").exists():
        return path.resolve()
    sys.exit(
        f"Couldn't find Pollen's microduck_rl repo at {path}. Clone it with\n"
        "    git clone https://github.com/pollen-robotics/microduck_rl third_party/microduck_rl\n"
        "or pass --rl-repo /path/to/microduck_rl."
    )


def _import_infer_policy(rl_repo: Path):
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


def hub_policy(name: str, revision: str | None = None, how_to_train: str = "") -> str:
    """Download policy `name` from Pollen's Hub repo (once; then it's cached)."""
    try:
        path = hf_hub_download(POLICY_REPO, name, revision=revision)
        print(f"Using {name} from {POLICY_REPO}")
        return path
    except Exception as e:  # noqa: BLE001
        sys.exit(
            f"Couldn't get {name} from https://huggingface.co/{POLICY_REPO} ({e}).\n\n"
            f"Download it yourself and pass its path on the command line"
            f"{' — ' + how_to_train if how_to_train else '.'}\n"
            f"If the Hub repo is gated, `uv run huggingface-cli login` first."
        )


def _session(path: str, what: str) -> ort.InferenceSession:
    s = ort.InferenceSession(path)
    dim = s.get_inputs()[0].shape[-1]
    if dim != OBS_DIM:
        sys.exit(f"The {what} policy {path} takes {dim} inputs; Microduck policies take {OBS_DIM}.")
    return s


# Pollen's ball sets a rolling friction but keeps MuJoCo's default condim=3,
# which ignores it: a kicked ball rolls at constant speed forever. Measured in
# MuJoCo 3.10, ball launched rolling at 0.6 m/s (a typical kick) on the floor:
# (rolling friction coefficient, metres rolled).
_ROLL_TABLE = ((0.0008, 4.97), (0.0012, 2.48), (0.0015, 1.69), (0.002, 1.02), (0.0025, 0.69),
               (0.003, 0.52), (0.004, 0.35), (0.006, 0.22), (0.01, 0.13))


def set_ball_roll(model, metres: float):
    """Give the ball rolling resistance so a typical kick rolls about `metres`
    (0.13-5 m). 0 leaves Pollen's ball as it is."""
    if not metres:
        return
    mu, dist = np.log(np.array(_ROLL_TABLE)).T
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    model.geom_condim[g] = 6                       # turns on torsional + rolling friction
    model.geom_friction[g, 2] = float(np.exp(np.interp(np.log(metres), dist[::-1], mu[::-1])))


def load_duck(rl_repo: Path, walking_path: str, skills: dict[str, tuple[str, float]] | None = None,
              ball_roll: float = 1.0) -> Duck:
    """The duck + ball scene with Pollen's BAM servo model, walking policy, and
    `skills` ({name: (onnx path, seconds it runs)})."""
    ip = _import_infer_policy(rl_repo)
    xml = str(rl_repo / ip.MICRODUCK_BALL_XML)   # groundcontact duck + 70 mm / 15 g ball
    bam_model = ip.load_bam_model(ip.BAM_KP_FW, 7.4, 0.0)
    model, data, bam_ctrl, _ = ip.load_mujoco_with_bam(xml, bam_model, 0.005, 0.1, ip.BAM_VIN_MIN)
    set_ball_roll(model, ball_roll)
    _session(walking_path, "walking")
    policy = ip.PolicyInference(model, data, walking_onnx_path=walking_path, bam_ctrl=bam_ctrl,
                                new_cmd_obs=True, use_projected_gravity=True)
    # Pollen's runner keeps episodic policies in these two dicts and swaps them
    # in by name; its constructor only knows kicks and the roulade, so fill them here.
    for name, (path, seconds) in (skills or {}).items():
        policy.behavior_sessions[name] = _session(path, name)
        policy.behavior_durations[name] = seconds
    return Duck(model, data, bam_ctrl, policy)


# ---------------------------------------------------------------------------
# The duck
# ---------------------------------------------------------------------------

class Duck:
    """Thin wrapper round the sim: reset, read state, step, move the ball, run skills."""

    def __init__(self, model, data, bam_ctrl, policy):
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
        # All of MuJoCo's state, not just positions and velocities, and no leftover
        # skill or command. (Some of Pollen's runner state still carries over: for
        # identical games, load a fresh Duck.)
        mujoco.mj_resetData(self.model, d)
        self.stop_skill()
        p.vel_cmd = np.zeros(3, dtype=np.float32)
        p.command[:] = 0.0
        d.qpos[self.qadr:self.qadr + 7] = [xy[0], xy[1], STAND_Z,
                                            math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        for i, qi in enumerate(p.joint_qpos_indices):
            d.qpos[qi] = p.default_pose[i]
        d.qpos[self.ball_q:self.ball_q + 7] = [xy[0] + 5.0, xy[1], 0.035, 1, 0, 0, 0]   # out of the way
        self.bam_ctrl.reset(d.qpos)
        p.set_position_targets(p.default_pose)
        p.last_action[:] = 0.0
        mujoco.mj_forward(self.model, d)

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

    # -- skills -------------------------------------------------------------------
    @property
    def skill(self) -> str | None:
        """The skill running now, or None while walking."""
        return self.policy.behavior_mode

    def start_skill(self, name: str):
        """Swap in skill `name` WITHOUT touching the ball. (Pollen's own
        trigger_behavior teleports the ball into the perfect spot before a kick;
        here the brain has to line the duck up.) `step` hands back to walking
        when its time is up."""
        p = self.policy
        if name not in p.behavior_sessions:
            raise RuntimeError(f"the brain asked for skill {name!r}, which isn't loaded ({list(p.behavior_sessions)})")
        p.behavior_mode = name
        p.behavior_time_left = p.behavior_durations[name]
        p.vel_cmd = np.zeros(3, dtype=np.float32)
        p.current_policy = name
        p.ort_session = p.behavior_sessions[name]
        p._update_command()

    def stop_skill(self):
        if self.skill:
            self.policy._end_behavior()

    # -- stepping -------------------------------------------------------------------
    def step(self, cmd=(0.0, 0.0, 0.0)):
        """One 50 Hz control step: command -> policy -> 4 physics steps.
        While a skill is running the command is ignored (skills use an all-zero one)."""
        p = self.policy
        if not self.skill:
            p.vel_cmd = np.asarray(cmd, dtype=np.float32)   # set directly: set_vel_cmd prints every call
            p._update_command()
        p.apply_action(p.infer())
        p.update_behavior(CONTROL_DT)
        for _ in range(DECIMATION):
            self.bam_ctrl.update()
            mujoco.mj_step(self.model, self.data)

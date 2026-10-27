#!/usr/bin/env python3
"""Train kickback's lining up by trial and error (PPO) in the toy world, then export it.

The network replaces the hand-written APPROACH + ALIGN (kickback/brain.py:
LearnedKickbackBrain): it has to get the duck behind the ball, facing the aim,
with the ball within 15 mm of the kick spot in front of a foot. Once it's there
the hand-written code settles, kicks and watches, as before.

One training episode is one line-up: a random ball, aim and duck pose, ending
when the duck is lined up (+10), knocks the ball (-10) or runs out of time.
Along the way it's rewarded for getting closer to the stance (where the trunk
must stand) and for turning towards the aim. The toy world's legs are `lazy`
like Pollen's walking policy (no sidestep, no backwards, a dead zone), and its
feet push the ball, so walking through the ball isn't a shortcut.

  uv run --group train kickback/train.py                             # ~20-30 min on a laptop CPU
  uv run kickback/game.py --brain runs/kickback_learned/brain.onnx   # watch it in MuJoCo

Saves to runs/kickback_learned/ (common/training.py): report.html (open this),
progress.mp4 (toy world), mujoco.mp4 (the same checkpoints in MuJoCo),
brain.onnx (what --brain loads), checkpoints/, ppo.zip (to keep training).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so `uv run kickback/train.py` finds the packages

from common.brain import DECIDE_EVERY, MAX_FORWARD, MAX_TURN  # noqa: E402
from common.duck import REPO, hub_policy, load_duck, find_rl_repo  # noqa: E402
from common.steer import wrap  # noqa: E402
from common.toy import DT, KICK_REACH, ToyWorld  # noqa: E402
from common.training import (TrainingLog, export_onnx, film_in_mujoco, progress_video, trace_frame,  # noqa: E402
                             write_report)
from kickback.brain import OBS_SIZE, TRAIN_MARGIN, KickbackBrain, KickbackParams, LearnedKickbackBrain, observe  # noqa: E402
from kickback.game import KICK_SECONDS, Kickback  # noqa: E402

ENDS = ((-0.5, 0.0), (0.5, 0.0))     # the game's line (kickback/game.py, --length 1)
START = ((-0.45, 0.0), 0.0)          # where the game starts the duck (ball at 0, 0)
EPISODE_S = 30.0                     # one line-up attempt at most
EPISODE_DECISIONS = round(EPISODE_S / (DECIDE_EVERY * DT))
DUCK_R = (0.25, 1.2)                 # the duck starts this far from the ball, anywhere round it
PROGRESS = 10.0                      # reward per metre closer to the stance
TURNING = 1.0                        # reward per radian closer to facing the aim
LINED_UP = 10.0
KNOCK = -10.0
KNOCKED = 0.01                       # ball moved this far (m): the duck knocked it
PER_DECISION = -0.01                 # -0.1 a second: hurry up
N_ENVS = 16                          # each in its own process: the toy world is pure Python
PPO_SETTINGS = dict(n_steps=256, batch_size=1024, gamma=0.99)   # x 16 envs: one update per 4096 decisions
# Curriculum: a random walker almost never lands within 15 mm and 8°, so training
# starts with a "lined up" window EASY_START x bigger and shrinks it to the real one
# by EASY_UNTIL of the way through. Scoring and playing always use the real one.
EASY_START, EASY_UNTIL = 4.0, 0.7
SCORE_S, SCORE_STARTS = 120.0, 8     # toy-world score: kicks / min over this many 2-minute games
CLIP_S, MUJOCO_S = 60.0, 90.0        # each checkpoint's toy-world clip / MuJoCo game
LONG_S = 300.0                       # then the final and hand-written brains, longer, in MuJoCo


def unit(a):
    return np.array([math.cos(a), math.sin(a)])


class LineUpEnv(gym.Env):
    """One line-up attempt in the toy world, one step = one decision (0.1 s). The real
    brain runs it, asking this env's latest action whenever it wants a decision."""

    observation_space = gym.spaces.Box(-3.0, 3.0, (OBS_SIZE,), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
    easy = 1.0                           # the curriculum: how many times the real "lined up" window

    def set_easy(self, easy):
        self.easy = easy

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        duck = rng.uniform(*DUCK_R) * unit(rng.uniform(-math.pi, math.pi))
        self.world = ToyWorld(duck_xy=duck, yaw=rng.uniform(-math.pi, math.pi), lazy=True)
        self.action = np.zeros(2, dtype=np.float32)
        p = KickbackParams()
        p = replace(p, tolerance=self.easy * p.tolerance, heading_tol=self.easy * p.heading_tol)
        b = self.brain = LearnedKickbackBrain(ENDS, net=lambda obs: self.action, params=p, margin=TRAIN_MARGIN)
        b.quiet = True
        b.aim, b.aim_from = unit(rng.uniform(-math.pi, math.pi)), self.world.ball.copy()
        s = self.world.snapshot()
        b.start_attempt(s)
        self.phi = self.potential()
        self.decisions = 0
        return observe(s, b.aim, b.spot_rel(), b.cmd), {}

    def potential(self):
        """Minus how far the duck still has to go: to the stance, and turning to the aim."""
        w, b = self.world, self.brain
        dist = float(np.linalg.norm(b.stance_for(w.ball, b.foot) - w.duck))
        turn = abs(wrap(math.atan2(b.aim[1], b.aim[0]) - w.yaw))
        return -(PROGRESS * dist + TURNING * turn)

    def step(self, action):
        w, b = self.world, self.brain
        self.action = np.asarray(action, dtype=np.float32)
        reward, done, outcome = PER_DECISION, False, None
        for _ in range(DECIDE_EVERY):
            w.step(b.update(w.snapshot()))
            if np.linalg.norm(w.ball - b.aim_from) > KNOCKED:
                reward, done, outcome = reward + KNOCK, True, "knocked"
                break
            if b.state == "SETTLE":
                reward, done, outcome = reward + LINED_UP, True, "lined up"
                break
        phi = self.potential()
        if outcome != "knocked":          # (a knock's push shouldn't count as progress)
            reward += phi - self.phi
        self.phi = phi
        self.decisions += 1
        truncated = not done and self.decisions >= EPISODE_DECISIONS
        obs = observe(w.snapshot(), b.aim, b.spot_rel(), b.cmd)
        return obs, reward, done, truncated, {"outcome": outcome}


class Outcomes(BaseCallback):
    """Counts how training episodes ended: lined up, knocked, or out of time."""

    def __init__(self):
        super().__init__()
        self.counts = {"lined up": 0, "knocked": 0, None: 0}

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.counts[info.get("outcome")] += 1
        return True


def toy_kickback(brain, start, seconds, trace=None):
    """A toy-world kickback game (ball at 0, 0, the game's line, a slightly noisy kick).
    Returns (kicks, knocks); appends `trace_frame`s to `trace`."""
    (x, y), yaw = start
    brain.quiet = True
    w = ToyWorld(duck_xy=(x, y), yaw=yaw, lazy=True, kick_noise=0.05)
    for _ in range(int(round(seconds / DT))):
        w.step(brain.update(w.snapshot()))
        if trace is not None:
            trace.append(trace_frame(w))
    return len(brain.events["kicks"]), len(brain.events["knocks"])


def score_starts(n=SCORE_STARTS, seed=1000):
    """The same `n` starts for every brain: the game's own, then random ones round the ball."""
    rng = np.random.default_rng(seed)
    return [START] + [(tuple(rng.uniform(0.3, 0.8) * unit(rng.uniform(-math.pi, math.pi))),
                       rng.uniform(-math.pi, math.pi)) for _ in range(n - 1)]


def kicks_per_min(make_brain):
    """(kicks, knocks) per minute over the same toy-world games, a fresh brain each."""
    kicks = knocks = 0
    starts = score_starts()
    for start in starts:
        k, n = toy_kickback(make_brain(), start, SCORE_S)
        kicks, knocks = kicks + k, knocks + n
    minutes = len(starts) * SCORE_S / 60
    return kicks / minutes, knocks / minutes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=2_000_000, help="training decisions (0.1 s each)")
    ap.add_argument("--checkpoints", type=int, default=10, help="score, save and film the brain this many times")
    ap.add_argument("--out", type=str, default=str(REPO / "runs" / "kickback_learned"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mujoco-seconds", type=float, default=MUJOCO_S, help="how long each checkpoint plays in MuJoCo")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    ends = [np.array(e) for e in ENDS]
    baseline, base_knocks = kicks_per_min(lambda: KickbackBrain(ends))
    print(f"Hand-written KickbackBrain in the toy world: {baseline:.2f} kicks / min, {base_knocks:.2f} knocks / min")

    torch.set_num_threads(1)              # a 64-64 network: threads only add overhead
    env = make_vec_env(LineUpEnv, n_envs=N_ENVS, seed=args.seed, vec_env_cls=SubprocVecEnv)
    model = PPO("MlpPolicy", env, seed=args.seed, verbose=0, **PPO_SETTINGS)
    log, checkpoints, clips, t0 = TrainingLog(), [], [], time.time()
    print(f"{'decisions':>10}  {'kicks/min':>9}  {'knocks/min':>10}  {'minutes':>7}   training episodes since last")
    chunk = max(args.steps // args.checkpoints, 1)
    trained = ""
    while True:
        # Checkpoint: export, score it as the game will run it, film the game's start.
        n = model.num_timesteps
        path = out / "checkpoints" / f"brain_{n:07d}.onnx"
        export_onnx(model, path, OBS_SIZE)
        score, knocks = kicks_per_min(lambda: LearnedKickbackBrain(ends, path))
        checkpoints.append({"decisions": n, "score": score, "knocks": knocks, "file": str(path.relative_to(out))})
        frames = []
        kicked, _ = toy_kickback(LearnedKickbackBrain(ends, path), START, CLIP_S, trace=frames)
        clips.append((f"after {n:,} decisions: {kicked} kicks in {CLIP_S:g} s", frames))
        print(f"{n:>10}  {score:9.2f}  {knocks:10.2f}  {(time.time() - t0) / 60:7.1f}   {trained}", flush=True)
        if n >= args.steps:
            break
        easy = max(1.0, EASY_START - (EASY_START - 1.0) * n / (EASY_UNTIL * args.steps))
        env.env_method("set_easy", easy)
        outcomes = Outcomes()
        model.learn(chunk, callback=[log, outcomes], reset_num_timesteps=False)
        c = outcomes.counts
        total = max(sum(c.values()), 1)
        checkpoints[-1]["training"] = {"easy": easy, "episodes": total, "lined_up": c["lined up"] / total,
                                       "knocked": c["knocked"] / total, "timed_out": c[None] / total}
        trained = (f"window x{easy:.2f}: {total} episodes, {100 * c['lined up'] / total:.0f}% lined up, "
                   f"{100 * c['knocked'] / total:.0f}% knocked, {100 * c[None] / total:.0f}% out of time")
    train_minutes = (time.time() - t0) / 60

    model.save(out / "ppo.zip")
    export_onnx(model, out / "brain.onnx", OBS_SIZE)
    print("Making the toy-world progress video...")
    progress_video(out / "progress.mp4", clips, ring=0, lines=[ENDS], centre=(0.0, 0.0), extent=1.2)

    # The same checkpoints (and the hand-written brain, first) playing the real game in MuJoCo.
    print(f"Playing each checkpoint in MuJoCo for {args.mujoco_seconds:g} s...")
    skills = {f"kick_{f}": (hub_policy(f"ball_kick_{f}.onnx"), KICK_SECONDS) for f in ("left", "right")}
    walking = hub_policy("velstand.onnx")

    def make_game(brain_file):
        brain = LearnedKickbackBrain(ends, out / brain_file) if brain_file else KickbackBrain(ends)
        brain.quiet = True
        return Kickback(load_duck(find_rl_repo(), walking, skills), brain)

    films = [("hand-written brain", None)] + [(f"after {c['decisions']:,} decisions", c["file"]) for c in checkpoints]
    summaries = film_in_mujoco(out / "mujoco.mp4", make_game, films, args.mujoco_seconds)
    mj_base = summaries[0]["kicks_per_min"]
    for c, s in zip(checkpoints, summaries[1:]):
        c["mujoco_score"], c["mujoco"] = s["kicks_per_min"], {k: v for k, v in s.items() if k != "params"}
    final = checkpoints[-1]
    print(f"Final and hand-written brains in MuJoCo for {LONG_S:g} s each...")
    cols = ["kicks", "kicks_per_min", "knocks", "ball_crept", "falls", "timeouts", "line_up_retries",
            "mean_lineup_error_mm", "mean_abs_aim_error_deg", "mean_off_line_m"]
    long_rows = [(label, {k: s[k] for k in cols}) for label, s in zip(
        ["hand-written", "learned (final)"],
        film_in_mujoco(None, make_game, [("hand-written", None), ("final", "brain.onnx")], LONG_S))]
    hw, mj = long_rows[0][1], long_rows[1][1]

    p = KickbackParams()
    write_report(
        out / "report.html", title="Learning to line up a kick",
        summary=(f"A small network learned kickback's lining up (getting behind the ball, facing the aim, ball on "
                 f"the kick spot) by trial and error in the toy world, over {model.num_timesteps:,} decisions "
                 f"({train_minutes:.0f} min). In the toy world the game then made {final['score']:.2f} kicks a "
                 f"minute against {baseline:.2f} for the hand-written brain. In MuJoCo, over {LONG_S / 60:g} minutes: "
                 f"{mj['kicks_per_min']:.2f} kicks a minute against {hw['kicks_per_min']:.2f}; knocks "
                 f"{mj['knocks']} against {hw['knocks']}, times stuck {mj['timeouts']} against {hw['timeouts']}, and "
                 f"{mj['falls']} falls."),
        aim=[
            ("Goal", "Replace the hand-written APPROACH + ALIGN of kickback (kickback/brain.py) with a network: "
                     "get the duck to stand with the ball within "
                     f"{p.tolerance * 1000:.0f} mm of the kick spot in front of a foot, facing the aim within "
                     f"{math.degrees(p.heading_tol):.0f}°, without touching the ball. Then the hand-written code "
                     "settles, kicks (Pollen's blind kick policy) and watches, as before."),
            ("World", "The toy world (common/toy.py), not MuJoCo: the duck is a point with a heading, the ball a "
                      "point that rolls and slows down (0.4 m/s²). Legs are 'lazy' like Pollen's walking policy "
                      "measured in MuJoCo: nothing below ~0.25 m/s or ~0.7 rad/s, ~30% of the commanded forward "
                      "speed, no sidestep, no walking backwards, and the duck stops dead when told to. The feet are "
                      "a 7 x 13 cm rectangle (measured from Pollen's model) that pushes the ball if they touch it. "
                      f"A kick sends the ball straight ahead at 1 m/s only if it is within {KICK_REACH * 1000:.0f} mm "
                      "of the kick spot."),
            ("Lined up", f"In training: within {TRAIN_MARGIN:g} x the tolerance ({TRAIN_MARGIN * p.tolerance * 1000:.0f} mm). "
                         f"Playing: anywhere within it ({p.tolerance * 1000:.0f} mm, what the kick policy was trained "
                         "for). The difference leaves room for the real duck's lunge and sway when it stops."),
            ("Sees", f"{OBS_SIZE} numbers, 10 times a second (kickback/brain.py: observe), all from the duck's point "
                     "of view: where the ball is, how far it is from the kick spot, where the trunk has to stand, "
                     "which way the aim points (sin, cos), which foot, and its last command."),
            ("Does", f"2 numbers: forward speed 0 to {MAX_FORWARD} m/s and turn rate up to ±{MAX_TURN} rad/s, sent "
                     "to the walking policy and held for 0.1 s. The brain itself stops the duck once it is lined up "
                     f"and hands over to the kick after {p.lined_up_s:g} s still."),
            ("Reward", f"+{PROGRESS:g} per metre closer to the stance (the spot the trunk must stand on; minus for "
                       f"moving away), +{TURNING:g} per radian closer to facing the aim, {PER_DECISION * 10:g} a "
                       f"second; +{LINED_UP:g} for lining up and {KNOCK:g} for moving the ball more than "
                       f"{KNOCKED * 1000:.0f} mm, both of which end the episode."),
            ("Episodes", f"One line-up attempt, at most {EPISODE_S:g} s: a random aim, the duck {DUCK_R[0]}-{DUCK_R[1]} m "
                         "from the ball anywhere round it, facing anywhere. So about half the time it starts on the "
                         "wrong side and has to go round the ball."),
            ("Considerations", "The foot is picked by rule (the nearer stance), not learned. 15 mm is tight and "
                               "the toy duck stops dead while the real one lunges and sways, so the MuJoCo section "
                               "is the real test. Tried and dropped (plans.md §3 Stage 1b): playing with the "
                               "training margin (froze for minutes 13 mm off: small turns are in the walking "
                               "policy's dead zone), training with no margin (froze after near misses), -10 for "
                               "running out of time (learned far slower), and a toy duck that wobbles when it "
                               "stops with a longer look-ahead (never kicked in MuJoCo)."),
        ],
        training=[
            ("Decisions", f"{model.num_timesteps:,} (0.1 s of toy time each), in {train_minutes:.0f} min on the CPU"),
            ("Updates", f"{len(log.rows)} PPO updates, one per {PPO_SETTINGS['n_steps'] * N_ENVS:,} decisions "
                        f"({N_ENVS} environments, each in its own process), batch {PPO_SETTINGS['batch_size']}, discount "
                        f"{PPO_SETTINGS['gamma']}, default 64-64 network (stable-baselines3)"),
            ("Curriculum", f"A random walker almost never lands within {p.tolerance * 1000:.0f} mm and "
                           f"{math.degrees(p.heading_tol):.0f}°, so the 'lined up' window started {EASY_START:g}x "
                           f"bigger and shrank steadily to the real one by {EASY_UNTIL:.0%} of the way through. "
                           "Scores always use the real one."),
            ("Training episodes", "; ".join(
                f"to {nxt['decisions']:,}: window x{c['training']['easy']:.1f}, "
                f"{c['training']['lined_up']:.0%} lined up, {c['training']['knocked']:.0%} knocked"
                for c, nxt in zip(checkpoints, checkpoints[1:]))),
            ("Checkpoints", f"{len(checkpoints)}, every {chunk:,} decisions: each exported, scored and filmed"),
            ("Toy score", f"kicks per minute of the full kickback game (learned line-up + hand-written kick), over "
                          f"the same {SCORE_STARTS} toy-world games of {SCORE_S:g} s (the game's start and "
                          f"{SCORE_STARTS - 1} random ones), kick noise ±3°"),
        ],
        checkpoints=checkpoints, rows=log.rows, score_name="kicks / min (toy world)", baseline=baseline,
        mujoco={"video": "mujoco.mp4", "score_name": "kicks / min (MuJoCo)", "baseline": mj_base,
                "seconds": args.mujoco_seconds, "long": {"seconds": LONG_S, "rows": long_rows}},
        commands=[f"uv run kickback/game.py --brain {out / 'brain.onnx'}",
                  f"uv run kickback/game.py --brain {out / 'brain.onnx'} --headless --duration 300 --video",
                  f"uv run kickback/game.py --brain {out / checkpoints[len(checkpoints) // 2]['file']}"])
    print(f"\nSaved to {out}/  (report.html, progress.mp4, mujoco.mp4, brain.onnx, checkpoints/, ppo.zip)")
    print(f"Kicks / min: toy learned {final['score']:.2f} vs hand-written {baseline:.2f}; "
          f"MuJoCo ({LONG_S:g} s) learned {mj['kicks_per_min']:.2f} vs hand-written {hw['kicks_per_min']:.2f}")


if __name__ == "__main__":
    main()

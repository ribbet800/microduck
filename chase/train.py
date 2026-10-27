#!/usr/bin/env python3
"""Train a chase brain by trial and error (PPO) in the toy world, then export it.

The network sees the ball relative to the duck and its own last command
(chase/brain.py: `observe`), and picks forward speed and turn rate 10 times a
second. It's rewarded for getting closer to the ball and for reaching it. The
toy world's legs are `lazy`, like Pollen's walking policy in MuJoCo (dead zone,
~30% speed, no sidestep), so what it learns should carry over.

  uv run --group train chase/train.py                          # ~8 min on a laptop CPU
  uv run chase/game.py --brain runs/chase_learned/brain.onnx   # watch it in MuJoCo

Saves to runs/chase_learned/ (common/training.py): report.html (open this),
progress.mp4 (toy world), mujoco.mp4 (the same checkpoints in MuJoCo),
brain.onnx (what --brain loads), checkpoints/, ppo.zip (to keep training).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so `uv run chase/train.py` finds the packages

from chase.brain import OBS_SIZE, ChaseBrain, LearnedBrain, observe  # noqa: E402
from chase.game import CATCH_RADIUS, SPAWN_R, Chase  # noqa: E402
from common.brain import DECIDE_EVERY, MAX_FORWARD, MAX_TURN, Action, to_command  # noqa: E402
from common.duck import REPO, find_rl_repo, hub_policy, load_duck  # noqa: E402
from common.toy import DT, ToyWorld  # noqa: E402
from common.training import TrainingLog, film_in_mujoco, progress_video, trace_frame, write_report  # noqa: E402

EPISODE_S = 30.0
EPISODE_DECISIONS = round(EPISODE_S / (DECIDE_EVERY * DT))
PROGRESS = 10.0       # reward per metre closer to the ball
CATCH_BONUS = 10.0    # reward for reaching it (then a new ball drops)
PPO_SETTINGS = dict(n_steps=512, batch_size=1024, gamma=0.99)   # x 8 envs: one update per 4096 decisions
MUJOCO_S = 60.0       # each checkpoint's MuJoCo game


def spawn_ball(world, rng):
    r, a = rng.uniform(*SPAWN_R), rng.uniform(-math.pi, math.pi)
    world.ball = world.duck + r * np.array([math.cos(a), math.sin(a)])


class ChaseEnv(gym.Env):
    """The chase game in the toy world, one step = one decision (0.1 s)."""

    observation_space = gym.spaces.Box(-3.0, 3.0, (OBS_SIZE,), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.world = ToyWorld(yaw=self.np_random.uniform(-math.pi, math.pi), lazy=True)
        spawn_ball(self.world, self.np_random)
        self.cmd = np.zeros(3, dtype=np.float32)
        self.catches = self.decisions = 0
        return observe(self.world.snapshot(), self.cmd), {}

    def step(self, action):
        w, self.cmd = self.world, to_command(action)
        reward = 0.0
        for _ in range(DECIDE_EVERY):
            before = float(np.linalg.norm(w.ball - w.duck))
            w.step(Action(cmd=self.cmd))
            dist = float(np.linalg.norm(w.ball - w.duck))
            reward += PROGRESS * (before - dist)
            if dist < CATCH_RADIUS:
                reward += CATCH_BONUS
                self.catches += 1
                spawn_ball(w, self.np_random)
        self.decisions += 1
        truncated = self.decisions >= EPISODE_DECISIONS
        return observe(w.snapshot(), self.cmd), reward, False, truncated, {"catches": self.catches}


def toy_chase(brain, rng, seconds=EPISODE_S, trace=None):
    """One toy-world chase with `brain` (through the same Snapshot -> Action interface
    the real game uses). Returns balls reached; appends `trace_frame`s to `trace`."""
    w = ToyWorld(yaw=rng.uniform(-math.pi, math.pi), lazy=True)
    spawn_ball(w, rng)
    catches = 0
    for _ in range(int(round(seconds / DT))):
        w.step(brain.update(w.snapshot()))
        if trace is not None:
            trace.append(trace_frame(w))
        if np.linalg.norm(w.ball - w.duck) < CATCH_RADIUS:
            catches += 1
            spawn_ball(w, rng)
    return catches


def balls_per_min(make_brain, episodes=20, seed=1000):
    """Balls reached per minute over the same `episodes` toy-world chases, a fresh brain each."""
    rng = np.random.default_rng(seed)
    return sum(toy_chase(make_brain(), rng) for _ in range(episodes)) / (episodes * EPISODE_S / 60)


class _Actor(torch.nn.Module):
    """The policy's deterministic action (its mean), for export."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        return self.policy.get_distribution(obs).distribution.mean


def export_onnx(model, path: Path):
    torch.onnx.export(_Actor(model.policy).eval(), torch.zeros(1, OBS_SIZE), str(path),
                      input_names=["obs"], output_names=["action"], dynamo=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=500_000, help="training decisions (0.1 s each)")
    ap.add_argument("--checkpoints", type=int, default=10, help="score, save and film the brain this many times")
    ap.add_argument("--out", type=str, default=str(REPO / "runs" / "chase_learned"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mujoco-seconds", type=float, default=MUJOCO_S, help="how long each checkpoint plays in MuJoCo")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    baseline = balls_per_min(ChaseBrain)
    print(f"Hand-written ChaseBrain in the toy world: {baseline:.1f} balls / min")

    env = make_vec_env(ChaseEnv, n_envs=8, seed=args.seed)
    model = PPO("MlpPolicy", env, seed=args.seed, verbose=0, **PPO_SETTINGS)
    log, checkpoints, clips, t0 = TrainingLog(), [], [], time.time()
    print(f"{'decisions':>10}  {'balls/min':>9}  {'minutes':>7}")
    chunk = max(args.steps // args.checkpoints, 1)
    while True:
        # Checkpoint: export, score it as the game will run it, film one episode.
        n = model.num_timesteps
        path = out / "checkpoints" / f"brain_{n:07d}.onnx"
        export_onnx(model, path)
        score = balls_per_min(lambda: LearnedBrain(path))
        checkpoints.append({"decisions": n, "score": score, "file": str(path.relative_to(out))})
        frames = []
        caught = toy_chase(LearnedBrain(path), np.random.default_rng(7), seconds=20, trace=frames)
        clips.append((f"after {n:,} decisions: {caught} balls in 20 s", frames))
        print(f"{n:>10}  {score:9.1f}  {(time.time() - t0) / 60:7.1f}")
        if n >= args.steps:
            break
        model.learn(chunk, callback=log, reset_num_timesteps=False)

    minutes = (time.time() - t0) / 60
    model.save(out / "ppo.zip")
    export_onnx(model, out / "brain.onnx")
    print("Making the toy-world progress video...")
    progress_video(out / "progress.mp4", clips)

    # The same checkpoints (and the hand-written brain, first) chasing in MuJoCo, same balls each.
    print(f"Playing each checkpoint in MuJoCo for {args.mujoco_seconds:g} s...")
    walking = hub_policy("velstand.onnx")

    def make_game(brain_file):
        brain = LearnedBrain(out / brain_file) if brain_file else ChaseBrain()
        brain.quiet = True
        return Chase(load_duck(find_rl_repo(), walking), brain, rng=np.random.default_rng(0))

    films = [("hand-written brain", None)] + [(f"after {c['decisions']:,} decisions", c["file"]) for c in checkpoints]
    per_min = [60 * s["balls_reached"] / s["sim_seconds"]
               for s in film_in_mujoco(out / "mujoco.mp4", make_game, films, args.mujoco_seconds)]
    for c, m in zip(checkpoints, per_min[1:]):
        c["mujoco_score"] = m
    final, mj_final, mj_base = checkpoints[-1]["score"], per_min[-1], per_min[0]

    write_report(
        out / "report.html", title="Learning to chase",
        summary=(f"A small network learned to chase the ball by trial and error (PPO) in the toy world, "
                 f"over {model.num_timesteps:,} decisions ({minutes:.0f} min). It finished at {final:.1f} balls "
                 f"a minute in the toy world, against {baseline:.1f} for the hand-written brain; in MuJoCo, "
                 f"{mj_final:.1f} against {mj_base:.1f}."),
        aim=[
            ("Goal", "Learn the whole chase brain: walk to the ball as fast as Pollen's walking policy allows, "
                     "instead of the hand-written steering (chase/brain.py: ChaseBrain)."),
            ("World", "The toy world (common/toy.py), not MuJoCo: the duck is a point with a heading, the ball a "
                      "point. Legs are 'lazy' like Pollen's walking policy measured in MuJoCo: nothing below "
                      "~0.25 m/s or ~0.7 rad/s, ~30% of the commanded forward speed, no sidestep, no walking "
                      f"backwards. Reaching the ball (the trunk within {CATCH_RADIUS} m of it) drops a new one."),
            ("Sees", "7 numbers, 10 times a second (chase/brain.py: observe): the ball's position relative to "
                     "the duck (ahead, left), its distance, its bearing as sin and cos, and the forward and "
                     "turn command it sent last."),
            ("Does", f"2 numbers (to_command): forward speed 0 to {MAX_FORWARD} m/s and turn rate up to "
                     f"±{MAX_TURN} rad/s, sent to Pollen's walking policy and held for 0.1 s."),
            ("Reward", f"+{PROGRESS:g} per metre it gets closer to the ball (and minus for moving away), "
                       f"+{CATCH_BONUS:g} for reaching it (within {CATCH_RADIUS} m); then a new ball drops "
                       f"{SPAWN_R[0]}-{SPAWN_R[1]} m away and the episode goes on."),
            ("Episodes", f"{EPISODE_S:g} s, random starting heading and ball position."),
            ("Considerations", "The toy legs keep their speed while turning hard, and the real ones nearly do: "
                               "the MuJoCo section shows whether what it learned carries over."),
        ],
        training=[
            ("Decisions", f"{model.num_timesteps:,} (0.1 s of toy time each), in {minutes:.0f} min on the CPU"),
            ("Updates", f"{len(log.rows)} PPO updates (stable-baselines3), one per {PPO_SETTINGS['n_steps'] * 8:,} "
                        f"decisions (8 environments), discount {PPO_SETTINGS['gamma']}, default 64-64 network"),
            ("Checkpoints", f"{len(checkpoints)}, every {chunk:,} decisions: each exported, scored and filmed"),
            ("Toy score", "balls reached per minute over the same 20 toy-world episodes, running the exported "
                          "brain exactly as the game does"),
        ],
        checkpoints=checkpoints, rows=log.rows, score_name="balls / min (toy world)", baseline=baseline,
        mujoco={"video": "mujoco.mp4", "score_name": "balls / min (MuJoCo)", "baseline": mj_base,
                "seconds": args.mujoco_seconds},
        commands=[f"uv run chase/game.py --brain {out / 'brain.onnx'}",
                  f"uv run chase/game.py --brain {out / 'brain.onnx'} --headless --duration 180 --video",
                  f"uv run chase/game.py --brain {out / checkpoints[len(checkpoints) // 2]['file']}"])
    print(f"\nSaved to {out}/  (report.html, progress.mp4, mujoco.mp4, brain.onnx, checkpoints/, ppo.zip)")
    print(f"Balls / min: toy learned {final:.1f} vs hand-written {baseline:.1f}; "
          f"MuJoCo learned {mj_final:.1f} vs hand-written {mj_base:.1f}")
    print(f"Watch it: uv run chase/game.py --brain {out / 'brain.onnx'}")


if __name__ == "__main__":
    main()

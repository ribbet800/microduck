"""What every training run saves: a log of how PPO went, a progress video, and an HTML report.

A training script (e.g. chase/train.py) passes `TrainingLog()` as the callback
to `model.learn`, scores a checkpoint brain every so often, records a toy-world
episode of each checkpoint with `trace_frame`, then calls `progress_video` and
`write_report`. Everything lands in the run's folder:

    report.html     what it was trained to do, how the reward and losses went, the video
    progress.mp4    one short toy-world clip per checkpoint, early to late
    checkpoints/    every checkpoint's brain.onnx, to watch any stage in MuJoCo
"""

from __future__ import annotations

import base64
import html
import io
import math
import time

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# What PPO reports after each update, and what to call it.
LOSSES = {"train/policy_gradient_loss": "policy loss", "train/value_loss": "value loss",
          "train/entropy_loss": "entropy loss", "train/approx_kl": "approx KL",
          "train/clip_fraction": "clip fraction", "train/explained_variance": "explained variance"}


class TrainingLog(BaseCallback):
    """Records, once per PPO update: decisions so far, mean episode reward, and the losses."""

    def __init__(self):
        super().__init__()
        self.rows: list[dict] = []

    def _on_rollout_start(self):
        # PPO records its losses during the previous update; they're still in the
        # logger here, at the start of the next rollout.
        vals = self.model.logger.name_to_value
        if "train/value_loss" not in vals:
            return
        eps = self.model.ep_info_buffer
        self.rows.append({"decisions": self.num_timesteps,
                          "episode_reward": float(np.mean([e["r"] for e in eps])) if eps else float("nan"),
                          **{name: float(vals[k]) for k, name in LOSSES.items() if k in vals}})

    def _on_step(self):
        return True


def trace_frame(world) -> tuple:
    """One moment of a toy-world episode, for `progress_video`: (t, duck x, y, yaw, ball x, y)."""
    return (world.t, *map(float, world.duck), float(world.yaw), *map(float, world.ball))


def progress_video(path, clips, every_s=0.2, fps=25, extent=1.8):
    """A top-down video, one clip per checkpoint. `clips`: [(title, frames)], frames
    from `trace_frame`, each clip centred on where its duck started."""
    writer = imageio.get_writer(str(path), fps=fps, quality=7, macro_block_size=8)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    for title, frames in clips:
        step = max(1, round(every_s / max(frames[1][0] - frames[0][0], 1e-9))) if len(frames) > 1 else 1
        cx, cy = frames[0][1], frames[0][2]
        for i in range(0, len(frames), step):
            t, x, y, yaw, bx, by = frames[i]
            past = np.array([f[1:3] for f in frames[:i + 1]])
            ax.clear()
            ax.plot(past[:, 0], past[:, 1], color="0.5", lw=1)
            ax.add_patch(plt.Circle((bx, by), 0.035, color="tab:orange", zorder=3))
            ax.add_patch(plt.Circle((bx, by), 0.13, color="tab:green", alpha=0.15, zorder=1))
            ax.arrow(x, y, 0.12 * math.cos(yaw), 0.12 * math.sin(yaw), width=0.025, color="tab:blue", zorder=4)
            ax.set_xlim(cx - extent, cx + extent)
            ax.set_ylim(cy - extent, cy + extent)
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)
            ax.set_title(f"{title}\nt = {t:4.1f} s", fontsize=10)
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)
    writer.close()


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def _score_chart(checkpoints, score_name, baseline):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot([c["decisions"] for c in checkpoints], [c["score"] for c in checkpoints], "o-", label="learned brain")
    if baseline is not None:
        ax.axhline(baseline, color="k", ls="--", lw=1, label="hand-written brain")
    ax.set_xlabel("training decisions")
    ax.set_ylabel(score_name)
    ax.grid(alpha=0.3)
    ax.legend()
    return _png(fig)


def _log_charts(rows):
    x = [r["decisions"] for r in rows]
    keys = ["episode_reward", *LOSSES.values()]
    keys = [k for k in keys if any(k in r for r in rows)]
    n = len(keys)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(10, 2.6 * ((n + 1) // 2)), squeeze=False)
    for ax, k in zip(axes.flat, keys):
        ax.plot(x, [r.get(k, np.nan) for r in rows], lw=1.2, color="tab:green" if k == "episode_reward" else "tab:blue")
        ax.set_title("mean episode reward" if k == "episode_reward" else k, fontsize=10)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("training decisions")
    return _png(fig)


_GUIDE = [
    ("mean episode reward", "the total reward per episode, averaged over recent episodes. The thing PPO is "
     "trying to push up. Rising, then flattening, means it learned what it can."),
    ("policy loss", "how much the last update changed the policy towards better-than-expected actions. "
     "Noisy and near zero is normal; it is not a score."),
    ("value loss", "how wrong the critic's guesses of future reward were. Falls as it learns to predict, "
     "can rise again when the policy gets better and rewards change."),
    ("entropy loss", "minus how random the policy still is. Creeping up towards zero = it is getting more "
     "decisive (exploring less)."),
    ("approx KL", "how far each update moved the policy. PPO keeps it small on purpose; spikes mean big jumps."),
    ("clip fraction", "share of samples where PPO's clipping kicked in to stop too big a change."),
    ("explained variance", "how much of the reward the critic can predict, 1 = all of it. Approaching 1 is good."),
]


def write_report(path, *, title, summary, objective, checkpoints, rows, score_name, baseline=None,
                 video="progress.mp4", commands=()):
    """A self-contained HTML report of one training run.

    summary     one paragraph: what was trained and how it ended
    objective   [(heading, text)]: what it sees, what it does, the reward, the training setup
    checkpoints [{"decisions", "score", "file"}]: the brain scored every so often
    rows        TrainingLog.rows
    """
    esc = html.escape
    obj = "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in objective)
    cps = "".join(f"<tr><td>{c['decisions']:,}</td><td>{c['score']:.1f}</td><td><code>{esc(c['file'])}</code></td></tr>"
                  for c in checkpoints)
    guide = "".join(f"<li><b>{esc(k)}</b>: {esc(v)}</li>" for k, v in _GUIDE)
    cmds = "".join(f"<pre>{esc(c)}</pre>" for c in commands)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>
  body {{ font: 15px/1.5 system-ui, sans-serif; max-width: 980px; margin: 0 auto; padding: 16px; color: #1b1b1b; background: #fff; }}
  h1 {{ font-size: 24px; margin-bottom: 4px; }} h2 {{ font-size: 18px; margin-top: 28px; border-bottom: 1px solid #ddd; }}
  table {{ border-collapse: collapse; width: 100%; }} th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #eee; vertical-align: top; }}
  th {{ width: 180px; color: #444; }} img, video {{ max-width: 100%; }} pre {{ background: #f4f4f4; padding: 8px; overflow-x: auto; }}
  .meta {{ color: #666; }}
</style></head><body>
<h1>{esc(title)}</h1>
<p class="meta">Trained {time.strftime('%Y-%m-%d %H:%M')}</p>
<p>{esc(summary)}</p>
<h2>What it was trained to do</h2><table>{obj}</table>
<h2>How it improved</h2>
<p>Every checkpoint scored on the same toy-world episodes, against the hand-written brain.</p>
{_score_chart(checkpoints, score_name, baseline)}
<h2>Watch it learn</h2>
<p>One clip per checkpoint, same starting position, early to late.</p>
<video src="{esc(video)}" controls loop muted playsinline></video>
<h2>Reward and losses</h2>
<p>Logged after every PPO update.</p>
{_log_charts(rows)}
<ul>{guide}</ul>
<h2>Checkpoints</h2>
<table><tr><th>decisions</th><th>{esc(score_name)}</th><th>file</th></tr>{cps}</table>
<h2>Try it</h2>{cmds}
</body></html>"""
    path.write_text(page, encoding="utf-8")

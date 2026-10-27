"""What every training run saves: a log of how PPO went, progress videos, and an HTML report.

A training script (e.g. chase/train.py) passes `TrainingLog()` as the callback
to `model.learn`, scores a checkpoint brain every so often, records a toy-world
episode of each checkpoint with `trace_frame`, then calls `progress_video`,
`film_in_mujoco` (the same checkpoints playing the real game) and `write_report`.
Everything lands in the run's folder:

    report.html     aim, training, results, and how it does in MuJoCo (open this)
    progress.mp4    one short toy-world clip per checkpoint, early to late
    mujoco.mp4      the same checkpoints in MuJoCo, with Pollen's policies as the legs
    checkpoints/    every checkpoint's brain.onnx, to watch any stage in MuJoCo

A brain trained straight in MuJoCo skips the toy parts: its progress video is
already MuJoCo (`write_report(..., world="mujoco")`, no `mujoco=`).
"""

from __future__ import annotations

import base64
import html
import io
import json
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


def progress_video(path, clips, every_s=0.2, fps=25, extent=1.8, ring=0.13, lines=(), centre=None):
    """A top-down video of the toy world, one clip per checkpoint. `clips`: [(title, frames)],
    frames from `trace_frame`, each clip centred on `centre` (default: where its duck started). `ring`: a green
    disc this big round the ball (e.g. a catch radius; 0 = none). `lines`: fixed segments
    to draw ([((x0, y0), (x1, y1))], e.g. kickback's line)."""
    writer = imageio.get_writer(str(path), fps=fps, quality=7, macro_block_size=8)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    for title, frames in clips:
        step = max(1, round(every_s / max(frames[1][0] - frames[0][0], 1e-9))) if len(frames) > 1 else 1
        cx, cy = centre if centre is not None else (frames[0][1], frames[0][2])
        for i in range(0, len(frames), step):
            t, x, y, yaw, bx, by = frames[i]
            past = np.array([f[1:3] for f in frames[:i + 1]])
            ax.clear()
            for a, b in lines:
                ax.plot([a[0], b[0]], [a[1], b[1]], color="0.8", lw=3, zorder=0)
            ax.plot(past[:, 0], past[:, 1], color="0.5", lw=1)
            ax.add_patch(plt.Circle((bx, by), 0.035, color="tab:orange", zorder=3))
            if ring:
                ax.add_patch(plt.Circle((bx, by), ring, color="tab:green", alpha=0.15, zorder=1))
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


def film_in_mujoco(path, make_game, brains, seconds, speed=4.0, fps=25, size=(960, 544)):
    """Play the real game in MuJoCo once per brain, `seconds` each, and film them into
    one video (`speed` x real time, each clip captioned with its title; `path` None:
    no video, much faster). `make_game(brain_file)` -> a fresh Game on a freshly loaded
    Duck (playing game after game on one Duck drifts: a brain that kicked on its own
    froze after ten other games). `brains`: [(title, brain_file)]. Returns each
    game's `summary()`."""
    import mujoco
    from PIL import Image, ImageDraw

    from common.duck import CONTROL_DT
    from common.run import _apply_camera

    writer = imageio.get_writer(str(path), fps=fps, quality=7, macro_block_size=8) if path else None
    renderer, summaries = None, []
    for title, brain_file in brains:
        game = make_game(brain_file)
        game.start()
        duck = game.duck
        if writer and renderer is None:
            vis = duck.model.vis.global_
            vis.offwidth, vis.offheight = max(vis.offwidth, size[0]), max(vis.offheight, size[1])
            renderer = mujoco.Renderer(duck.model, width=size[0], height=size[1])
        cam = mujoco.MjvCamera()
        _apply_camera(cam, game.camera, duck.trunk_id)
        next_frame, every = 0.0, speed / fps
        for i in range(int(round(seconds / CONTROL_DT))):
            game.tick()
            if writer and (i + 1) * CONTROL_DT >= next_frame:
                next_frame += every
                renderer.update_scene(duck.data, camera=cam)
                game.draw(renderer.scene)
                img = Image.fromarray(renderer.render())
                ImageDraw.Draw(img).text((12, 10), f"{title}   t = {game.t:4.1f} s   ({speed:g}x)",
                                         fill=(255, 255, 255), font_size=22, stroke_width=2, stroke_fill=(0, 0, 0))
                writer.append_data(np.asarray(img))
        summaries.append(game.summary())
        print(f"  MuJoCo, {title}: {json.dumps({k: v for k, v in summaries[-1].items() if k != 'params'})}")
    if writer:
        writer.close()
    return summaries


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def _score_chart(checkpoints, score_name, baseline, key="score"):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot([c["decisions"] for c in checkpoints], [c[key] for c in checkpoints], "o-", label="learned brain")
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


def _long_table(long):
    if not long:
        return ""
    esc = html.escape
    cols = list(long["rows"][0][1])
    head = "<th></th>" + "".join(f"<th>{esc(c.replace('_', ' '))}</th>" for c in cols)
    body = "".join(f"<tr><th>{esc(label)}</th>" + "".join(f"<td>{esc(str(r.get(c)))}</td>" for c in cols) + "</tr>"
                   for label, r in long["rows"])
    return (f"<h3>Longer games: {long['seconds'] / 60:g} minutes each</h3>"
            f"<p>One short clip per checkpoint is noisy; these are the final brain and the hand-written one "
            f"over a longer game.</p><table><tr>{head}</tr>{body}</table>")


def write_report(path, *, title, summary, aim, training, checkpoints, rows, score_name, baseline=None,
                 world="toy", video="progress.mp4", mujoco=None, commands=()):
    """A self-contained HTML report of one training run, in four parts:

    1 Aim       `aim` [(heading, text)]: what it's for, the world it trained in (for the
                toy world, exactly what that models), what it sees and does, the reward
    2 Training  `training` [(heading, text)]: how long, how many decisions, settings
    3 Results   `video` (top-down toy-world clips if world="toy"), score per checkpoint,
                reward and losses
    4 MuJoCo    only for toy-trained brains: `mujoco` = {"video", "score_name", "baseline",
                "seconds"}, with each checkpoint's "mujoco_score" in `checkpoints`; optionally
                "long" = {"seconds", "rows": [(label, {column: value})]}: longer games
                (one clip is noisy), shown as a table

    summary     one paragraph: what was trained and how it ended
    checkpoints [{"decisions", "score", "file", "mujoco_score"?}]
    rows        TrainingLog.rows
    """
    esc = html.escape

    def table(pairs):
        return "<table>" + "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in pairs) + "</table>"

    toy = world == "toy"
    where = "toy world" if toy else "MuJoCo"
    guide = "".join(f"<li><b>{esc(k)}</b>: {esc(v)}</li>" for k, v in _GUIDE)
    cmds = "".join(f"<pre>{esc(c)}</pre>" for c in commands)
    head = f"<th>decisions</th><th>{esc(score_name)}</th>" + (
        f"<th>{esc(mujoco['score_name'])}</th>" if mujoco else "") + "<th>file</th>"
    cps = "".join(f"<tr><td>{c['decisions']:,}</td><td>{c['score']:.1f}</td>"
                  + (f"<td>{c['mujoco_score']:.1f}</td>" if mujoco else "")
                  + f"<td><code>{esc(c['file'])}</code></td></tr>" for c in checkpoints)
    watch = (f"<table><tr>{head}</tr>{cps}</table><h3>Watch it yourself</h3>"
             f"<p>From the repo folder (<code>cd microduck</code> first). Any checkpoint file works with "
             f"<code>--brain</code>.</p>{cmds}")
    parts = [f"""<h2>3. Results</h2>
<h3>Watch it learn</h3>
<p>One clip per checkpoint, early to late, in the {where}{" (top-down: blue arrow = duck, orange = ball)" if toy else ""}.</p>
<video src="{esc(video)}" controls loop muted playsinline></video>
<h3>Score per checkpoint</h3>
<p>Each checkpoint scored on the same {where} episodes{", against the hand-written brain" if baseline is not None else ""}.</p>
{_score_chart(checkpoints, score_name, baseline)}
<h3>Reward and losses</h3>
<p>Logged after every PPO update.</p>
{_log_charts(rows)}
<details><summary>What each plot means</summary><ul>{guide}</ul></details>"""]
    if mujoco:
        parts.append(f"""<h2>4. In MuJoCo</h2>
<p>The same checkpoints as the toy-world video, each playing the real game in MuJoCo for
{mujoco['seconds']:g} s, with Pollen's policies as the legs.</p>
<video src="{esc(mujoco['video'])}" controls loop muted playsinline></video>
{_score_chart(checkpoints, mujoco['score_name'], mujoco.get('baseline'), key="mujoco_score")}
{_long_table(mujoco.get("long"))}
{watch}""")
    else:
        parts.append(f"<h3>Checkpoints</h3>{watch}")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>
  body {{ font: 15px/1.5 system-ui, sans-serif; max-width: 980px; margin: 0 auto; padding: 16px; color: #1b1b1b; background: #fff; }}
  h1 {{ font-size: 24px; margin-bottom: 4px; }} h2 {{ font-size: 19px; margin-top: 32px; border-bottom: 1px solid #ddd; }}
  h3 {{ font-size: 16px; margin-top: 20px; }}
  table {{ border-collapse: collapse; width: 100%; }} th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #eee; vertical-align: top; }}
  th {{ width: 180px; color: #444; }} img, video {{ max-width: 100%; }} pre {{ background: #f4f4f4; padding: 8px; overflow-x: auto; }}
  .meta {{ color: #666; }} summary {{ cursor: pointer; color: #444; }}
</style></head><body>
<h1>{esc(title)}</h1>
<p class="meta">Trained {time.strftime("%Y-%m-%d %H:%M")}</p>
<p>{esc(summary)}</p>
<h2>1. Aim</h2>{table(aim)}
<h2>2. Training</h2>{table(training)}
{"".join(parts)}
</body></html>"""
    path.write_text(page, encoding="utf-8")

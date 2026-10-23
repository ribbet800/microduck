"""After-the-run pictures for the kick loop."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

STATE_ORDER = ["APPROACH", "ALIGN", "SETTLE", "KICK", "WATCH", "RECOVER"]
STATE_COLOURS = {"APPROACH": "#4C78A8", "ALIGN": "#F2CF5B", "SETTLE": "#E8A33D",
                 "KICK": "#D9534F", "WATCH": "#B5B5B5", "RECOVER": "#7B3F99"}


def _pitch(ax, p):
    hx, hy = p.pitch_half
    ax.add_patch(Rectangle((-hx, -hy), 2 * hx, 2 * hy, fill=False, ec="0.6", lw=1))
    ax.axvline(0, color="0.8", lw=1, zorder=0)
    for sgn in (-1, 1):
        ax.add_patch(Circle((sgn * p.goal_x, 0), p.goal_radius, color="gold", alpha=0.25, zorder=0))


def topdown(log, events, p, path):
    """Bird's-eye view: duck path, ball path, every kick as an arrow (coloured by
    how fast the ball went), goals, falls."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    _pitch(ax, p)
    ax.plot(log["duck_x"], log["duck_y"], color="0.35", lw=0.8, label="duck path", zorder=2)
    ax.plot(log["ball_x"], log["ball_y"], color="tab:orange", lw=0.8, alpha=0.8, label="ball path", zorder=2)

    kicks = [k for k in events["kicks"] if "peak_ball_speed" in k]
    if kicks:
        speeds = np.array([k["peak_ball_speed"] for k in kicks])
        bx = np.array([k["ball_xy"][0] for k in kicks])
        by = np.array([k["ball_xy"][1] for k in kicks])
        ux = np.array([k["aim"][0] for k in kicks])
        uy = np.array([k["aim"][1] for k in kicks])
        q = ax.quiver(bx, by, ux, uy, speeds, cmap="plasma", clim=(0, max(1.2, speeds.max())),
                      scale=12, width=0.006, zorder=4)
        fig.colorbar(q, ax=ax, label="ball speed after kick (m/s)", shrink=0.8)
        assisted = [k for k in kicks if k["assisted"]]
        if assisted:
            ax.scatter([k["ball_xy"][0] for k in assisted], [k["ball_xy"][1] for k in assisted],
                       s=80, facecolors="none", edgecolors="tab:purple", lw=1.5, zorder=5, label="assisted kick")
    if events["falls"]:
        ax.scatter([f["x"] for f in events["falls"]], [f["y"] for f in events["falls"]],
                   marker="x", color="tab:red", s=70, lw=2.5, zorder=5, label="fell over")
    for g in events["goals"]:
        ax.plot(g["goal"] * p.goal_x, 0, "*", color="tab:green", ms=20, mec="k", zorder=6)

    ax.set_aspect("equal")
    ax.set_xlim(-p.pitch_half[0] - 0.2, p.pitch_half[0] + 0.2)
    ax.set_ylim(-p.pitch_half[1] - 0.2, p.pitch_half[1] + 0.2)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Kick loop from above — {len(events['goals'])} goals from {len(events['kicks'])} kicks "
                 f"(arrows: each kick, aimed at the goal being attacked)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def alignment(events, p, path):
    """Where the ball was, relative to the duck, at the moment of each kick —
    drawn over the duck's feet and the spot the kick policy was trained on.
    This is the whole game: kicks inside the box should fly, ones outside whiff."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    # Feet (roughly, at the standing pose): centres at (0, ±4.2 cm), toe at x≈3.4 cm.
    for sy in (-1, 1):
        ax.add_patch(Rectangle((-0.034, sy * p.spot_y - 0.02), 0.068, 0.04, color="0.75", zorder=1))
        ax.add_patch(Rectangle((p.spot_x - p.tolerance, sy * p.spot_y - p.tolerance), 2 * p.tolerance,
                               2 * p.tolerance, fill=False, ec="tab:green", lw=1.5, zorder=2))
        ax.add_patch(Circle((p.spot_x, sy * p.spot_y), 0.035, fill=False, ec="tab:green", ls="--", lw=1, zorder=2))
    ax.text(0, 0, "duck\n(trunk)", ha="center", va="center", fontsize=8, color="0.4")

    kicks = events["kicks"]
    speeds = [k.get("peak_ball_speed", np.nan) for k in kicks]
    if kicks:
        x = [k["ball_rel"][0] for k in kicks]
        y = [k["ball_rel"][1] for k in kicks]
        sc = ax.scatter(x, y, c=speeds, cmap="plasma", vmin=0, vmax=max(1.2, np.nanmax(speeds + [0])),
                        s=45, edgecolors="k", lw=0.6, zorder=3)
        fig.colorbar(sc, ax=ax, label="ball speed after kick (m/s)", shrink=0.85)
        for k in kicks:
            if k["assisted"]:
                ax.scatter(*k["ball_rel"], s=110, facecolors="none", edgecolors="tab:purple", lw=1.5, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(-0.06, 0.16)
    ax.set_ylim(-0.1, 0.1)
    ax.set_xticks([-0.05, 0, 0.05, 0.1, 0.15])
    ax.set_xlabel("ball ahead of duck (m)")
    ax.set_ylabel("ball to duck's left (m)")
    ax.set_title("Ball position at each kick\n(green box = line-up tolerance, dashed = ball at the trained spot)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    errs = [1000 * k["err_m"] for k in kicks]
    if kicks:
        colours = ["tab:purple" if k["assisted"] else "tab:blue" for k in kicks]
        ax.scatter(errs, speeds, c=colours, s=40, edgecolors="k", lw=0.5)
    ax.axvline(1000 * p.tolerance, color="tab:green", ls="--", lw=1, label="tolerance")
    ax.axhline(0.2, color="tab:red", ls=":", lw=1, label="below this = whiff")
    ax.set_xlabel("line-up error at kick (mm)")
    ax.set_ylabel("ball speed after kick (m/s)")
    ax.set_title("Does precision matter?\n(purple = assisted)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def timeline(log, events, p, path):
    """What the brain was doing over time, how well it was lining up, and the ball's speed."""
    t = np.asarray(log["t"])
    states = log["state"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [0.5, 1, 1]})

    ax = axes[0]
    if len(t):
        start = 0
        for i in range(1, len(t) + 1):
            if i == len(t) or states[i] != states[start]:
                ax.axvspan(t[start], t[i - 1] + 0.02, color=STATE_COLOURS.get(states[start], "0.5"), lw=0)
                start = i
    ax.set_yticks([])
    ax.set_title("What the brain is doing")
    handles = [Rectangle((0, 0), 1, 1, color=STATE_COLOURS[s]) for s in STATE_ORDER]
    ax.legend(handles, STATE_ORDER, ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15),
              frameon=False)

    ax = axes[1]
    ax.plot(t, 1000 * np.asarray(log["align_err"]), color="tab:olive", lw=1)
    ax.axhline(1000 * p.tolerance, color="tab:green", ls="--", lw=1, label="tolerance")
    ax.set_ylabel("line-up error (mm)")
    ax.set_ylim(0, 120)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    ax.plot(t, log["ball_speed"], color="tab:orange", lw=1)
    ax.set_ylabel("ball speed (m/s)")
    ax.set_xlabel("time (s)")

    for ax in axes[1:]:
        for k in events["kicks"]:
            ax.axvline(k["t"], color="tab:red", alpha=0.4, lw=1)
        for g in events["goals"]:
            ax.axvline(g["t"], color="tab:green", alpha=0.8, lw=2)
        ax.grid(alpha=0.3)
    axes[1].set_title("Red lines: kicks   ·   Green lines: goals", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

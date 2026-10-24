"""After-the-run pictures: a top-down map of the chase and time series of what happened."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def topdown(log: dict, events: dict, path: str, catch_radius: float):
    """Bird's-eye view: the duck's path (coloured by time), each ball spawn,
    where each ball was reached, and any falls."""
    fig, ax = plt.subplots(figsize=(7, 7))
    t = np.asarray(log["t"])
    x, y = np.asarray(log["duck_x"]), np.asarray(log["duck_y"])

    sc = ax.scatter(x, y, c=t, s=2, cmap="viridis", zorder=2)
    fig.colorbar(sc, ax=ax, label="time (s)", shrink=0.8)

    for i, (bx, by) in enumerate(events["spawns"]):
        ax.add_patch(plt.Circle((bx, by), catch_radius, color="tab:green", alpha=0.12, zorder=1))
        ax.plot(bx, by, "o", color="tab:orange", ms=9, mec="k", zorder=3)
        ax.annotate(str(i + 1), (bx, by), textcoords="offset points", xytext=(7, 7), fontsize=9)
    if events["catches"]:
        cx, cy = zip(*[(c["duck_x"], c["duck_y"]) for c in events["catches"]])
        ax.plot(cx, cy, "*", color="tab:green", ms=14, mec="k", zorder=4, label="reached ball")
    if events["falls"]:
        fx, fy = zip(*[(f["x"], f["y"]) for f in events["falls"]])
        ax.plot(fx, fy, "x", color="tab:red", ms=11, mew=2.5, zorder=4, label="fell over")
    ax.plot(x[:1], y[:1], "s", color="k", ms=8, zorder=4, label="start")
    ax.plot([], [], "o", color="tab:orange", mec="k", label="ball spawn (numbered)")

    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.15)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Ball chase from above — {len(events['catches'])} reached, "
                 f"{len(events['falls'])} falls")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def timeseries(log: dict, events: dict, path: str, catch_radius: float):
    """Three stacked panels: distance to the ball, and commanded vs achieved
    forward speed and turn rate (how well the 'legs' follow the 'brain')."""
    t = np.asarray(log["t"])
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)

    ax = axes[0]
    ax.plot(t, log["dist"], color="tab:orange", lw=1.2)
    ax.axhline(catch_radius, color="tab:green", ls="--", lw=1, label="catch radius")
    ax.set_ylabel("distance to ball (m)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.plot(t, log["cmd_vx"], color="tab:blue", lw=1.2, label="commanded")
    ax.plot(t, log["vel_fwd"], color="tab:blue", lw=0.8, alpha=0.45, label="achieved")
    ax.set_ylabel("forward speed (m/s)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    ax.plot(t, log["cmd_wz"], color="tab:purple", lw=1.2, label="commanded")
    ax.plot(t, log["yaw_rate"], color="tab:purple", lw=0.8, alpha=0.45, label="achieved")
    ax.set_ylabel("turn rate (rad/s)")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", fontsize=8)

    for ax in axes:
        for c in events["catches"]:
            ax.axvline(c["t"], color="tab:green", alpha=0.5, lw=1)
        for f in events["falls"]:
            ax.axvline(f["t"], color="tab:red", alpha=0.6, lw=1)
        ax.grid(alpha=0.3)
    axes[0].set_title("Green lines: ball reached   ·   Red lines: fell over")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

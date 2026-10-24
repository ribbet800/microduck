"""After-the-run pictures for kickback."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from kickback.brain import WHIFF_SPEED

STATE_ORDER = ["APPROACH", "ALIGN", "SETTLE", "KICK", "WATCH", "RECOVER"]
STATE_COLOURS = {"APPROACH": "#4C78A8", "ALIGN": "#F2CF5B", "SETTLE": "#E8A33D",
                 "KICK": "#D9534F", "WATCH": "#B5B5B5", "RECOVER": "#7B3F99"}


def topdown(log, events, ends, path):
    """Bird's-eye view: the line, duck path, ball path, and every kick as two
    arrows from where the ball was: where it was aimed (grey) and where it
    actually went (coloured by ball speed), both as long as the ball travelled."""
    fig, ax = plt.subplots(figsize=(9, 8))
    (x0, y0), (x1, y1) = ends
    ax.plot([x0, x1], [y0, y1], "o--", color="0.6", lw=1, ms=8, mfc="none", zorder=0)
    ax.plot(log["duck_x"], log["duck_y"], color="0.35", lw=0.8, label="duck path", zorder=2)
    ax.plot(log["ball_x"], log["ball_y"], color="tab:orange", lw=0.8, alpha=0.8, label="ball path", zorder=2)

    kicks = [k for k in events["kicks"] if k.get("actual_deg") is not None]
    if kicks:
        speeds = np.array([k["peak_ball_speed"] for k in kicks])
        bx = np.array([k["ball_xy"][0] for k in kicks])
        by = np.array([k["ball_xy"][1] for k in kicks])
        travel = np.array([k["travel_m"] for k in kicks])
        aim = np.radians([k["aim_deg"] for k in kicks])
        went = np.radians([k["actual_deg"] for k in kicks])
        ax.quiver(bx, by, travel * np.cos(aim), travel * np.sin(aim), color="0.55", alpha=0.7,
                  angles="xy", scale_units="xy", scale=1, width=0.004, zorder=3)
        q = ax.quiver(bx, by, travel * np.cos(went), travel * np.sin(went), speeds, cmap="plasma",
                      clim=(0, max(1.2, speeds.max())), angles="xy", scale_units="xy", scale=1,
                      width=0.006, zorder=4)
        fig.colorbar(q, ax=ax, label="ball speed after kick (m/s)", shrink=0.8)
        for i, k in enumerate(kicks):
            ax.annotate(str(i + 1), k["ball_xy"], textcoords="offset points", xytext=(-9, 5), fontsize=8)
        ax.plot([], [], color="0.55", lw=2, label="aimed")
    if events["falls"]:
        ax.scatter([f["x"] for f in events["falls"]], [f["y"] for f in events["falls"]],
                   marker="x", color="tab:red", s=70, lw=2.5, zorder=5, label="fell over")

    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.1)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Kickback from above — {len(events['kicks'])} kicks\n"
                 "grey arrow = where it was aimed, coloured arrow = where the ball went; dashed = the line")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=5, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def aiming(events, path):
    """How well does 'face that way, then kick blind' send the ball where it was aimed?"""
    kicks = [k for k in events["kicks"] if k.get("aim_err_deg") is not None]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    if kicks:
        errs = [k["aim_err_deg"] for k in kicks]
        ax.hist(errs, bins=np.arange(-60, 61, 5), color="tab:blue", edgecolor="k", lw=0.5)
        ax.axvline(0, color="k", lw=1)
        ax.set_title(f"Where it went minus where it was aimed\n(mean |error| {np.mean(np.abs(errs)):.0f}°, "
                     f"{len(errs)} kicks; + = to the left)")
    else:
        ax.set_title("No finished kicks")
    ax.set_xlabel("direction error (°)")
    ax.set_ylabel("kicks")
    ax.grid(alpha=0.3)

    ax = axes[1]
    if kicks:
        ax.scatter([1000 * k["err_m"] for k in kicks], [k["aim_err_deg"] for k in kicks],
                   color="tab:blue", s=40, edgecolors="k", lw=0.5)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("line-up error at kick (mm)")
    ax.set_ylabel("direction error (°)")
    ax.set_title("Does a sloppier line-up send it off course?")
    ax.grid(alpha=0.3)
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
        ax.scatter(errs, speeds, color="tab:blue", s=40, edgecolors="k", lw=0.5)
    ax.axvline(1000 * p.tolerance, color="tab:green", ls="--", lw=1, label="tolerance")
    ax.axhline(WHIFF_SPEED, color="tab:red", ls=":", lw=1, label="below this = whiff")
    ax.set_xlabel("line-up error at kick (mm)")
    ax.set_ylabel("ball speed after kick (m/s)")
    ax.set_title("Does precision matter?")
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
        ax.grid(alpha=0.3)
    axes[1].set_title("Red lines: kicks", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

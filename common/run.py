"""Running a game: the shared command-line options, the live viewer, headless runs and their outputs.

Every game script ends the same way:

    ap = argparse.ArgumentParser(...); run.add_args(ap); args = ap.parse_args()
    duck = run.duck_from_args(args, skills={...})
    run.play(MyGame(duck, MyBrain()), args)
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from common.duck import CONTROL_DT, POLICY_REPO, REPO, Duck, find_rl_repo, hub_policy, load_duck

MIN_SPEED, MAX_SPEED = 0.125, 16.0
# Live viewer keys (GLFW key codes). MuJoCo's own viewer uses these for the same
# things, but ignores them in the passive mode used here.
KEY_SPACE, KEY_RIGHT, KEY_MINUS, KEY_EQUAL, KEY_PAD_MINUS, KEY_PAD_PLUS = 32, 262, 45, 61, 333, 334
KEYS_HINT = "space pause, right arrow step, - + speed"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def add_args(ap):
    ap.add_argument("--headless", action="store_true", help="no viewer: run fast and save plots/logs")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds of sim time (headless)")
    ap.add_argument("--video", action="store_true", help="headless: also record video.mp4")
    ap.add_argument("--speed", type=float, default=1.0,
                    help=f"playback speed: live, sim time runs this many times faster than real time "
                         f"({MIN_SPEED:g}-{MAX_SPEED:g}, keys - and + change it); headless, the video plays this fast")
    ap.add_argument("--out", type=str, default=None, help="output folder (default runs/<game>_<time>)")
    ap.add_argument("--policy", type=str, default=None,
                    help=f"walking policy ONNX (default: download velstand.onnx from {POLICY_REPO})")
    ap.add_argument("--policy-revision", type=str, default=None, help="Hub revision/tag for downloads, e.g. v8")
    ap.add_argument("--rl-repo", type=str, default=None, help="path to a microduck_rl checkout")
    ap.add_argument("--ball-roll", type=float, default=1.0,
                    help="how far a typical kick rolls (m, 0.13-5); 0 = Pollen's ball, which never slows down")


def duck_from_args(args, skills: dict[str, tuple[str, float]] | None = None) -> Duck:
    """The duck from the shared options, with `skills` ({name: (onnx path, seconds)}) loaded."""
    walking = args.policy or hub_policy(
        "velstand.onnx", args.policy_revision,
        how_to_train="or train your own in microduck_rl: `uv run train Mjlab-VelStand-Flat-MicroDuck`, "
                     "then `uv run scripts/export.py ...`, and pass --policy")
    return load_duck(find_rl_repo(args.rl_repo), walking, skills, ball_roll=args.ball_roll)


def play(game, args):
    """Start the game, then watch it live or run it headless and save the outputs."""
    game.start()
    print(f"\nRunning {game.name}: " + ("headless" if args.headless else
                                         f"close the viewer window to stop. Keys: {KEYS_HINT}."))
    speed = float(np.clip(args.speed, MIN_SPEED, MAX_SPEED))
    if args.headless:
        out = Path(args.out) if args.out else REPO / "runs" / time.strftime(f"{game.name}_%Y%m%d-%H%M%S")
        run_headless(game, args.duration, out, args.video, speed)
    else:
        run_live(game, speed)


# ---------------------------------------------------------------------------
# Live viewer and headless runs
# ---------------------------------------------------------------------------

def _apply_camera(cam, spec, trunk_id):
    """`spec` is a game's `camera` dict (common/game.py)."""
    if spec["track"]:
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = trunk_id
    else:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = spec["lookat"]
    cam.distance, cam.elevation, cam.azimuth = spec["distance"], spec["elevation"], spec["azimuth"]


def _panel(text: str) -> str:
    """MuJoCo's on-screen font only has plain ASCII (anything else draws as boxes):
    spell out the symbols our status lines use, and drop any other."""
    for symbol, plain in (("°", " deg"), ("±", "+/-"), ("—", "-"), ("→", "->"), ("×", "x")):
        text = text.replace(symbol, plain)
    return text.encode("ascii", "ignore").decode()


def run_live(game, speed: float = 1.0):
    """Run the game in MuJoCo's viewer, `speed` x real time (keys change it)."""
    import mujoco.viewer   # here, not at the top: it loads GLFW, which headless EGL machines may lack

    duck = game.duck
    ctl = {"speed": speed, "paused": False, "steps": 0}

    def on_key(key):   # called from the viewer's thread: only flip settings here
        if key == KEY_SPACE:
            ctl["paused"] = not ctl["paused"]
        elif key == KEY_RIGHT:
            ctl["paused"], ctl["steps"] = True, ctl["steps"] + 1
        elif key in (KEY_EQUAL, KEY_PAD_PLUS):
            ctl["speed"] = min(ctl["speed"] * 2, MAX_SPEED)
        elif key in (KEY_MINUS, KEY_PAD_MINUS):
            ctl["speed"] = max(ctl["speed"] / 2, MIN_SPEED)

    frame_dt = 1 / 60          # draw at most this often
    owed = 0.0                 # sim time behind where `speed` says it should be
    with mujoco.viewer.launch_passive(duck.model, duck.data, key_callback=on_key,
                                      show_left_ui=False, show_right_ui=False) as viewer:
        _apply_camera(viewer.cam, game.camera, duck.trunk_id)
        last = time.perf_counter()
        while viewer.is_running():
            t0 = time.perf_counter()
            wall, last = t0 - last, t0
            with viewer.lock():
                if ctl["paused"]:
                    owed = 0.0
                    for _ in range(ctl["steps"]):
                        game.tick()
                    ctl["steps"] = 0
                else:
                    owed += ctl["speed"] * wall
                    # Run the ticks that are due, but give up after a frame's worth
                    # of wall time and forget the rest: if the PC can't keep up, it
                    # runs as fast as it can rather than falling ever further behind.
                    while owed >= CONTROL_DT and time.perf_counter() - t0 < frame_dt:
                        game.tick()
                        owed -= CONTROL_DT
                    owed = min(owed, CONTROL_DT)
                viewer.user_scn.ngeom = 0
                game.draw(viewer.user_scn)
            rows = game.status() + [("speed", "paused" if ctl["paused"] else f"{ctl['speed']:g}x"), ("keys", KEYS_HINT)]
            legend = game.legend
            viewer.set_texts([(None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               _panel("\n".join(r[0] for r in rows)), _panel("\n".join(r[1] for r in rows))),
                              (None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                               _panel("\n".join(r[0] for r in legend)), _panel("\n".join(r[1] for r in legend)))])
            viewer.sync()
            time.sleep(max(0.0, frame_dt - (time.perf_counter() - t0)))


def run_headless(game, duration: float, out_dir: Path, video: bool, speed: float = 1.0,
                 fps: int = 25, size=(1280, 720)):
    """Run as fast as possible, then save the outputs. The video plays at `speed` x real time."""
    duck = game.duck
    renderer = writer = None
    if video:
        vis = duck.model.vis.global_          # the offscreen buffer must fit the frame
        vis.offwidth, vis.offheight = max(vis.offwidth, size[0]), max(vis.offheight, size[1])
        renderer = mujoco.Renderer(duck.model, width=size[0], height=size[1])
        cam = mujoco.MjvCamera()
        _apply_camera(cam, game.camera, duck.trunk_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(out_dir / "video.mp4", fps=fps, quality=7, macro_block_size=8)
    frame_every = speed / fps                 # sim seconds between video frames

    n_steps = int(duration / CONTROL_DT)
    t0 = time.time()
    next_frame = 0.0
    for i in range(n_steps):
        game.tick()
        frame = None
        # Due a frame? (Below 1x, a tick can be due several: slow motion repeats it.)
        while writer is not None and (i + 1) * CONTROL_DT >= next_frame - 1e-9:
            if frame is None:
                renderer.update_scene(duck.data, camera=cam)
                game.draw(renderer.scene)
                frame = renderer.render()
            writer.append_data(frame)
            next_frame += frame_every
    if writer is not None:
        writer.close()
    wall = time.time() - t0
    print(f"Simulated {duration:.0f}s in {wall:.0f}s of wall time ({duration / max(wall, 1e-6):.1f}x real time)")
    save_outputs(game, out_dir, video)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save_outputs(game, out_dir: Path, video=False):
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

    print(f"\nSaved to {out_dir}/")
    for name in ["summary.json", *pngs, "log.csv", "events.json"] + (["video.mp4"] if video else []):
        print(f"  {name}")
    print(json.dumps({k: v for k, v in summary.items() if k != "params"}, indent=2, default=_jsonable))

# Microduck

Fun experiments for a Microduck bipedal robot, built on top of Pollen Robotics'
[microduck_rl](https://github.com/pollen-robotics/microduck_rl) training stack.
Everything here runs in simulation, so it works before the robot arrives.

Two behaviours:

- **chase**: the duck chases a ball; when it reaches it, the ball jumps somewhere new.
- **kickback**: the duck kicks a ball back and forth along a 1 m line.

Pollen's trained policies do the motor control: the **walking policy** (the
legs) turns a velocity command into servo targets, and **skills** such as the
kick take over for a few seconds. What this repo adds is the **brain**: each
50 Hz step it looks at where the duck and ball are and decides which command
to send, and when to start a skill.

```
world ──► Snapshot ──► brain ──► Action ──► Pollen's policies ──► 14 servos (BAM model) ──► MuJoCo
  ▲       (duck, ball)            (walk command,                                              │
  │                                or start a skill)                                          │
  └────────────────────────────────────────────────────────────────────────────────────────────┘
```

Diagrams of the loop and both brains: **[docs/how_it_works.md](docs/how_it_works.md)**.

## Setup (once)

Needs [uv](https://docs.astral.sh/uv/). No GPU needed: this only *runs* policies.

```bash
git clone https://github.com/ribbet800/microduck && cd microduck
git clone https://github.com/pollen-robotics/microduck_rl third_party/microduck_rl
git -C third_party/microduck_rl checkout b02fdea   # the version this code was checked against
uv sync
```

The pin matters: `common/duck.py` drives some of Pollen's internals directly (to
start a skill without teleporting the ball), which a newer microduck_rl may change.

The policies (`velstand.onnx`, which walks, stands still at zero command and
gets back up after a fall, and `ball_kick_{left,right}.onnx`) download from
[pollen-robotics/microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies)
on first run. To use your own, pass `--policy`, `--kick-left` or `--kick-right`.

## Run it

```bash
uv run chase/game.py                                  # watch live (macOS: uv run mjpython chase/game.py)
uv run kickback/game.py
uv run kickback/game.py --headless --duration 180 --video   # no viewer: fast, saves plots + video
```

Headless runs save to `runs/<game>_<time>/`: `summary.json` (the score),
`log.csv` (every step), `events.json` (catches, kicks, falls...), plots, and
`video.mp4` with `--video`. `--help` lists every option.

### What you see

The bottom-left of the viewer repeats this. Every colour means the same thing in every game.

| Marker | Meaning |
|---|---|
| cyan route | the brain's plan: the route it means to walk now |
| pink arrow | the command sent to the legs: where it would take the duck in the next 2 s |
| white arrow | where the duck is **actually** going in the next 2 s. Usually much shorter than the pink one: the walking policy walks at about 30% of the commanded speed |
| white line | where the duck has been |
| green disc (chase) | the catch radius |
| yellow arrow (kickback) | where the next kick should send the ball; the yellow line is where the ball has been |
| red / green disc (kickback) | the kick spot in front of the foot: **green when the ball is on it** |
| top-left text | time, the brain's state and why it last changed, score, playback speed |

Keys: **space** pauses, **right arrow** steps one 50 Hz tick, **-** / **+**
halve / double the playback speed (`--speed 4` starts at 4x). Double-click the
ball, then **Ctrl + right-drag** to shove it.

## Layout

```
common/                 shared by every behaviour
├── brain.py            what a brain sees (Snapshot) and does (Action), + state / why it changed
├── game.py             the 50 Hz loop: falls, Snapshot -> brain -> Action -> duck, logging, trails, markers
├── duck.py             Pollen's scene + policies behind a small API: walk, start a skill, read state
├── run.py              command-line options, live viewer, headless runs and their outputs
├── steer.py            "walk to that point" steering + frame geometry
├── toy.py              a numpy duck + ball (no MuJoCo) for testing and training brains
├── training.py         what every training run saves: report.html, progress.mp4, the PPO log
└── overlay.py          3D markers and the one colour palette they share
chase/                  one folder per behaviour:
├── game.py             rules, drawing, score, plots; the script you run
├── brain.py            CHASE / RECOVER: hand-written (ChaseBrain) or learned (LearnedBrain)
├── train.py            trains the learned brain in the toy world
└── plots.py
kickback/
├── game.py
├── brain.py            APPROACH (orbiting round the ball) / ALIGN / SETTLE / KICK / WATCH / RECOVER
└── plots.py
tests/                  both brains checked in the toy world (uv run pytest, ~15 s)
docs/how_it_works.md    diagrams
plans.md                roadmap, measurements, hand-over notes
```

## Making new things

**A new behaviour.** Copy `chase/` (the simplest). The brain subclasses
`common.brain.Brain` and implements `update(snapshot) -> Action` (the base class
supplies the `state`, `last_change`, `events` and `plan` the game loop reads).
The game subclasses `common.game.Game`: give it a `name`, `start()` (where the
duck and ball begin) and optionally `rules()`, `draw()`, `status()`,
`summary()`, `plots()`. Falls, stepping, logging, trails, the viewer and
headless runs come for free. Test the brain first in `common/toy.py`, like
`tests/test_chase.py`: if it can't work with toy legs, it won't with real ones.

**A new skill** (a trick trained in microduck_rl, like the kick). Train and
export it in Pollen's stack (read `third_party/microduck_rl/AGENTS.md` first),
then load it by name and start it from a brain:

```python
duck = run.duck_from_args(args, skills={"wave": ("wave.onnx", 2.0)})   # name: (ONNX, seconds it runs)
...
return Action(skill="wave")                                             # in the brain's update()
```

Any episodic policy with the shared 61-D observation works: it takes over from
walking for its time, then hands back.

**A learned brain.** Chase has one: a small network trained by trial and
error (PPO) in the toy world, rewarded for getting closer to the ball and for
reaching it. Train it (about 8 minutes on a laptop CPU; the first run installs
PyTorch), then watch it in MuJoCo:

```bash
uv run --group train chase/train.py
```
```bash
uv run chase/game.py --brain runs/chase_learned/brain.onnx
```

Every training run saves to its folder (here `runs/chase_learned/`):

- `report.html`: open this. What it was trained to do (what it sees, does, and is
  rewarded for), its score at each checkpoint against the hand-written brain, the
  reward and PPO losses after every update (with what each one means), and the video.
- `progress.mp4`: one top-down clip per checkpoint, from random flailing to beelining.
- `checkpoints/`: the brain at each stage, so you can watch any of them in MuJoCo with `--brain`.
- `brain.onnx` (the final brain) and `ppo.zip` (to keep training).

`chase/train.py` is the template for training other brains: a Gymnasium env
around `common/toy.py`, with what the brain sees and does defined once in its
`brain.py`, and `common/training.py` for the report and video.

## License

MIT, see [LICENSE](LICENSE). Pollen's microduck_rl is Apache-2.0 (code) and
CC-BY-NC-SA-4.0 (robot models).

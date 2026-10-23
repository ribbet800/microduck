# Microduck

Fun experiments for a Microduck bipedal robot, built on top of Pollen Robotics'
[microduck_rl](https://github.com/pollen-robotics/microduck_rl) training stack.
Everything here runs in simulation, so it works before the robot arrives.

## Fetch, Stage 0: a duck that chases a ball

Pollen's trained walking policy is the **legs**: it turns a velocity command
(forward speed, turn rate) into motor movements. `fetch/chaser.py` is the
**brain**: it looks at where the ball is relative to the duck and decides which
command to send. No training needed. When the duck reaches the ball, the ball
jumps somewhere new and the chase starts again.

```
ball position ──► chaser.py (brain) ──► [vx, vy, turn] ──► Pollen's policy (legs) ──► 14 servos
      ▲                                                                                  │
      └──────────────────────────── MuJoCo physics ◄────────────────────────────────────┘
```

### Setup (once)

Needs [uv](https://docs.astral.sh/uv/). No GPU needed: this only *runs* a policy.

```bash
git clone https://github.com/ribbet800/microduck && cd microduck
git clone https://github.com/pollen-robotics/microduck_rl third_party/microduck_rl
uv sync
```

The walking policy (`velstand.onnx`: walks, stands still at zero command, and
gets back up after a fall) downloads from
[pollen-robotics/microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies)
on first run. If that fails, grab an ONNX policy yourself and pass `--policy path/to/file.onnx`.

### Watch it live

```bash
uv run mjpython fetch/chase_ball.py    # macOS (the MuJoCo viewer needs mjpython)
uv run fetch/chase_ball.py             # Linux / Windows
```

What you're seeing in the viewer:

| Marker | Meaning |
|---|---|
| orange line | duck → ball: what it's aiming at |
| green disc | the catch radius: inside it counts as "reached" |
| blue arrow | the command being sent (longer = faster, swung by the turn command) |
| white dots | trail of where the duck has been |
| top-left text | time, balls reached, falls, current distance and command |

Mess with it: double-click the ball to select it, then **Ctrl + right-drag** to shove it.

### Run it headless and get plots

```bash
uv run fetch/chase_ball.py --headless --duration 90 --video
```

Saves to `runs/chase_<time>/`:

- `topdown.png`: bird's-eye map: the duck's path coloured by time, numbered ball spawns, ★ where each ball was reached, ✕ falls
- `timeseries.png`: distance to the ball, plus **commanded vs achieved** forward speed and turn rate (how well the legs follow the brain)
- `chase.mp4`: video with all the markers drawn in
- `summary.json`: balls reached, average time to reach, falls; `log.csv`: every step

### Things to try

- Tune the brain: `--max-forward 0.35 --turn-gain 2.5` (faster, more aggressive), or edit `fetch/chaser.py`. Keep forward ≤ 0.4 m/s and turn ≤ 1.0 rad/s, the ranges the walking policy was trained on.
- `--no-bam` swaps Pollen's realistic servo model for ideal PD servos. Compare how much worse/better it gets.
- `uv run pytest` checks the brain against an ideal walker (if it can't catch the ball with perfect legs, it won't with real ones).

## Roadmap

- [x] **Stage 0**: hand-coded chaser on top of Pollen's walking policy
- [ ] **Stage 1**: replace the chaser with a small learned policy (reward = getting closer)
- [ ] **Stage 2**: grab the ball with the beak
- [ ] **Stage 3**: carry it back home
- [ ] **Stage 4**: glue it together: find → approach → grab → return → drop
- [ ] **Stage 5**: real robot: camera + colour detector instead of the sim's ball position

## Layout

```
fetch/
├── chase_ball.py   # run it: sim setup, chase loop, live viewer / headless outputs
├── chaser.py       # the brain: ball position -> velocity command
├── overlay.py      # debug markers drawn into the 3D scene
└── plots.py        # top-down map + time series
tests/test_chaser.py
third_party/microduck_rl   # Pollen's repo (cloned, not committed)
```

## License

MIT, see [LICENSE](LICENSE). Pollen's microduck_rl is Apache-2.0 (code) and
CC-BY-NC-SA-4.0 (robot models).

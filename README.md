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
uv run fetch/chase_ball.py             # Windows / Linux
uv run mjpython fetch/chase_ball.py    # macOS (the MuJoCo viewer needs mjpython)
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
- `video.mp4`: video with all the markers drawn in
- `summary.json`: balls reached, average time to reach, falls; `log.csv`: every step

### Things to try

- Tune the brain: `--turn-gain 2.5` (more aggressive), or edit `fetch/chaser.py`. Keep forward ≤ 0.4 m/s and turn ≤ 1.0 rad/s, the ranges the walking policy was trained on.
  The walking policy has a big **dead zone**: from a stand it ignores forward commands below ~0.25 m/s and turns below ~0.7 rad/s, never sidesteps or walks backwards, and walks at only ~30% of the commanded speed. So the brains never send less than `--min-forward` (0.3); set it to 0 to see the duck freeze near the ball.
- `--no-bam` swaps Pollen's realistic servo model for ideal PD servos. Compare how much worse/better it gets.
- `uv run pytest` checks the brain against an ideal walker (if it can't catch the ball with perfect legs, it won't with real ones).

## Kick loop: the duck plays football against itself

A separate game. The duck walks round behind the ball, turns to face the goal,
walks in along the ball→goal line steering like a car until the ball sits
exactly in front of one foot, stands still, then swaps in Pollen's **kick
policy**. Roll the ball into the goal it's attacking to score; then it attacks
the other end.

```bash
uv run kick/kick_loop.py                                  # watch live (macOS: uv run mjpython ...)
uv run kick/kick_loop.py --headless --duration 180 --video
```

The kick policy is **blind**: it never sees the ball, and was trained with the
ball 9 cm ahead and 4.2 cm to the side of the trunk (±1.5 cm). So lining up is
the whole game. The brain (`kick/brain.py`) is a state machine:
APPROACH → ALIGN → SETTLE → KICK → WATCH (+ RECOVER after falls).

| Marker | Meaning |
|---|---|
| yellow disc | the goal being attacked (grey = the other one) |
| yellow line | ball → goal: the line it's lining up on |
| small red/green disc | the kick spot in front of the foot: **green when the ball is on it** |
| white dot + arrow | where the duck needs to stand, and which way to face |
| blue dot | where it's walking to right now |
| arrow from the duck | the command (colour = state: blue approach, yellow align, red kick) |
| white / orange dots | duck and ball trails |

Headless runs save `topdown.png` (every kick as an arrow coloured by ball speed),
`alignment.png` (where the ball was at each kick, drawn over the feet: does
precision matter?) and `timeline.png` (what the brain was doing, line-up error,
ball speed).

Options: `--assist` nudges the ball onto the spot when the duck gets close
(Pollen's own demo always does this), `--feet right` uses one foot only,
`--tolerance 0.02` makes it less fussy, `--follow` makes the camera follow the duck.

## Roadmap

Fetch:
- [x] **Stage 0**: hand-coded chaser on top of Pollen's walking policy
- [ ] **Stage 1**: replace the chaser with a small learned policy (reward = getting closer)
- [ ] **Stage 2**: grab the ball with the beak
- [ ] **Stage 3**: carry it back home
- [ ] **Stage 4**: glue it together: find → approach → grab → return → drop
- [ ] **Stage 5**: real robot: camera + colour detector instead of the sim's ball position

Also:
- [x] **Kick loop**

Detailed plans, design notes, known unknowns and a log are in **[plans.md](plans.md)**.

## Layout

```
common/
├── sim.py          # shared: finds Pollen's code + policies, Duck wrapper, live / headless runners
└── overlay.py      # debug markers drawn into the 3D scene
fetch/
├── chase_ball.py   # the fetch game
├── chaser.py       # its brain: ball position -> velocity command
└── plots.py
kick/
├── kick_loop.py    # the kick game
├── brain.py        # its brain: approach / align / settle / kick / watch
└── kick_plots.py
tests/              # brains checked against an imaginary perfect duck (uv run pytest)
plans.md            # roadmap + hand-over notes
third_party/microduck_rl   # Pollen's repo (cloned, not committed)
```

## License

MIT, see [LICENSE](LICENSE). Pollen's microduck_rl is Apache-2.0 (code) and
CC-BY-NC-SA-4.0 (robot models).

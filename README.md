# Microduck

Fun experiments for a Microduck bipedal robot, built on top of Pollen Robotics'
[microduck_rl](https://github.com/pollen-robotics/microduck_rl) training stack.
Everything here runs in simulation, so it works before the robot arrives.

Two systems: **chase** (fetch, Stage 0) and **kicking** (the endless kick loop,
and its kickback variant). For
diagrams of how the brains loop through their states, and which parts are ours
vs Pollen's pre-trained policies, see **[docs/how_it_works.md](docs/how_it_works.md)**.

## Fetch, Stage 0: a duck that chases a ball

Pollen's trained walking policy is the **legs**: it turns a velocity command
(forward speed, turn rate) into motor movements. `fetch/chaser.py` is the
**brain**: it looks at where the ball is relative to the duck and decides which
command to send (the steering itself is `common/steer.py`). No training needed.
When the duck reaches the ball, the ball jumps somewhere new and the chase starts again.

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
git -C third_party/microduck_rl checkout b02fdea   # the version this code was checked against
uv sync
```

The pin matters: `common/sim.py` drives some of Pollen's internals directly (to
start a kick without teleporting the ball), which a newer microduck_rl may change.

The walking policy (`velstand.onnx`: walks, stands still at zero command, and
gets back up after a fall) downloads from
[pollen-robotics/microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies)
on first run. If that fails, grab an ONNX policy yourself and pass `--policy path/to/file.onnx`.

### Watch it live

```bash
uv run fetch/chase_ball.py             # Windows / Linux
uv run mjpython fetch/chase_ball.py    # macOS (the MuJoCo viewer needs mjpython)
```

What you're seeing in the viewer (every game; each colour means the same thing
everywhere, and the bottom-left of the viewer repeats this):

| Marker | Meaning |
|---|---|
| cyan route | the brain's plan: the route it means to walk now, small arrowheads showing which way, a disc where it ends |
| pink arrow | the command sent to the legs: where it would take the duck in the next 2 s |
| white arrow | where the duck is **actually** going in the next 2 s, at its current (smoothed) speed and turn rate. Same scale as the pink one, so they'd overlap if the legs did exactly as told; when it's standing, a short stub showing which way it faces |
| white line | where the duck has been |
| green disc | the catch radius: inside it counts as "reached" |
| top-left text | time, the brain's state and why it last changed, falls, balls reached, distance, command, playback speed |

The white arrow is usually much shorter than the pink one: the walking policy
walks at about 30% of the commanded speed.

Keys: **space** pauses / resumes, **right arrow** steps one 50 Hz tick (and
pauses), **-** / **+** halve / double the playback speed. `--speed 4` starts
it at 4x (0.125-16x). If the computer can't keep up, it runs as fast as it can.

Mess with it: double-click the ball to select it, then **Ctrl + right-drag** to shove it.

### Run it headless and get plots

```bash
uv run fetch/chase_ball.py --headless --duration 90 --video
```

Saves to `runs/chase_<time>/`:

- `topdown.png`: bird's-eye map: the duck's path coloured by time, numbered ball spawns, ★ where each ball was reached, ✕ falls
- `timeseries.png`: distance to the ball, plus **commanded vs achieved** forward speed and turn rate (how well the legs follow the brain)
- `video.mp4`: video with all the markers drawn in (`--speed 2` makes it play at 2x, `--speed 0.5` in slow motion)
- `summary.json`: balls reached, average time to reach, falls; `log.csv`: every step

### Things to try

- Tune the brain: `--turn-gain 2.5` (more aggressive), or edit `common/steer.py`. Keep forward ≤ 0.4 m/s and turn ≤ 1.0 rad/s, the ranges the walking policy was trained on.
  The walking policy has a big **dead zone**: from a stand it ignores forward commands below ~0.25 m/s and turns below ~0.7 rad/s, never sidesteps or walks backwards, and walks at only ~30% of the commanded speed. So the brains never send less than `--min-forward` (0.3); set it to 0 to see the duck freeze near the ball.
- `uv run pytest` checks the brain against imaginary legs, both perfect ones and ones with the real policy's dead zone (if it can't catch the ball with those, it won't with real ones).

## Kick loop: walk up, kick, follow, kick again

A separate game. Each time the ball stops, the duck picks a **random direction**
to kick it, leaning back towards the middle the further out the ball is, inside
a **soft arena**. It gets behind the ball (orbiting round it when the ball is in
the way), walks in along the line until the ball sits exactly in front of one
foot, stands still, then swaps in Pollen's **kick policy**. Then it follows the
ball and does it again. How each step works: [docs/how_it_works.md](docs/how_it_works.md).

```bash
uv run kick/kick_loop.py                                  # watch live (macOS: uv run mjpython ...)
uv run kick/kick_loop.py --headless --duration 180 --video
```

The kick policy is **blind**: it never sees the ball, kicks straight ahead, and
was trained with the ball 9 cm ahead and 4.2 cm to the side of the trunk
(±1.5 cm). So the kick's direction is just which way the duck faces, and lining
up is the whole game. The brain (`kick/brain.py`) is a state machine:
APPROACH → ALIGN → SETTLE → KICK → WATCH (+ RECOVER after falls). The viewer's
top-left panel shows the current state and why it last changed.

On top of the markers every game has (plan, command, actual motion: see
above), the plan here ends where the duck has to stand to kick, and:

| Marker | Meaning |
|---|---|
| yellow arrow | where the next kick should send the ball, about as far as a kick rolls (`--ball-roll`); **orange** when it's aiming back to the middle. It stays where the kick happened, so you can see how far off the ball went |
| yellow line | where the ball has been |
| small red / green disc | the kick spot in front of the foot: **green when the ball is on it** |
| faint white circle | the soft arena |

Headless runs save `topdown.png` (every kick as two arrows: where it was aimed
and where the ball actually went), `aiming.png` (direction error per kick, and
whether a sloppier line-up sends it further off course), `alignment.png` (where
the ball was at each kick, drawn over the feet) and `timeline.png` (what the
brain was doing, line-up error, ball speed).

Options (`--help` lists them all):

- `--aim-spread 30`: tamer random aims; `--centre-bias 0.5`: a weaker pull back to the middle (0 = always onwards); `--arena 1.0`: a smaller arena
- `--ball-roll 2`: kicks roll ~2 m instead of ~1 m (`0` is Pollen's own ball, which never slows down)
- `--seed 3`: a different, repeatable sequence of aims
- `--assist`: nudge the ball onto the spot when the duck gets close (Pollen's own demo always does this)
- `--feet right`: one foot only; `--tolerance 0.02`: less fussy line-up; `--follow`: the camera follows the duck

### Kickback: back and forth on a line

The same brain, kicking the ball back the way it last went, between the ends of
a line (a ball that drifts off the line is steered back to it). Every kick
means getting round the ball, so this is the orbit's showcase.

```bash
uv run kick/kickback.py
uv run kick/kickback.py --headless --duration 180 --video
```

The faint white line with a disc at each end is the line; the rest of the
markers are the kick loop's. `--length 1.5` makes the line longer (about one
kick's roll works best); the kick loop's other options apply too.
`summary.json` adds `mean_off_line_m`: how far from the line the kicks leave the ball.

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
├── sim.py          # finds Pollen's code + policies, Duck wrapper, live / headless runners
├── game.py         # the loop every game shares: falls, Snapshot -> brain -> Action, logging, trails
├── brain.py        # what a brain sees (Snapshot) and says (Action), + state / last_change
├── steer.py        # "walk to that point" steering + frame geometry, used by both brains
└── overlay.py      # 3D markers and the one colour palette they share
fetch/
├── chase_ball.py   # the chase game: rules (new ball, give up), drawing, plots
├── chaser.py       # its brain: CHASE / RECOVER
└── plots.py
kick/
├── kick_loop.py    # the kick game: drawing, score, plots
├── kickback.py     # the same game kicking back and forth along a line
├── brain.py        # its brain: approach (with orbit) / align / settle / kick / watch / recover,
│                   #   and KickbackBrain (a different pick_aim)
└── plots.py
docs/               # how_it_works.md: diagrams of the loop and both brains
tests/              # brains checked against imaginary legs (uv run pytest)
plans.md            # roadmap + hand-over notes
third_party/microduck_rl   # Pollen's repo (cloned, not committed)
```

## License

MIT, see [LICENSE](LICENSE). Pollen's microduck_rl is Apache-2.0 (code) and
CC-BY-NC-SA-4.0 (robot models).

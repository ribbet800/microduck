# Plans

Roadmap and hand-over notes for this repo. Written so that Jack, or an agent
starting cold, can pick up any item without re-deriving the context. Keep it
current: when something is done, tick it, add what was learned, and move on.

Purpose of the project: **fun**. Simulation games and tricks for a Microduck
(Pollen Robotics / Hugging Face 25 cm biped, arriving early 2027), built on top
of Pollen's open-source stack. Not research, not a CV piece. Prefer things that
are quick to try and good to watch.

---

## 1. Where things stand (2026-10-23)

| Thing | Status |
|---|---|
| `fetch/chase_ball.py`: Fetch Stage 0, hand-coded ball chaser | Works with the real velstand policy: 9 balls / 90 s, 0 falls |
| `kick/kick_loop.py`: endless kicking, random aims in a soft arena | Works with the real policies: ~15 kicks / 5 min (6 seeds), 0 falls, ~10° mean direction error, ball rolls ~0.8 m per kick. Orbits round the ball on reversals |
| `kick/kickback.py`: back and forth along a 1 m line | Works with the real policies: 12 kicks / 5 min, strictly alternating, an orbit before every kick, ball ends ~0.18 m off the line on average |
| `common/`: shared game loop, brain interface, steering, sim setup | Built; matches Pollen's `infer_policy.py` main loop. `docs/how_it_works.md` has diagrams |
| Runs on Jack's Windows laptop | Headless and live viewer both checked (2026-10-23) |

Hub repo and file names are confirmed (`pollen-robotics/microduck-policies`:
`velstand.onnx`, `ball_kick_{left,right}.onnx`).

### The legs (velstand v8) in MuJoCo, measured 2026-10-23 with BAM

The brains are written around these numbers; `tests/legs.py:lazy_legs` models them.

| command | what happens |
|---|---|
| vx < ~0.25 m/s from a stand (0.05, 0.1, 0.15 all 0) | nothing: stands still |
| vx 0.2 / 0.3 / 0.4 | 0.04 / 0.08 / 0.11 m/s (~30%; Pollen's bench says 0.4 → 0.17) |
| once walking, vx 0.15 / 0.1 | keeps walking (0.05 m/s) / stops |
| vy up to 0.3 (standing or walking) | ~nothing: no sidestep |
| vx −0.2 | nothing: no walking backwards |
| turn in place 0.3 / 0.6 / 0.7 / 0.8 | 0 / 0 (bimodal, sometimes −0.59) / 0.57 / 0.73 rad/s |
| 0.25 m/s pulses of 0.1–0.4 s | 0 mm: too short to start a gait |
| stopping from walking | coasts ~4 mm back, 7–11 mm sideways |

Same picture with `--no-bam`, so it's the policy, not the servo model.

### Facts worth knowing (checked in Pollen's code, `microduck_rl` @ b02fdea, 2026-10-12)

- **Policies are 50 Hz, 61-D obs → 14 joint targets.** Obs = 3 gyro + 3 projected gravity +
  14 joint pos (rel. default pose) + 14 joint vel + 14 last action + 13 command.
  Command = `[vx, vy, wz, neck_pitch, head_pitch, head_yaw, head_roll, body x, y, z, roll, pitch, yaw]`.
- **Walking command ranges in training:** vx ±0.4 m/s, vy ±0.3 m/s, wz ±1.0 rad/s. Stay inside them.
- `velstand` = walk + stand at zero command + protective fall + get up, all in one policy.
  We use it as "the legs" so falls recover without any extra logic.
- **The kick policy is blind** (no ball in its obs). It was trained from a still stand with the
  ball centre at (x=0.09, y=±0.042) m in the robot's yaw frame (+y = left; right-foot kick
  ⇒ y = −0.042), with ±1.5 cm noise. It kicks straight ahead, aiming for about 1 m/s.
  Pollen's own `infer_policy.py` *teleports* the ball to the exact spot before every kick;
  `common/sim.py:Duck.start_kick` deliberately doesn't.
- **Ball:** 70 mm / 15 g. Pollen's ball sets a rolling friction but keeps MuJoCo's default
  `condim=3`, which ignores it: a kicked ball never slows down (24 m in 40 s at 0.6 m/s).
  `common/sim.py:set_ball_roll` turns on `condim=6` and picks the coefficient from a measured
  table so a typical kick rolls `--ball-roll` metres (default 1). Kicks leave at ~0.5–0.7 m/s.
- **Servos:** BAM M6 XL330 model (voltage control, load-dependent friction). `--no-bam` swaps
  in ideal PD servos for comparison. Pollen's docs: actuator fidelity is most of the sim2real gap.
- **CPU speed** on Jack's laptop (one env, BAM on): ~6–7× real time headless (several runs at
  once still fine), ~0.8× with `--video`, ~1× in the live viewer.
- Pollen's `AGENTS.md` (in `third_party/microduck_rl`) is required reading before training
  anything *inside* their stack. It has hard-won rules on reward sign conventions, jackpots,
  curricula and obs invariants.

---

## 2. Next: watch it live, then tune

Both games run with the real policies (§7). Watch them in the viewer:
`uv run kick/kick_loop.py` and `uv run fetch/chase_ball.py` (Mac: `uv run mjpython ...`).

**Where the kick loop's time goes** (300 s runs): APPROACH ~40%, ALIGN ~35–45%, KICK ~12%,
SETTLE ~4%, WATCH ~3%. APPROACH is mostly walking at the legs' ~0.1 m/s, already at the
trained maximum command, so the levers are in ALIGN:
- **Retries** (~2–3 per 5 min): stopping 12–15+ mm off the spot, or overshooting. Each costs a
  walk round (~10 s). Ideas: stop a few mm earlier (`stop_lead`) and creep in with one more
  short walk; or allow a slightly bigger tolerance and check `alignment.png` / `aiming.png`
  for whether 15–20 mm kicks still go straight.
- **Turn-in-place** is ~10% of all time (inside ALIGN). Tried and rejected: steering at a
  "carrot" on the aim line during APPROACH so the duck arrives facing along it: 14.0 vs 15.7
  kicks / 5 min (the longer path costs more than the turning saved).
- **Aim spread** (3 seeds x 300 s): ±30° 13.3, **±60° 15.7**, ±90° 12.7 kicks. ±60° stays.
- **Accidental knocks** (~11 per 5 min, as many as kicks). Two kinds: (a) harmless — after
  WATCH says "stopped" (5 cm/s) the ball creeps another 3–4.5 cm and trips the 5 cm "knocked"
  check, which only re-picks the aim; (b) real — after a retry the duck is standing at the
  kick stance and turns round on the spot to walk back, and a turning foot sweeps the ball
  (any in-place turn > ~30° there touches it). Tried and rejected: kick at up to 20 mm
  instead of retrying (16.0 vs 15.7 kicks, whiffs 0.7 -> 1.7); "pivot" a ≤15° turn on the spot
  to swing the kick spot onto the ball and kick that way (6 seeds: 15.7 vs 15.8 kicks; fewer
  retries but no faster). Also tried, no better (6 seeds each): ignore the creep (knocks
  11.7 -> 4.5, kicks 15.8 -> 14.7) and pick the aim only once within 0.6 m of the ball (14.8).
  Untried: after a retry, first walk forward-diagonally away from the ball before turning round.
- **Conclusion so far:** ~15 kicks / 5 min looks set by the legs (walking ~0.1 m/s, turning
  bursts). Differences under ~1.5 kicks are noise at 6 seeds: each code change shifts the
  random aim sequence, so runs aren't paired. Bigger wins probably need better legs (a faster
  walking policy) or a different game (shorter rolls: `--ball-roll 0.5`).
- **Orbit** (replaced the single go-round waypoint): ~5 orbits per 5 min, all 126–169°
  reversals; kicks 15.5 vs 15.8 (noise). No real knock happens while walking or orbiting:
  13 of 16 are the turn on the spot right after a retry, when the duck stands at the kick
  stance 9 cm from the ball. Geometry (foot box vs ball): turning away from the ball side
  clips it within ~30°, towards it the other foot clips it at ~60–90°, walking forward
  kicks it. With no sidestep or backward walk there is no clean way out; the fix would be
  fewer retries.
- **Chaser:** ~10 s per ball, limited by the legs. Check `timeseries.png` for turn lag.

`runs/` sweeps: `--seed` makes runs repeatable; compare several seeds, single runs are noisy
(±3 kicks per 5 min).

---

## 3. Fetch roadmap

### Stage 1: learn the brain (replace `ChaseBrain` with a trained policy)

**Goal:** a small network maps (ball position in the duck's frame) → (vx, vy, wz), trained
with RL, with Pollen's walking policy frozen as the legs. Success means it matches or beats the
hand-coded chaser on balls/minute with the same or fewer falls. It's also fun to watch it
learn: start spinning randomly, end beelining.

**Design:**
- **Observation (small, ~6–8 numbers):** ball (x, y) in the duck frame, distance, bearing as
  sin/cos, the last command (3), optionally the duck's forward speed and yaw rate. Normalise.
- **Action:** 3 numbers squashed to the training ranges (vx ∈ [−0.1, 0.35], vy ∈ ±0.2, wz ∈ ±1.0).
- **Decision rate:** run the brain at 10 Hz (hold each command for 5 walking-policy steps).
  That makes the effective horizon 5× shorter and learning much faster. The legs still run at 50 Hz.
- **Reward:** potential-based progress, `r = (d_prev − d_now) / dt` scaled. Add a catch bonus
  (+10, then respawn the ball, keep the episode going), a fall penalty (−5, end the episode),
  and a small action-change penalty for smoothness. Per Pollen's AGENTS.md: avoid jackpots and
  never pay a reward for sitting in a state.
- **Episodes:** 20–30 s, ball respawned at 0.5–1.5 m on catch, random duck start heading.

**Where to train:** three options, cheapest first.
1. **Surrogate first (recommended):** fit a simple model of how the real legs respond to
   commands, using `log.csv` from Stage 0 runs (commanded vs achieved vx, wz: lag + gain +
   noise, e.g. a first-order filter per axis). Train PPO on that in pure numpy or torch.
   That's thousands of environments on a laptop CPU in minutes. Then evaluate and fine-tune in
   real MuJoCo. `tests/legs.py:lazy_legs` and `tests/test_kick_brain.py:IdealWorld` are a
   starting point for the surrogate.
2. **Real MuJoCo on CPU:** wrap `common.sim.Duck` as a Gymnasium env; run 4–8 copies with
   `gymnasium.vector.AsyncVectorEnv`; use stable-baselines3 PPO. Fine for fine-tuning, slow
   for learning from scratch (single-env ~18× real time).
3. **Pollen's GPU stack (mjlab / MuJoCo Warp):** a new task in `microduck_rl` with the walking
   policy frozen inside the env. Fastest at scale but needs CUDA (or `--hf-jobs`, paid HF Jobs)
   and following all of AGENTS.md. Overkill for this.

**Steps:**
1. `fetch/env.py`: a Gymnasium env around `Duck` (reset / step at 10 Hz / reward / done).
2. `fetch/surrogate.py`: fit the leg-response model from Stage 0 logs; a surrogate env with the same interface.
3. `fetch/train_brain.py`: SB3 PPO, logs to TensorBoard, saves the brain as `.zip` + ONNX.
4. `fetch/chase_ball.py --brain learned.onnx`: a `Brain` (common/brain.py) that runs the ONNX
   in `update(snapshot)`, passed to `Chase` instead of `ChaseBrain`.
5. **Visualise learning:** reward and catches-per-episode curves; a video every N updates
   (headless runner already makes videos); top-down paths of early vs late policy side by side.

**Deps to add:** `gymnasium`, `stable-baselines3`, `torch` (CPU), `tensorboard`.

### Stage 2: grab the ball with the beak
- First check physics: can the beak hold a 70 mm / 15 g ball at all in sim? Find the jaw
  joint and geoms in `robot_groundcontact.xml`. Try scripted poses: crouch (the GroundPick
  pose), open, close on the ball, lift. If it can't hold this ball, use a smaller or lighter
  one (add a second ball body) or a "beanbag".
- Pollen has `Mjlab-GroundPick-*` (crouch, touch the ground with the mouth tip, return)
  and a `pickup/` module and scripts (`pickup_train.py`, `pickup_demo.py`) in microduck_rl.
  **Read those first.** There may already be a grasping pipeline to reuse.
- If training is needed: an episodic task in Pollen's stack, built from the GroundPick
  template, with ball-lifted reward. Follow AGENTS.md. Needs GPU.

### Stage 3: carry it back
- Walking with something in the beak changes the head load. Check whether velstand copes
  (head mass is 38% of the body). It may need a policy trained with a held object, or just slower commands.
- The brain is the same `steer_to`, with "home" as the target instead of the ball.

### Stage 4: glue it together
A state machine like `kick/brain.py`: FIND → APPROACH → GRAB → RETURN → DROP → repeat.
A `Brain` subclass (common/brain.py) in a `Game` (common/game.py), with an ideal-world test.

### Stage 5: perception (towards the real robot)
- Use the duck's simulated camera (`microduck_rl/src/mjlab_microduck/sim/camera.py`) to
  render what it sees; detect the orange ball by colour threshold → bearing and rough distance
  from blob size → feed the same brain.
- **Visualise:** picture-in-picture of the duck's view with the detection box.
- Then on the real robot: the same detector on real frames, via Pollen's `microduck` runtime.
  Read that repo's docs on how to run custom code and policies (`robotctl policy add`, the
  manifest schema).

---

## 4. Kick loop: ideas to extend it
- **Learned aiming:** kick from wherever it is and learn where the ball goes (a kick-outcome
  model from `events.json`: ball position relative to the foot -> direction and distance),
  instead of always lining up exactly behind the ball. `aiming.png` is the first look at this.
- **Targets:** bring back goals, or cones to hit, as a different aim picker (`pick_aim` is
  the only place aims come from).
- **Dribbling:** small pushes with the walking policy instead of full kicks.
- **Goalkeeper:** a second duck (add another robot body to the scene; heavy but doable) or a
  scripted moving "wall".

## 5. Other fun ideas (unordered, from the original brainstorm)
- **Duckling imprinting:** follow a moving target (mouse-dragged in the viewer, later a person
  via the camera) and get "upset" (head droop) when left behind.
- **Dance:** head/body pose commands driven by beat detection from an audio file.
- **Parkour course** of steps and ramps; Pollen's rough-terrain and slope tasks are a start.
- **Silly gaits:** reward-hacked styles (moonwalk, strut, hop). Needs training in Pollen's stack.
- **Sumo** self-play in sim with two ducks.
- **Guard duck:** patrol, spot a coloured "intruder", turn to face it and quack (speaker on the real robot).

---

## 6. Dev notes (setup, gotchas, conventions)

**Layout**
```
common/   sim.py (setup, Duck, set_ball_roll, run_live/run_headless, save_outputs),
          game.py (shared loop), brain.py (Snapshot, Action, Brain), steer.py, overlay.py
fetch/    chase_ball.py (game rules), chaser.py (ChaseBrain), plots.py
kick/     kick_loop.py (game), brain.py (KickBrain state machine), kick_plots.py
docs/     how_it_works.md (diagrams: 50 Hz loop, both brains, ours vs Pollen's)
tests/    test_chaser.py, test_kick_brain.py, legs.py  (imaginary legs, no MuJoCo, ~10 s)
third_party/microduck_rl   Pollen's repo, cloned by the user, gitignored
runs/     headless outputs, gitignored
```

**A game** subclasses `common.game.Game` (falls, stepping, logging, trails, live/headless
come for free) and adds a brain, `start()`, and optionally `rules()`, `draw()`, `status()`,
`log_row()`, `summary()`, `plots()`. See docs/how_it_works.md "Adding things".

**Brains** take a `Snapshot` and return an `Action` (common/brain.py), with no MuJoCo inside,
and are tested against imaginary legs first (perfect, and `lazy_legs`). If it can't work with
those, it won't with real ones.

**Running**
- `uv sync` once; `uv run <script>`; `uv run pytest`.
- Python 3.12 only: BAM requires `<3.13`. MuJoCo is pinned to 3.10.0 and BAM to the commit
  microduck_rl pins (`62bd8ce`). Newer BAM changed the `MujocoController` signature.

**Windows gotchas (Jack's laptop)**
- No `mjpython` on Windows: plain `uv run`.
- If Anaconda's `(base)` env is active, uv can build `.venv` on Anaconda's Python, which
  breaks MuJoCo's DLL load (`WinError 1114`). Fix: `conda deactivate`, delete `.venv`,
  `uv sync --python-preference only-managed`.
- Pollen's `infer_policy.py` imports `termios`/`tty` (Unix-only). `common/sim.py`
  stubs them on Windows; they're only used by its keyboard reader.

**Headless video on Linux** needs EGL: `sim.setup_gl_for_headless()` sets `MUJOCO_GL=egl`,
and it must run before anything imports mujoco.

---

## 7. Log
Add dated entries: what was tried, numbers, what was learned.

- **2026-10-23:** Built fetch Stage 0 and the kick loop. Ideal-duck tests: the chaser catches
  balls placed in all 12 test directions; the kick brain scores 9 goals from 10 kicks in
  150 s (with ±0.15 rad kick noise) and walks round the ball when on the wrong side. One bug
  found and fixed through the plots: goals scored mid-kick weren't counted. Not yet run with
  real policies.
- **2026-10-23 (later):** First headless runs with the real Hub policies on Jack's laptop
  (BAM on, ~5–6× real time). Before: fetch 4 balls / 90 s (~18 s each, duck stalled near the
  ball); kick loop **0 kicks** in 180 s (stalled on approach, never reached ALIGN). Cause:
  velstand's dead zone (table in §1). Fixes: chaser `min_forward` (every forward command
  ≥ 0.3, default max raised to 0.4); kick ALIGN redesigned from holonomic shuffling to
  turn-in-place + car-style walk-in along the ball→goal line (25 cm run-in, overshoot or
  sideways miss = retry); `--assist` now actually triggers (it used to need the duck to be
  lined up already); `lazy_legs` ideal-world tests; `uv run pytest` no longer collects
  Pollen's tests. After: fetch **9 balls / 90 s**, 0 falls; kick **3 goals / 5 kicks / 240 s**,
  0 whiffs, mean line-up 8.6 mm; with `--assist` 3 goals / 7 kicks (3 assisted).
- **2026-10-23 (evening):** Kick loop rebuilt as endless kicking: random aim each kick
  (duck->ball ±60°, back home ±45° outside a 1.5 m soft arena), goals removed, each kick scored
  on aimed vs actual direction. Found the ball never slows down in MuJoCo (rolling friction
  off at condim 3): 1 kick in 240 s, the duck walking after a 20+ m roll. Added
  `--ball-roll` (calibrated table, default 1 m). Steering now judges sideways error in the
  world (heading error had leaked in and ended walk-ins ~10 mm off). Tolerance 12 -> 15 mm
  (the kick's training noise) and "lined up" must hold 0.3 s (a walk-in's first step lunges
  the trunk: 10 mm that sways back to 17). Shared game loop / brain interface / steering
  moved into `common/`, diagrams in `docs/`. Run-in sweep (3 seeds x 300 s): 0.15 / 0.20 /
  0.25 m -> 14.0 / **15.7** / 13.0 kicks. Now: ~15 kicks / 5 min, 0 falls, ~10° direction
  error, ~0.8 m per kick. Live viewer checked for both games.
- **2026-10-24:** Orbit approach (circle the ball 10 cm outside the pre-stance point when the
  straight way is blocked; leave it within 30° of the aim line, so the duck arrives facing
  along the aim). First version aimed at an "entry" waypoint 12 cm behind the pre-stance
  point: the duck circled it, since its tightest walking turn is wider than that; dropped.
  Kickback added (back the way the last kick went; "far end" kept kicking the same way when
  the ball stopped mid-line). Brains publish `plan` (their intended route) for the viewer.

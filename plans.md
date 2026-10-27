# Plans

Roadmap and hand-over notes for this repo. Written so that Jack, or an agent
starting cold, can pick up any item without re-deriving the context. Keep it
current: when something is done, tick it, add what was learned, and move on.

Purpose of the project: **fun**. Simulation games and tricks for a Microduck
(Pollen Robotics / Hugging Face 25 cm biped, arriving early 2027), built on top
of Pollen's open-source stack. Not research, not a CV piece. Prefer things that
are quick to try and good to watch.

---

## 1. Where things stand (2026-10-24)

| Thing | Status |
|---|---|
| `chase/`: hand-coded ball chaser | Works with the real velstand policy: ~6 balls / min (~9 s each), 0 falls |
| `kickback/`: back and forth along a 1 m line | Works with the real policies: ~12 kicks / 5 min, strictly alternating, an orbit before every kick, ~8° direction error, ball ends ~0.1–0.2 m off the line |
| `common/`: loop, brain interface, duck + skills, runners, steering, toy world | Matches Pollen's `infer_policy.py` main loop. `docs/how_it_works.md` has diagrams |
| Runs on Jack's Windows laptop | Headless and live viewer both checked |

The random-aim "kick loop" (soft arena, centre bias, assist) was removed on
2026-10-24: too many interacting parameters for what it showed. Kickback keeps
its brain. Its lessons are in §2 and the log.

Hub repo and file names are confirmed (`pollen-robotics/microduck-policies`:
`velstand.onnx`, `ball_kick_{left,right}.onnx`).

### The legs (velstand v8) in MuJoCo, measured 2026-10-23 with BAM

The brains are written around these numbers; `common/toy.py:lazy_legs` models them.

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

Same picture with ideal PD servos instead of BAM, so it's the policy, not the servo model.

### Facts worth knowing (checked in Pollen's code, `microduck_rl` @ b02fdea, 2026-10-12)

- **Policies are 50 Hz, 61-D obs → 14 joint targets.** Obs = 3 gyro + 3 projected gravity +
  14 joint pos (rel. default pose) + 14 joint vel + 14 last action + 13 command.
  Command = `[vx, vy, wz, neck_pitch, head_pitch, head_yaw, head_roll, body x, y, z, roll, pitch, yaw]`.
- **Walking command ranges in training:** vx ±0.4 m/s, vy ±0.3 m/s, wz ±1.0 rad/s. Stay inside them.
- `velstand` = walk + stand at zero command + protective fall + get up, all in one policy.
  We use it as "the legs" so falls recover without any extra logic.
- **Skills** (episodic policies: kicks, roulade, anything published with `--kind episodic`)
  share the 61-D obs with an all-zero command. Pollen's `PolicyInference` keeps them in
  `behavior_sessions` / `behavior_durations` and swaps them in by name;
  `common/duck.py:load_duck` fills those dicts for any skill it's given.
- **The kick policy is blind** (no ball in its obs). It was trained from a still stand with the
  ball centre at (x=0.09, y=±0.042) m in the robot's yaw frame (+y = left; right-foot kick
  ⇒ y = −0.042), with ±1.5 cm noise. It kicks straight ahead, aiming for about 1 m/s.
  Pollen's own `infer_policy.py` *teleports* the ball to the exact spot before every kick;
  `common/duck.py:Duck.start_skill` deliberately doesn't.
- **Ball:** 70 mm / 15 g. Pollen's ball sets a rolling friction but keeps MuJoCo's default
  `condim=3`, which ignores it: a kicked ball never slows down (24 m in 40 s at 0.6 m/s).
  `common/duck.py:set_ball_roll` turns on `condim=6` and picks the coefficient from a measured
  table so a typical kick rolls `--ball-roll` metres (default 1). Kicks leave at ~0.5–0.7 m/s.
- **Servos:** BAM M6 XL330 model (voltage control, load-dependent friction), always on.
  Pollen's docs: actuator fidelity is most of the sim2real gap.
- **CPU speed** on Jack's laptop (one env, BAM on): ~6–7× real time headless (several runs at
  once still fine), ~0.8× with `--video`, ~1× in the live viewer.
- Pollen's `AGENTS.md` (in `third_party/microduck_rl`) is required reading before training
  anything *inside* their stack. It has hard-won rules on reward sign conventions, jackpots,
  curricula and obs invariants.

---

## 2. What we learned lining up kicks (from the kick loop and kickback, 2026-10-23/24)

Kick-loop numbers (random aims, mostly onwards, so less walking round the ball than
kickback's ~12 kicks / 5 min) unless marked otherwise.

- **Time goes to walking.** APPROACH ~40%, ALIGN ~35–45%, KICK ~12%. APPROACH is walking at
  the legs' ~0.1 m/s, already at the trained maximum command. Kicks / minute looks set by the
  legs; bigger wins need faster legs or shorter rolls (`--ball-roll 0.5`).
- **Run-in** (`standoff`): 0.15 / **0.20** / 0.25 m → 14.0 / 15.7 / 13.0 kicks per 5 min.
- **Retries** (stopping 12–15+ mm off the spot, or overshooting) cost ~10 s each. Tried and
  rejected: kicking at up to 20 mm (more whiffs), a "carrot" on the aim line during APPROACH
  (longer path), pivoting the last ≤15° on the spot (no faster).
- **Accidental knocks** are almost all the turn on the spot right after a retry, standing 9 cm
  from the ball: a turning foot sweeps it. With no sidestep or backward walk there is no
  clean way out; the fix is fewer retries.
- **Orbit** beats a single go-round waypoint (the duck circled a waypoint closer than its
  tightest turn). It never knocks the ball while orbiting.
- Single runs are noisy (±3 kicks / 5 min); compare several, and remember any code change
  that shifts timing changes the whole run.

---

## 3. Chase roadmap: towards fetch

### Stage 1: learn the brain — done (2026-10-24)

`chase/train.py` trains a small network (PPO, stable-baselines3) in the toy world with lazy
legs; `LearnedBrain` in `chase/brain.py` runs the exported ONNX; `chase/game.py --brain`.

- **Sees** (`observe`): ball x, y in the duck frame, distance, bearing as sin/cos, last command.
- **Does** (`to_command`): forward 0–0.4 m/s and turn ±1 rad/s (no vy: the legs can't
  sidestep), decided at 10 Hz and held for 5 ticks.
- **Reward:** 10 per metre of progress + 10 per catch (then a new ball), 30 s episodes.
- **Result:** in the toy world it matches the hand-written brain within ~1 min of training
  (5.5 balls/min both: the toy legs' top speed is the ceiling). **In MuJoCo it beats it:**
  25 and 27 balls / 180 s vs 18 and 19 (seeds 0, 1), 0 falls, ~6.8 s vs ~9.5 s per ball.
  Why: it turns much harder while walking (mean |wz| 0.56 vs 0.24 rad/s) and rarely stops to
  turn (5% vs 9% of the time), and the real velstand walks faster that way (0.148 vs
  0.102 m/s achieved). The toy legs don't model that, so the toy world under-rated it.

Retrained 2026-10-27 for the new report layout (MuJoCo clips per checkpoint): 6.4 vs 5.5
balls/min in the toy world, 9 vs 6 balls in 60 MuJoCo seconds.

Next, if wanted: make `lazy_legs` speed depend on turning (fit from `log.csv` of real runs)
so the toy world ranks brains like MuJoCo does; fine-tune in MuJoCo (a Gymnasium env around
`common.game.Game` + `Duck`, a few CPU copies). The same recipe for kickback's line-up: done,
Stage 1b.

### Stage 1b: learn kickback's line-up (2026-10-26)

`kickback/train.py` trains a network to replace APPROACH + ALIGN (`LearnedKickbackBrain`);
the hand-written SETTLE / KICK / WATCH stay. `kickback/game.py --brain`. One episode = one
line-up attempt (random ball, aim, duck pose), +10 lined up, −10 for moving the ball 1 cm,
+10/m towards the stance, +1/rad towards the aim, −0.1/s.

- **Toy world first needed ball pushing** (`common/toy.py: push_ball`): the duck walked
  through the ball. Feet = a rectangle measured from Pollen's model (x −3.3…+3.5 cm,
  y ±6.3 cm); a ball on the kick spot is 2 cm clear of the toes; turning on the spot next
  to it sweeps it (the real knock cause, §2).
- **Curriculum was essential:** without it ~2% of attempts lined up and it didn't learn. The
  "lined up" window starts 4x bigger and shrinks to 15 mm / 8° by 70% of training.
- **16 environments in subprocesses + torch 1 thread:** ~2000 decisions/s, 2M in ~17 min.
- **What went wrong, in order:**
  1. Run 1 (lined up = inside 12 mm, 0.8 x tolerance): beat the hand-written brain in the
     toy world (2.7 vs 2.1 kicks/min) but in MuJoCo froze for minutes 13 mm off: small
     turns are inside velstand's dead zone, so nothing changed and it never moved again.
  2. Run 2 (−10 for running out of time, to make freezing costly): learned far slower and
     never got good (45% lined up at 1.4M). Dropped.
  3. Run 1's network with "lined up = inside 15 mm": 15 kicks / 5 min vs 12, 1 knock vs 4,
     0 stuck, aim error 6.4° vs 10.3° (MuJoCo, one run each). So the 12 mm margin was the trap.
  4. Run 3 (trained with 15 mm): 10 kicks / 5 min, 3 knocks, stuck twice: after a near
     miss next to the ball it still freezes rather than walk off and come round. Training
     with no margin leaves no room for the real duck's lunge and sway.
  5. Run 4: toy duck lunged/wobbled when it stopped, so near misses happen in training, and
     discount 0.995 (~20 s look-ahead) so a walk-off pays. Worse everywhere: 0.5 kicks/min in
     the toy world, 0 kicks and 6 times stuck in 5 MuJoCo minutes. Dropped.
  6. **Final:** run 1's training (margin 0.8, discount 0.99) and the brain plays to the full
     tolerance (`LearnedKickbackBrain(margin=1.0)`, `TRAIN_MARGIN = 0.8`). 2M decisions,
     ~21 min. Toy world 3.0 vs 2.1 kicks/min. **MuJoCo, 5 min: 15 kicks vs 12, 1 knock vs 4,
     0 stuck, 0 falls, aim error 6.4° vs 10.3°, ball ends 0.13 vs 0.18 m off the line.**
     `runs/kickback_learned/report.html`. In the 90 s clips, checkpoints before 1.6M mostly
     get stuck once (a 45 s timeout); 1.6M on kick 5-6 times in 90 s.
- **Measurement bugs found on the way:** (a) "knocks" counted the ball still creeping at
  1–2 cm/s after WATCH called it stopped, with the duck 0.4–1 m away; now only within 0.3 m
  counts (`crept` for the rest). (b) Filming many games on one `Duck` drifted: late
  checkpoints froze in the report but not on their own. `reset` now calls `mj_resetData` and
  stops skills, which helped but wasn't enough (some of Pollen's runner state carries over),
  so the trainers load a fresh `Duck` per game.
- **Still open:** it can still freeze after a near miss (the walk-off-and-come-round that
  the hand-written ALIGN does as a "retry" was never learned). Ideas: a stuck detector that
  hands one attempt to the hand-written approach; fine-tuning in MuJoCo, where the near
  misses actually happen.
- One 5-minute MuJoCo run per brain is noisy (§2: ±3 kicks); the game always starts the same.

### Stage 2: grab the ball with the beak
- First check physics: can the beak hold a 70 mm / 15 g ball at all in sim? Find the jaw
  joint and geoms in `robot_groundcontact.xml`. Try scripted poses: crouch (the GroundPick
  pose), open, close on the ball, lift. If it can't hold this ball, use a smaller or lighter one.
- Pollen has `Mjlab-GroundPick-*` and a `pickup/` module and scripts in microduck_rl.
  **Read those first.** A trained grab becomes a skill (`common/duck.py`).

### Stage 3: carry it back
- Walking with something in the beak changes the head load (head mass is 38% of the body).
  It may need a policy trained with a held object, or just slower commands.
- The brain is the same `steer_to`, with "home" as the target instead of the ball.

### Stage 4: glue it together
A state machine like `kickback/brain.py`: FIND → APPROACH → GRAB → RETURN → DROP → repeat.

### Stage 5: perception (towards the real robot)
- Use the duck's simulated camera (`microduck_rl/src/mjlab_microduck/sim/camera.py`); detect
  the orange ball by colour threshold → bearing and rough distance → feed the same brain.
- Then on the real robot: the same detector on real frames, via Pollen's `microduck` runtime
  (`robotctl policy add`, the manifest schema).

---

## 4. Other fun ideas (unordered)
- **Kickback extensions:** targets or cones to hit (a different `pick_aim`); dribbling with
  small walking pushes instead of kicks; a kick-outcome model learned from `events.json`
  (ball position relative to the foot → direction and distance) so it can kick from wherever it is.
- **Duckling imprinting:** follow a moving target (mouse-dragged in the viewer, later a person
  via the camera) and get "upset" (head droop) when left behind.
- **Dance:** head/body pose commands driven by beat detection from an audio file.
- **Parkour course** of steps and ramps; Pollen's rough-terrain and slope tasks are a start.
- **Silly gaits / tricks:** trained in Pollen's stack, then loaded here as skills.
- **Sumo** self-play in sim with two ducks.

---

## 5. Dev notes (setup, gotchas, conventions)

**Layout:** see the README. One folder per behaviour (`game.py` = rules + the script,
`brain.py`, `plots.py`); everything shared in `common/`.

**Brains** take a `Snapshot` and return an `Action` (common/brain.py), with no MuJoCo inside,
and are tested in `common/toy.py` first (perfect and `lazy` legs). If it can't work with
those, it won't with real ones.

**Running**
- `uv sync` once; `uv run <behaviour>/game.py`; `uv run pytest`.
- Imports are package imports from the repo root (`from kickback.brain import ...`): each
  script adds the root to `sys.path` once; pytest gets it from `pythonpath` in pyproject.toml.
- microduck_rl is pinned to `b02fdea` (README): `common/duck.py` uses PolicyInference
  internals (`_update_command`, `_end_behavior`, `behavior_*`). Re-check them before moving it.
- Python 3.12 only: BAM requires `<3.13`. MuJoCo is pinned to 3.10.0 and BAM to the commit
  microduck_rl pins (`62bd8ce`). Newer BAM changed the `MujocoController` signature.

**Windows gotchas (Jack's laptop)**
- No `mjpython` on Windows: plain `uv run`.
- If Anaconda's `(base)` env is active, uv can build `.venv` on Anaconda's Python, which
  breaks MuJoCo's DLL load (`WinError 1114`). Fix: `conda deactivate`, delete `.venv`,
  `uv sync --python-preference only-managed`.
- Pollen's `infer_policy.py` imports `termios`/`tty` (Unix-only). `common/duck.py`
  stubs them on Windows; they're only used by its keyboard reader.

**Headless video on Linux** needs EGL: `common/__init__.py` sets `MUJOCO_GL=egl` for
`--headless` runs, before anything in `common` imports mujoco.

---

## 6. Log
Add dated entries: what was tried, numbers, what was learned.

- **2026-10-23:** Built the chaser and a kick game. First headless runs with the real Hub
  policies: chaser stalled near the ball, kicks never happened. Cause: velstand's dead zone
  (§1 table). Fixes: every forward command ≥ 0.3; kick line-up redesigned from shuffling to
  turn-in-place + car-style walk-in; toy-world tests with `lazy_legs`. After: 9 balls / 90 s.
  Found the ball never slows down in MuJoCo; added `--ball-roll`. Tolerance 15 mm (the kick's
  training noise), "lined up" must hold 0.3 s (a walk-in's first step lunges the trunk).
- **2026-10-24:** Orbit approach; kickback added. Repo simplified: package imports, one
  mujoco import, `--no-bam` and old-policy support removed. Then the random-aim kick loop
  removed; layout became one folder per behaviour (`chase/`, `kickback/`); `sim.py` split
  into `duck.py` (with generic named skills) and `run.py`; the tests' imaginary duck promoted
  to `common/toy.py`. Same-seed MuJoCo runs of both games identical before and after.
  First learned brain: chase PPO in the toy world (~6 min CPU), ~40% more balls than the
  hand-written chaser in MuJoCo (§3 Stage 1).
- **2026-10-26/27:** Learned kickback line-up (§3 Stage 1b): toy feet push the ball,
  curriculum on the "lined up" window, trained to 12 mm and played to 15 mm. 5 MuJoCo
  minutes: 15 kicks vs 12, 1 knock vs 4. Every training report now has four parts (aim,
  training, results, the same checkpoints in MuJoCo); chase retrained for it. Found and
  fixed: "knocks" counted a creeping ball; many games on one `Duck` drifted.

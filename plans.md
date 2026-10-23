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
| `fetch/chase_ball.py`: Fetch Stage 0, hand-coded ball chaser | Built. Pipeline tested in MuJoCo with a **stand-in** policy only |
| `kick/kick_loop.py`: duck plays football against itself | Built. Brain tested with an ideal duck (9 goals from 10 kicks); MuJoCo pipeline tested with stand-in policies only |
| `common/sim.py`: shared setup, Duck wrapper, live/headless runners | Built |
| Runs on Jack's Windows laptop | Setup done; first live run still pending (see §6) |

**Most important caveat:** nothing has yet been run with Pollen's *real*
trained policies. The dev sandbox couldn't reach Hugging Face. So the
following are all unverified:
- whether the Hub repo and file names are right (`pollen-robotics/microduck-policies`, `velstand.onnx`, `ball_kick_{left,right}.onnx`)
- how well the walking policy tracks the chaser's commands
- whether the kick line-up converges

The first job for whoever picks this up is to run both games with real
policies and write down what happens (§2).

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
- **Ball:** 70 mm / 15 g, very low rolling friction, so kicked balls roll a long way.
- **Servos:** BAM M6 XL330 model (voltage control, load-dependent friction). `--no-bam` swaps
  in ideal PD servos for comparison. Pollen's docs: actuator fidelity is most of the sim2real gap.
- **CPU speed** (2-core sandbox, one env, BAM on): ~18× real time headless, ~40× with `--no-bam`.
  Video rendering makes it much slower (~0.1× in the sandbox). Jack's laptop will differ.
- Pollen's `AGENTS.md` (in `third_party/microduck_rl`) is required reading before training
  anything *inside* their stack. It has hard-won rules on reward sign conventions, jackpots,
  curricula and obs invariants.

---

## 2. Next: run both games for real

1. `git pull`, then `uv run fetch/chase_ball.py` (on Windows; Mac needs `uv run mjpython ...`).
2. If the policy download fails, find the real file names on the Hub page (or Seeed's
   official-policies wiki page) and fix `common/sim.py:walking_policy` and the
   `hub_policy(...)` call in `kick/kick_loop.py`. Fallback: `--policy`, `--kick-left`,
   `--kick-right` with local paths.
3. Headless runs, then look at the PNGs:
   - `uv run fetch/chase_ball.py --headless --duration 120`
   - `uv run kick/kick_loop.py --headless --duration 240`
   - then the same with `--assist`, and the same with `--no-bam`
4. Record results in §7 (log): balls/min, falls, goals, kicks, line-up error, whiffs.

**Likely tuning needed** (guesses to check):
- **Chaser:** turn gain may be too high (wobbly) or too low (wide arcs). Check the
  commanded-vs-achieved panel in `timeseries.png` for lag and saturation.
- **Kick line-up:** velstand may not move at all for tiny commands (a dead zone).
  `KickParams.align_min_v` (0.025 m/s) is there for this; raise it if ALIGN stalls.
  If the duck overshoots, lower `align_gain` or `align_max_v`.
- **Tolerance:** check whether the kick actually needs 12 mm. `alignment.png`'s right panel
  (error vs ball speed) answers this. If kicks at 20 mm still fly, loosen `--tolerance`.
- **Approach:** the duck may bump the ball while circling round it. Widen `go_round`.

---

## 3. Fetch roadmap

### Stage 1: learn the brain (replace `fetch/chaser.py` with a trained policy)

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
   real MuJoCo. `tests/test_kick_brain.py:IdealWorld` is a starting point for the surrogate.
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
4. `fetch/chase_ball.py --brain learned.onnx`: drop-in replacement for `chase_command`.
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
- The brain is the same chaser, with "home" as the target instead of the ball.

### Stage 4: glue it together
A state machine like `kick/brain.py`: FIND → APPROACH → GRAB → RETURN → DROP → repeat.
Reuse the `Snapshot` / `Action` pattern and an ideal-world test.

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
  model from `events.json`), instead of always lining up perfectly behind the ball.
- **Dribbling:** small pushes with the walking policy instead of full kicks, for close-range goals.
- **Goalkeeper:** a second duck (add another robot body to the scene; heavy but doable) or a
  scripted moving "wall" the striker has to beat.
- **Penalty shoot-out mode:** fixed spot, keep score of accuracy and power.

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
common/   sim.py (setup, Duck, run_live/run_headless, save_outputs), overlay.py (3D markers)
fetch/    chase_ball.py (game), chaser.py (brain), plots.py
kick/     kick_loop.py (game), brain.py (state machine), kick_plots.py
tests/    test_chaser.py, test_kick_brain.py  (ideal-duck tests, no MuJoCo, <1 s)
third_party/microduck_rl   Pollen's repo, cloned by the user, gitignored
runs/     headless outputs, gitignored
```

**A "game"** implements `t`, `tick()`, `draw(scn)`, `status()`, `camera`, `log`,
`events`, `summary()`, `plots(out_dir)` (see the `common/sim.py` docstring). New games
should follow that and get live viewing and headless outputs for free.

**Brains** should be pure functions or classes with no MuJoCo inside, tested against an
ideal world first (see `tests/`). If it can't work with perfect legs, it won't with real ones.

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

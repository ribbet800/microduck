# How it works

Two systems, one loop: **chase**, and **kicking** (the endless kick loop, plus
its kickback variant). Pollen's pre-trained policies are the **legs** and the
**boot**; everything we wrote is the **brain** that decides what to send them,
plus the game rules, the 3D markers and the plots.

Colours in every diagram: **blue = ours** (this repo), **orange = pre-installed**
(Pollen's `microduck_rl` and their policies on the Hugging Face Hub).

## The 50 Hz loop

Every 20 ms, for both games (`common/game.py: Game.tick`):

```mermaid
flowchart TD
    W["MuJoCo world<br/>duck + ball"]
    G["Game loop<br/>fallen? (trunk below 7.5 cm,<br/>back up above 10.5 cm)"]
    S[/"Snapshot<br/>t, duck xy + heading, ball xy + speed,<br/>fallen, kicking"/]
    B["Brain.update()<br/>state machine"]
    A[/"Action<br/>cmd = vx, vy, turn<br/>kick = left / right<br/>stop_kick, place_ball"/]
    R["Game rules<br/>e.g. new ball"]
    P["PolicyInference<br/>velstand.onnx or ball_kick_*.onnx<br/>61-D observation"]
    J["14 joint targets"]
    BAM["BAM servo model"]
    PH["Physics<br/>4 x 5 ms steps"]
    O["log.csv, events, trails,<br/>3D markers, status panel"]

    W --> G --> S --> B --> A --> P --> J --> BAM --> PH --> W
    S --> R
    A --> O

    classDef ours fill:#dbeafe,stroke:#2563eb,color:#0b2545
    classDef pollen fill:#ffedd5,stroke:#ea580c,color:#431407
    class G,S,B,A,R,O ours
    class W,P,J,BAM,PH pollen
```

The brain never touches MuJoCo: it only sees a `Snapshot` and returns an
`Action` (`common/brain.py`). That's why `tests/` can check brains against an
imaginary duck in a fraction of a second.

## Kick loop: the brain's state machine

`kick/brain.py`. The edge labels are short versions of the reasons the brain
prints (and shows in the viewer); the table below gives the exact conditions.

```mermaid
stateDiagram-v2
    [*] --> APPROACH
    state "On its feet" as UP {
        APPROACH --> ALIGN : behind the ball
        ALIGN --> SETTLE : lined up
        ALIGN --> APPROACH : overshot / off the spot / too slow / ball got away
        SETTLE --> KICK : still for 1 s, on the spot
        SETTLE --> ALIGN : drifted
        KICK --> WATCH : kick done
        KICK --> APPROACH : kick never started
        WATCH --> APPROACH : ball stopped, new aim
        APPROACH --> APPROACH : ball knocked / taking too long, new aim
    }
    UP --> RECOVER : fell over
    RECOVER --> APPROACH : back up
```

| Transition | Exact condition (`KickParams` defaults) |
|---|---|
| APPROACH → ALIGN | trunk within 3 cm of the pre-stance point, 20 cm behind the kick stance (reached straight or via the orbit below) |
| ALIGN → SETTLE | facing the aim within 8° and the ball within 12 mm (0.8 × 15 mm tolerance) of the kick spot, or within 15 mm once fully walked in; held for 0.3 s while standing |
| ALIGN → APPROACH | walked 15 mm past the stance ("overshot"); right depth and heading but ball still off the spot ("can't shuffle"); 15 s in ALIGN; or more than 32 cm from the stance ("ball got away") |
| SETTLE → KICK | after 1 s standing: ball within 15 mm, heading within 12° |
| SETTLE → ALIGN | after 1 s: ball drifted off (more than 15 mm or 12°) |
| KICK → WATCH | the kick policy ran (~3 s) and handed back to walking |
| KICK → APPROACH | the kick policy hadn't started after 0.5 s |
| WATCH → APPROACH | ball slower than 5 cm/s (after at least 0.5 s), or 6 s: score the kick, pick a new aim |
| → APPROACH, new aim | from APPROACH / ALIGN / SETTLE: ball moved over 5 cm since aiming and is still ("ball knocked"); from any state but KICK: 45 s since the duck got within 60 cm of the ball ("taking too long") |
| → RECOVER | trunk below 7.5 cm, from any state (a running kick is stopped) |
| RECOVER → APPROACH | trunk back above 10.5 cm |

New aims are random: roughly onwards (the duck->ball direction ±60°), or back
towards the middle (±45°) when the ball is outside the 1.5 m arena.

Inside ALIGN the brain does one of: turn on the spot (facing over 15° off),
walk in along the line steering like a car, or give up and retry. The real
walking policy can't sidestep, walk backwards or take tiny steps, so retrying
is the only way to fix a small miss.

### Getting behind the ball: the orbit

In APPROACH the duck heads for the **pre-stance point** (20 cm behind where it
must stand to kick, on the aim line). If the straight way there would take it
closer than 20 cm to the ball, it **orbits** instead:

```mermaid
flowchart LR
    A["Straight route to the<br/>pre-stance point clear<br/>(stays 20 cm from the ball)?"] -- yes --> S["Walk straight there"]
    A -- no --> O["Orbit: circle the ball 10 cm further out<br/>than the pre-stance point, the shorter way round,<br/>steering at a point 35° ahead on the circle"]
    O -- "within 30° of the aim line" --> I["Head straight in:<br/>now along the aim, so it<br/>arrives facing the right way"]
    S --> AL["ALIGN"]
    I --> AL
    classDef ours fill:#dbeafe,stroke:#2563eb,color:#0b2545
    class A,S,O,I,AL ours
```

Near 180° (both ways round equally long) it keeps the way it started, so it
can't dither. The route it's following, arc included, is `brain.plan`, which
the viewer draws. Every orbit is logged in `events.json` (`orbits`). In MuJoCo
it orbits ~5 times per 5 minutes in the kick loop, almost always for a
reversal after the ball leaves the arena, and before every kick in kickback.

## Kickback: back and forth on a line

`kick/kickback.py`. The same brain with one different method,
`KickbackBrain.pick_aim`: each kick goes **back the way the last one went**, at
the end of a 1 m line behind the last kick's direction (the first kick goes
away from the duck). It aims at that end from wherever the ball stopped, so a
ball that drifted off the line is steered back towards it. Every kick then
needs the duck to get round the ball, which is what the orbit is for.

## Chase: the brain plus the game rules

Brain: `fetch/chaser.py` (two states). Rules: `fetch/chase_ball.py`.

```mermaid
stateDiagram-v2
    [*] --> CHASE : ball dropped 0.6-1.4 m away
    CHASE --> CHASE : reached it, new ball
    CHASE --> CHASE : gave up, new ball
    CHASE --> RECOVER : fell over
    RECOVER --> CHASE : back up
```

"Reached it" = trunk within 13 cm of the ball; "gave up" = 25 s on one ball.
Fallen = trunk below 7.5 cm, back up = above 10.5 cm.
In CHASE the command comes from `common/steer.py: steer_to`: turn towards the
ball, walk only when roughly facing it, slow down near it, and never send a
forward command the walking policy would ignore (below 0.3 m/s).

## Who does what

```mermaid
flowchart LR
    subgraph OURS["Ours: this repo"]
        direction TB
        KB["Kick brain<br/>APPROACH, ALIGN, SETTLE, WATCH, RECOVER:<br/>picks the velocity command"]
        KK["Kick brain, KICK:<br/>picks the foot, says when"]
        CB["Chase brain<br/>CHASE, RECOVER:<br/>picks the velocity command"]
        GL["Also ours: game loop, game rules,<br/>3D markers, status panel,<br/>plots, summary.json, video"]
    end
    subgraph POLLEN["Pre-installed: Pollen's microduck_rl + Hub"]
        direction TB
        VS["velstand.onnx walking policy<br/>walks at the command, stands at zero,<br/>gets itself up after a fall"]
        KP["ball_kick_left / right.onnx<br/>blind kick, ~3 s, then hands back"]
        PI["PolicyInference<br/>(scripts/infer_policy.py)"]
        BAM["BAM servo model, 14 joints"]
        MJ["MuJoCo scene: duck + 70 mm ball"]
    end
    KB -- "vx, vy, turn" --> VS
    CB -- "vx, vy, turn" --> VS
    KK -- "start kick" --> KP
    VS --> PI
    KP --> PI
    PI --> BAM --> MJ

    classDef ours fill:#dbeafe,stroke:#2563eb,color:#0b2545
    classDef pollen fill:#ffedd5,stroke:#ea580c,color:#431407
    class KB,KK,CB,GL ours
    class VS,KP,PI,BAM,MJ pollen
    style OURS fill:#eff6ff,stroke:#2563eb
    style POLLEN fill:#fff7ed,stroke:#ea580c
```

| State | Ours | Pre-installed (Pollen) |
|---|---|---|
| APPROACH | where to walk (behind the ball, round it if needed), the command | velstand walks |
| ALIGN | turn / walk in / give up, the command | velstand walks and turns |
| SETTLE | zero command, checks it's still lined up | velstand stands still |
| KICK | which foot, when, nothing else | ball_kick ONNX runs the whole kick, then hands back to velstand |
| WATCH | zero command, scores the kick, picks the next aim | velstand stands still |
| CHASE | `steer_to` the ball, the command | velstand walks |
| RECOVER | zero command, waits | velstand's built-in get-up |
| always | game loop, rules, markers, plots | PolicyInference, BAM servos, MuJoCo physics |

## Watching the brain live

The viewer's top-left panel shows the brain's `state` and **why** it last
changed (e.g. `ALIGN -> APPROACH (overshot the ball)`). The same lines are
printed to the terminal, and `log.csv` has a `state` column for every step
(the kick loop's `timeline.png` draws it).

## Adding things

- **A new brain** (e.g. a learned Stage-1 chaser): any object with
  `update(snapshot) -> Action`, `state`, `last_change` and `events`.
  Subclass `common.brain.Brain` to get the last three. Pass it to `Chase(...)`
  in `fetch/chase_ball.py:main` instead of `ChaseBrain()`.
- **A new game**: subclass `common.game.Game` and give it a brain, `start()`,
  and optionally `rules()`, `draw()`, `status()`, `log_row()`, `summary()`,
  `plots()`. Then `sim.run(duck, game, args, "name")`. Falls, stepping,
  logging, trails and the live / headless runners come for free.
- **Test it** like `tests/test_chaser.py`: feed Snapshots from an imaginary
  duck (`tests/legs.py:lazy_legs` makes it move like the real legs).

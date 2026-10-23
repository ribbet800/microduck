"""The kick loop's brain: a state machine that plays football against itself.

Pollen's kick policy is BLIND — it never sees the ball. It was trained starting
from a still stand with the ball centred 9 cm in front of the trunk and 4.2 cm
to the side (in front of the kicking foot), give or take 1.5 cm, and it kicks
straight ahead. So all the skill here is in lining up:

    APPROACH  walk to a point behind the ball, on the line from the ball to the
              goal (going round the ball if the duck is on the wrong side)
    ALIGN     shuffle slowly (forwards/sideways/turn) until the ball sits on the
              kick spot and the duck faces the goal
    SETTLE    stand still for a second (the kick was trained from a still stand)
    KICK      swap to the kick policy for ~3 s, then back to walking
    WATCH     wait for the ball to stop (or go in / out), then go again
    RECOVER   fell over: zero command while the walking policy gets up

Two goals sit at either end of the pitch. A goal is scored when the ball rolls
into the goal circle the duck is attacking; then the ball goes back to the
centre and the duck attacks the other end.

No MuJoCo in here: `update` takes a snapshot of the world and returns what to
do, so it can be tested with an imaginary perfect duck (tests/test_kick_brain.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from fetch.chaser import ChaserParams, ball_in_duck_frame, chase_command


@dataclass
class KickParams:
    # Pitch
    goal_x: float = 1.5                  # goals at (±goal_x, 0)
    goal_radius: float = 0.3
    pitch_half: tuple = (2.2, 1.4)        # beyond this the ball is "out"
    # Where the kick policy expects the ball (microduck_ball_kick_env_cfg.py:
    # BALL_OFFSET_X / BALL_OFFSET_ABS_Y, trained with ±1.5 cm noise).
    spot_x: float = 0.09
    spot_y: float = 0.042
    tolerance: float = 0.012             # kick only if the ball is this close to the spot (m)
    heading_tol: float = math.radians(8)  # ...and the duck faces the goal this well
    # Approach
    standoff: float = 0.15               # pre-stance point this far behind the kick stance
    approach_radius: float = 0.06
    go_round: float = 0.25               # how wide to circle round the ball
    max_forward: float = 0.25
    max_turn: float = 0.9
    # Fine alignment
    align_gain: float = 1.5              # m/s of command per m of error
    align_max_v: float = 0.06
    align_min_v: float = 0.025           # tiny commands may not move the duck at all
    align_turn_gain: float = 1.5
    align_max_turn: float = 0.5
    align_timeout_s: float = 15.0
    # Kick and after
    settle_s: float = 1.0
    watch_s: float = 4.0
    ball_stopped: float = 0.05           # m/s
    attempt_timeout_s: float = 45.0
    # Assist: if the duck gets within `assist_radius` but can't nail the spot,
    # nudge the ball into place (what Pollen's own demo does every time).
    assist: bool = False
    assist_radius: float = 0.05


@dataclass
class Snapshot:
    t: float
    duck_xy: np.ndarray
    duck_yaw: float
    ball_xy: np.ndarray
    ball_speed: float
    fallen: bool
    kicking: bool


@dataclass
class Action:
    cmd: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    kick: str | None = None              # "left" / "right": start that kick policy now
    stop_kick: bool = False
    place_ball: tuple | None = None      # teleport the ball here (assist / kick-off)


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def _left(a):
    return np.array([-a[1], a[0]])


def _wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class KickBrain:
    def __init__(self, params: KickParams, feet: list[str]):
        assert feet, "need at least one kick policy"
        self.p = params
        self.feet = feet
        self.state = "APPROACH"
        self.state_t0 = 0.0
        self.attempt_t0 = 0.0
        self.attack = +1                 # attacking the goal at (+goal_x, 0)
        self.foot = feet[0]
        self.score = 0
        self.events = {"kicks": [], "goals": [], "outs": [], "falls": [], "retries": [], "timeouts": []}
        self._kick_started = False
        self._kick_peak = 0.0
        # Exposed for drawing / logging:
        self.aim = np.array([1.0, 0.0])
        self.stance = np.zeros(2)
        self.target = np.zeros(2)
        self.align_err = float("nan")
        self.heading_err = float("nan")

    # -- geometry -------------------------------------------------------------
    @property
    def goal_xy(self):
        return np.array([self.attack * self.p.goal_x, 0.0])

    def spot_rel(self, foot=None):
        """Where the ball should sit, in the duck's frame (x ahead, y left)."""
        foot = foot or self.foot
        return np.array([self.p.spot_x, -self.p.spot_y if foot == "right" else self.p.spot_y])

    def stance_for(self, ball_xy, foot):
        """Trunk position that puts the ball on `foot`'s kick spot, facing the goal."""
        a = _unit(self.goal_xy - ball_xy)
        r = self.spot_rel(foot)
        return ball_xy - r[0] * a - r[1] * _left(a), a

    def spot_world(self, s: Snapshot):
        c, sn = math.cos(s.duck_yaw), math.sin(s.duck_yaw)
        r = self.spot_rel()
        return s.duck_xy + np.array([c * r[0] - sn * r[1], sn * r[0] + c * r[1]])

    def _set(self, state, t, why=""):
        if state != self.state:
            print(f"[{t:6.1f}s] {self.state} -> {state}" + (f"  ({why})" if why else ""))
        self.state, self.state_t0 = state, t

    # -- main step -------------------------------------------------------------
    def update(self, s: Snapshot) -> Action:
        p, act = self.p, Action()
        dt_state = s.t - self.state_t0

        # Falls trump everything.
        if s.fallen and self.state != "RECOVER":
            self.events["falls"].append({"t": s.t, "x": float(s.duck_xy[0]), "y": float(s.duck_xy[1]),
                                         "during": self.state})
            act.stop_kick = s.kicking
            self._set("RECOVER", s.t, "fell over")
            return act
        if self.state == "RECOVER":
            if not s.fallen:
                self._set("APPROACH", s.t, "back up")
            return act

        # Track how hard the last kick hit the ball.
        if self.state in ("KICK", "WATCH"):
            self._kick_peak = max(self._kick_peak, s.ball_speed)

        # Goals and outs — checked all the time, including mid-kick, since a good
        # shot can roll straight through the goal before the kick policy finishes.
        if np.linalg.norm(s.ball_xy - self.goal_xy) < p.goal_radius:
            self.score += 1
            self.events["goals"].append({"t": s.t, "goal": int(self.attack), "score": self.score})
            print(f"[{s.t:6.1f}s] GOAL! ({self.score})  now attacking the other end")
            self.attack = -self.attack
            return self._kickoff(s, act)
        if abs(s.ball_xy[0]) > p.pitch_half[0] or abs(s.ball_xy[1]) > p.pitch_half[1]:
            self.events["outs"].append({"t": s.t, "x": float(s.ball_xy[0]), "y": float(s.ball_xy[1])})
            print(f"[{s.t:6.1f}s] ball out of play — back to the centre")
            return self._kickoff(s, act)
        if self.state != "KICK" and s.t - self.attempt_t0 > p.attempt_timeout_s:
            self.events["timeouts"].append({"t": s.t})
            print(f"[{s.t:6.1f}s] taking too long — ball back to the centre")
            return self._kickoff(s, act)

        if self.state == "APPROACH":
            act.cmd = self._approach(s)
        elif self.state == "ALIGN":
            act = self._align(s, dt_state)
        elif self.state == "SETTLE":
            act = self._settle(s, dt_state)
        elif self.state == "KICK":
            if s.kicking:
                self._kick_started = True
            elif self._kick_started:
                self._set("WATCH", s.t, "kick done")
        elif self.state == "WATCH":
            if (dt_state > 0.5 and s.ball_speed < p.ball_stopped) or dt_state > p.watch_s:
                self._finish_kick(s)
                self.attempt_t0 = s.t
                self._set("APPROACH", s.t, f"ball reached {self._kick_peak:.2f} m/s")
        return act

    def _finish_kick(self, s):
        """Record how the last kick went (once)."""
        if self.events["kicks"] and "peak_ball_speed" not in self.events["kicks"][-1]:
            k = self.events["kicks"][-1]
            k["peak_ball_speed"] = round(self._kick_peak, 3)
            k["travel_m"] = round(float(np.linalg.norm(s.ball_xy - np.array(k["ball_xy"]))), 3)

    def _kickoff(self, s, act):
        if self.state in ("KICK", "WATCH"):
            self._finish_kick(s)
        act.place_ball = (0.0, 0.0)
        self.attempt_t0 = s.t
        self._set("APPROACH", s.t, "kick-off")
        return act

    # -- states ------------------------------------------------------------------
    def _approach(self, s: Snapshot):
        p = self.p
        # Pick the foot whose stance is closest to where the duck already is.
        options = {f: self.stance_for(s.ball_xy, f) for f in self.feet}
        self.foot = min(options, key=lambda f: np.linalg.norm(options[f][0] - s.duck_xy))
        stance, a = options[self.foot]
        self.aim, self.stance = a, stance
        pre = stance - p.standoff * a

        # On the wrong side of the ball? Walk round it, not through it.
        to_duck = s.duck_xy - s.ball_xy
        ahead = float(np.dot(to_duck, a))
        side = float(np.dot(to_duck, _left(a)))
        if ahead > -0.05 and abs(side) < p.go_round:
            sgn = 1.0 if side >= 0 else -1.0
            target = s.ball_xy + sgn * (p.go_round + 0.05) * _left(a) - 0.1 * a
        else:
            target = pre
        self.target = target

        rel = ball_in_duck_frame(s.duck_xy, s.duck_yaw, target)
        cp = ChaserParams(max_forward=p.max_forward, max_turn=p.max_turn,
                          catch_radius=p.approach_radius, slow_radius=0.3)
        cmd, dist, _ = chase_command(rel, cp)
        if target is pre and dist < p.approach_radius:
            self._set("ALIGN", s.t, f"behind the ball, {self.foot} foot")
        return cmd

    def _errors(self, s: Snapshot):
        rel = ball_in_duck_frame(s.duck_xy, s.duck_yaw, s.ball_xy)
        e = rel - self.spot_rel()
        self.aim = _unit(self.goal_xy - s.ball_xy)
        h = _wrap(math.atan2(self.aim[1], self.aim[0]) - s.duck_yaw)
        self.align_err, self.heading_err = float(np.linalg.norm(e)), h
        return e, h

    def _align(self, s: Snapshot, dt_state):
        p, act = self.p, Action()
        e, h = self._errors(s)
        self.target = self.spot_world(s)
        err = float(np.linalg.norm(e))
        # "Got away" is judged in the world (how far the duck is from the kick
        # stance), not in the duck's frame, which swings wildly while turning.
        stance, _ = self.stance_for(s.ball_xy, self.foot)
        if np.linalg.norm(stance - s.duck_xy) > p.standoff + 0.12:
            self._set("APPROACH", s.t, "ball got away")
            return act
        if err < 0.8 * p.tolerance and abs(h) < p.heading_tol:
            self._set("SETTLE", s.t, f"lined up ({err * 1000:.0f} mm off)")
            return act
        if dt_state > p.align_timeout_s:
            self.events["retries"].append({"t": s.t, "err_m": round(err, 4)})
            self._set("APPROACH", s.t, "couldn't line up, trying again")
            return act
        # Move so the ball's error shrinks: if the ball is too far ahead, step
        # forward; too far left, step left. Plus turn to face the goal.
        v = p.align_gain * e
        n = float(np.linalg.norm(v))
        if n > p.align_max_v:
            v *= p.align_max_v / n
        elif n < p.align_min_v and err > p.tolerance / 2:
            v *= p.align_min_v / max(n, 1e-9)
        wz = float(np.clip(p.align_turn_gain * h, -p.align_max_turn, p.align_max_turn))
        if abs(h) > math.radians(25):
            v[:] = 0.0          # facing well off: turn on the spot first, then shuffle
        act.cmd = np.array([v[0], v[1], wz], dtype=np.float32)
        return act

    def _settle(self, s: Snapshot, dt_state):
        p, act = self.p, Action()
        e, h = self._errors(s)
        self.target = self.spot_world(s)
        if dt_state < p.settle_s:
            return act
        err = float(np.linalg.norm(e))
        assisted = False
        if err > p.tolerance or abs(h) > 1.5 * p.heading_tol:
            if p.assist and err < p.assist_radius and abs(h) < 3 * p.heading_tol:
                act.place_ball = tuple(self.spot_world(s))
                assisted = True
            else:
                self._set("ALIGN", s.t, f"drifted to {err * 1000:.0f} mm while settling")
                return act
        act.kick = self.foot
        self._kick_started = False
        self._kick_peak = 0.0
        rel = ball_in_duck_frame(s.duck_xy, s.duck_yaw, s.ball_xy)
        self.events["kicks"].append({
            "t": s.t, "foot": self.foot, "assisted": assisted,
            "ball_rel": [round(float(v), 4) for v in rel],
            "err_m": round(err, 4), "heading_err_deg": round(math.degrees(h), 1),
            "ball_xy": [float(v) for v in (act.place_ball or s.ball_xy)],
            "aim": [float(v) for v in self.aim], "attacking": int(self.attack),
        })
        print(f"[{s.t:6.1f}s] KICK with the {self.foot} foot"
              f" ({'assisted' if assisted else f'{err * 1000:.0f} mm off'})")
        self._set("KICK", s.t)
        return act

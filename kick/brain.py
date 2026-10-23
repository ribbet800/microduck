"""The kick loop's brain: walk up to the ball, kick it, follow it, kick it again.

Pollen's kick policy is BLIND — it never sees the ball. It was trained starting
from a still stand with the ball centred 9 cm in front of the trunk and 4.2 cm
to the side (in front of the kicking foot), give or take 1.5 cm, and it kicks
straight ahead. So where the ball goes is decided by which way the duck faces
when it kicks, and all the skill is in lining up.

Each time the ball comes to rest, the brain picks a new random aim:
  - inside the arena (within `arena_radius` of the origin): the direction from
    the duck to the ball, give or take up to `aim_spread`. Roughly "onwards",
    so the duck rarely has to walk round the ball (when it tends to knock it).
  - outside it: back towards the origin, give or take up to `home_spread`.
    A soft wall: the ball is never teleported.

    APPROACH  walk to a point behind the ball, on the aim line (going round
              the ball if the duck is on the wrong side)
    ALIGN     turn in place to face along the aim, then walk in along the
              line, steering out any sideways error, and stop with the ball on
              the kick spot. (Pollen's walking policy can't shuffle: it ignores
              small, sideways and backward commands. Overshoot = start again.)
    SETTLE    stand still for a second (the kick was trained from a still stand)
    KICK      swap to the kick policy for ~3 s, then back to walking
    WATCH     wait for the ball to stop, record where it went against where it
              was aimed, pick the next aim
    RECOVER   fell over: zero command while the walking policy gets up

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
    # Aiming: random, roughly onwards, inside a soft arena.
    aim_spread: float = math.radians(60)   # aim = duck->ball direction ± up to this
    arena_radius: float = 1.5              # soft arena round the origin (m)
    home_spread: float = math.radians(45)  # outside it: aim back home ± up to this
    moved: float = 0.05                    # ball knocked this far since aiming (m): aim again
    # Where the kick policy expects the ball (microduck_ball_kick_env_cfg.py:
    # BALL_OFFSET_X / BALL_OFFSET_ABS_Y, trained with ±1.5 cm noise).
    spot_x: float = 0.09
    spot_y: float = 0.042
    tolerance: float = 0.012             # kick only if the ball is this close to the spot (m)
    heading_tol: float = math.radians(8)  # ...and the duck faces along the aim this well
    # Approach
    standoff: float = 0.25               # pre-stance point this far behind the kick stance:
                                         # the run-in that steers out sideways error
    approach_radius: float = 0.03
    go_round: float = 0.25               # how wide to circle round the ball
    max_forward: float = 0.4
    min_forward: float = 0.3             # velstand ignores less from a stand (see fetch/chaser.py)
    max_turn: float = 0.9
    # Fine alignment: turn in place, then walk in along the line, steering.
    min_turn: float = 0.75               # in-place turns below ~0.7 rad/s don't happen at all
    turn_first: float = math.radians(15)  # heading error above this: turn in place before walking in
    walk_in_v: float = 0.3               # forward command while walking in (~0.08 m/s achieved)
    stop_lead: float = 0.004             # stop when the ball is this far short of the spot (m)
    align_turn_gain: float = 3.0         # rad/s of steering per rad of heading error
    align_lat_gain: float = 10.0         # steer this much off the aim line per m of sideways error...
    align_max_cut: float = math.radians(30)  # ...up to this angle (like a car's lane keeping)
    align_max_turn: float = 0.9
    align_timeout_s: float = 15.0
    # Kick and after
    settle_s: float = 1.0
    watch_s: float = 6.0
    ball_stopped: float = 0.05           # m/s
    attempt_timeout_s: float = 45.0      # counted from when the duck gets within near_ball
    near_ball: float = 0.6
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
    place_ball: tuple | None = None      # teleport the ball here (assist only)


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def _left(a):
    return np.array([-a[1], a[0]])


def _wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class KickBrain:
    def __init__(self, params: KickParams, feet: list[str], rng: np.random.Generator | None = None):
        assert feet, "need at least one kick policy"
        self.p = params
        self.feet = feet
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.state = "APPROACH"
        self.state_t0 = 0.0
        self.attempt_t0 = 0.0
        self.foot = feet[0]
        self.events = {"kicks": [], "falls": [], "retries": [], "timeouts": []}
        self._kick_started = False
        self._kick_peak = 0.0
        # The current aim: a unit vector, where the ball was when it was picked,
        # and whether it was a free pick or a "come home" one.
        self.aim = None
        self.aim_from = None
        self.aim_mode = ""
        # Exposed for drawing / logging:
        self.stance = np.zeros(2)
        self.target = np.zeros(2)
        self.align_err = float("nan")
        self.heading_err = float("nan")

    # -- aiming -----------------------------------------------------------------
    def pick_aim(self, s: Snapshot):
        """Choose where the next kick should send the ball."""
        p, ball = self.p, np.asarray(s.ball_xy, float)
        if np.linalg.norm(ball) > p.arena_radius:
            base, spread, self.aim_mode = math.atan2(-ball[1], -ball[0]), p.home_spread, "home"
        else:
            d = ball - s.duck_xy
            base, spread, self.aim_mode = math.atan2(d[1], d[0]), p.aim_spread, "free"
        ang = base + self.rng.uniform(-spread, spread)
        self.aim = np.array([math.cos(ang), math.sin(ang)])
        self.aim_from = ball.copy()

    @property
    def aim_deg(self):
        return math.degrees(math.atan2(self.aim[1], self.aim[0]))

    # -- geometry -------------------------------------------------------------
    def spot_rel(self, foot=None):
        """Where the ball should sit, in the duck's frame (x ahead, y left)."""
        foot = foot or self.foot
        return np.array([self.p.spot_x, -self.p.spot_y if foot == "right" else self.p.spot_y])

    def stance_for(self, ball_xy, foot):
        """Trunk position that puts the ball on `foot`'s kick spot, facing along the aim."""
        a = self.aim
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
        if self.aim is None:
            self.pick_aim(s)
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
                self.attempt_t0 = s.t
                self._set("APPROACH", s.t, "back up")
            return act

        # Track how hard the last kick hit the ball.
        if self.state in ("KICK", "WATCH"):
            self._kick_peak = max(self._kick_peak, s.ball_speed)

        # The attempt clock only runs once the duck is near the ball: a long kick
        # can mean a long walk, which isn't being stuck.
        if self.state == "APPROACH" and np.linalg.norm(s.ball_xy - s.duck_xy) > p.near_ball:
            self.attempt_t0 = s.t
        if self.state != "KICK" and s.t - self.attempt_t0 > p.attempt_timeout_s:
            self.events["timeouts"].append({"t": s.t})
            if self.state == "WATCH":
                self._finish_kick(s)
            return self._new_attempt(s, "taking too long, aiming somewhere else")
        # Knocked the ball (not with a kick)? Once it's still again, aim afresh from there.
        if (self.state in ("APPROACH", "ALIGN", "SETTLE") and s.ball_speed < p.ball_stopped
                and np.linalg.norm(s.ball_xy - self.aim_from) > p.moved):
            return self._new_attempt(s, "ball was knocked")

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
                k = self._finish_kick(s)
                went = f"went {k['actual_deg']:+.0f}°" if k.get("actual_deg") is not None else "barely moved"
                return self._new_attempt(s, f"aimed {k['aim_deg']:+.0f}°, {went}, {k['travel_m']:.2f} m")
        return act

    def _new_attempt(self, s, why):
        self.pick_aim(s)
        self.attempt_t0 = s.t
        self._set("APPROACH", s.t, f"{why}; next aim {self.aim_deg:+.0f}°"
                  + (" (back towards the middle)" if self.aim_mode == "home" else ""))
        return Action()

    def _finish_kick(self, s):
        """Record how the last kick went: speed, distance, and where it went vs the aim."""
        k = self.events["kicks"][-1]
        if "peak_ball_speed" not in k:
            d = np.asarray(s.ball_xy, float) - np.array(k["ball_xy"])
            travel = float(np.linalg.norm(d))
            k["peak_ball_speed"] = round(self._kick_peak, 3)
            k["travel_m"] = round(travel, 3)
            k["end_xy"] = [float(v) for v in s.ball_xy]
            if travel > 0.05:
                actual = math.degrees(math.atan2(d[1], d[0]))
                k["actual_deg"] = round(actual, 1)
                k["aim_err_deg"] = round(math.degrees(_wrap(math.radians(actual - k["aim_deg"]))), 1)
            else:
                k["actual_deg"] = k["aim_err_deg"] = None
        return k

    # -- states ------------------------------------------------------------------
    def _approach(self, s: Snapshot):
        p = self.p
        # Pick the foot whose stance is closest to where the duck already is.
        options = {f: self.stance_for(s.ball_xy, f) for f in self.feet}
        self.foot = min(options, key=lambda f: np.linalg.norm(options[f][0] - s.duck_xy))
        stance, a = options[self.foot]
        self.stance = stance
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
        cp = ChaserParams(max_forward=p.max_forward, min_forward=p.min_forward, max_turn=p.max_turn,
                          catch_radius=p.approach_radius, slow_radius=0.3)
        cmd, dist, _ = chase_command(rel, cp)
        if target is pre and dist < p.approach_radius:
            self._set("ALIGN", s.t, f"behind the ball, {self.foot} foot")
        return cmd

    def _errors(self, s: Snapshot):
        rel = ball_in_duck_frame(s.duck_xy, s.duck_yaw, s.ball_xy)
        e = rel - self.spot_rel()
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
        stance, a = self.stance_for(s.ball_xy, self.foot)
        if np.linalg.norm(stance - s.duck_xy) > p.standoff + 0.12:
            self._set("APPROACH", s.t, "ball got away")
            return act
        if err < 0.8 * p.tolerance and abs(h) < p.heading_tol:
            self._set("SETTLE", s.t, f"lined up ({err * 1000:.0f} mm off)")
            return act
        if p.assist and err < p.assist_radius and abs(h) < 3 * p.heading_tol \
                and float(np.dot(stance - s.duck_xy, a)) < p.stop_lead:
            self._set("SETTLE", s.t, f"close enough for an assist ({err * 1000:.0f} mm off)")
            return act
        if dt_state > p.align_timeout_s:
            return self._retry(s, err, "couldn't line up, trying again")
        # Where the trunk is relative to the kick stance, in the world along the
        # aim line: `ahead` > 0 means it still has to walk forward, `lat` > 0
        # that it needs to move left. (Judged in the world, not from the ball in
        # the duck's frame: there, any heading error leaks into the sideways error,
        # and a last turn on the spot swings the kick spot 9 cm ahead sideways.)
        to_stance = stance - s.duck_xy
        ahead, lat = float(np.dot(to_stance, a)), float(np.dot(to_stance, _left(a)))
        # Steer like a car onto the line: aim off the kick direction, towards
        # the side the duck needs to move to, by an angle that shrinks as the
        # sideways error does. `hd` is the heading error from that.
        cut = float(np.clip(math.atan(p.align_lat_gain * lat), -p.align_max_cut, p.align_max_cut))
        hd = _wrap(h + cut) if ahead > p.stop_lead else h
        if abs(hd) > p.turn_first:
            # Facing well off: turn on the spot first (hard enough to beat the dead zone).
            act.cmd = np.array([0.0, 0.0, math.copysign(p.min_turn, hd)], dtype=np.float32)
        elif ahead > p.stop_lead:
            wz = float(np.clip(p.align_turn_gain * hd, -p.align_max_turn, p.align_max_turn))
            act.cmd = np.array([p.walk_in_v, 0.0, wz], dtype=np.float32)
        elif ahead < -p.tolerance:
            # Walked past it. The duck can't step back, so go round again.
            return self._retry(s, err, "overshot the ball")
        elif abs(h) > p.heading_tol:
            act.cmd = np.array([0.0, 0.0, math.copysign(p.min_turn, h)], dtype=np.float32)
        else:
            # Right depth, facing along the aim, but off to the side: no sidestep, so retry.
            return self._retry(s, err, f"{abs(lat) * 1000:.0f} mm off to the side")
        return act

    def _retry(self, s, err, why):
        self.events["retries"].append({"t": s.t, "err_m": round(err, 4), "why": why})
        self._set("APPROACH", s.t, why)
        return Action()

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
        ball_xy = [float(v) for v in (act.place_ball or s.ball_xy)]
        self.aim_from = np.array(ball_xy)
        self.events["kicks"].append({
            "t": s.t, "foot": self.foot, "assisted": assisted,
            "ball_rel": [round(float(v), 4) for v in rel],
            "err_m": round(err, 4), "heading_err_deg": round(math.degrees(h), 1),
            "ball_xy": ball_xy,
            "aim": [float(v) for v in self.aim], "aim_deg": round(self.aim_deg, 1), "aim_mode": self.aim_mode,
        })
        print(f"[{s.t:6.1f}s] KICK with the {self.foot} foot, aiming {self.aim_deg:+.0f}°"
              f" ({'assisted' if assisted else f'{err * 1000:.0f} mm off'})")
        self._set("KICK", s.t)
        return act

"""Kickback's brain: kick the ball back and forth along a straight line.

Every kick aims back the way the last one went, at the end of the line behind
it (the first kick: away from the duck). The aim points at that end from
wherever the ball stopped, so a ball that drifted off the line is sent back
towards it.

Pollen's kick policy is BLIND: it never sees the ball. It was trained starting
from a still stand with the ball centred 9 cm in front of the trunk and 4.2 cm
to the side (in front of the kicking foot), give or take 1.5 cm, and it kicks
straight ahead. So where the ball goes is decided by which way the duck faces
when it kicks, and all the skill is in lining up.

    APPROACH  get behind the ball on the aim line, facing along it: if the
              straight way passes too close to the ball, orbit it at a safe
              distance until nearly on the aim line, then walk straight in
    ALIGN     turn in place to face along the aim, then walk in along the
              line, steering out any sideways error, and stop with the ball on
              the kick spot. (Pollen's walking policy can't shuffle: it ignores
              small, sideways and backward commands. Overshoot = start again.)
    SETTLE    stand still for a second (the kick was trained from a still stand)
    KICK      run the kick skill for ~3 s, then back to walking
    WATCH     wait for the ball to stop, record where it went against where it
              was aimed, aim the next kick
    RECOVER   fell over: zero command while the walking policy gets up

`LearnedKickbackBrain` (at the end) swaps APPROACH + ALIGN for a network
trained by kickback/train.py; the rest is the same.

docs/how_it_works.md draws this as a diagram, with every transition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from common.brain import DECIDE_EVERY, MAX_FORWARD, MAX_TURN, Action, Brain, Network, Snapshot, to_command
from common.steer import SteerParams, from_duck_frame, left_of, steer_to, to_duck_frame, wrap

WHIFF_SPEED = 0.2   # a kick that sends the ball slower than this (m/s) is a whiff


def _gap(p, a, b):
    """Distance from point p to the segment a-b."""
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _on_circle(centre, radius, angle):
    return centre + radius * np.array([math.cos(angle), math.sin(angle)])


@dataclass
class KickbackParams:
    moved: float = 0.05                    # ball knocked this far since aiming (m): aim again
    reach: float = 0.3                     # ...by the duck if its trunk was this close, else it crept
    # Where the kick policy expects the ball (microduck_ball_kick_env_cfg.py:
    # BALL_OFFSET_X / BALL_OFFSET_ABS_Y, trained with ±1.5 cm noise).
    spot_x: float = 0.09
    spot_y: float = 0.042
    tolerance: float = 0.015             # kick only if the ball is this close to the spot (m),
                                         # i.e. inside the kick's training noise
    heading_tol: float = math.radians(8)  # ...and the duck faces along the aim this well
    # Approach
    standoff: float = 0.2                # pre-stance point this far behind the kick stance:
                                         # the run-in that steers out sideways error (0.2 beat
                                         # 0.15 and 0.25 in MuJoCo: most kicks/min, fewest retries)
    # Walking to the pre-stance point: ALIGN takes over within catch_radius of it.
    approach: SteerParams = field(default_factory=lambda: SteerParams(catch_radius=0.03, slow_radius=0.3))
    # Getting behind the ball when it's in the way: orbit it, a little further out
    # than the pre-stance point, until nearly on the aim line, then walk straight
    # in (which is then along the aim, so the duck arrives facing the right way).
    clearance: float = 0.2               # a straight route must pass the ball by this much (trunk
                                         # to ball centre; feet + ball touch at ~0.11 m)
    orbit_margin: float = 0.1            # orbit this much further out than the pre-stance point
    orbit_lead: float = math.radians(35)  # steer at a point this far round the orbit ahead
    orbit_exit: float = math.radians(30)  # leave the orbit this close to the aim line
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
    lined_up_s: float = 0.3              # must stay lined up this long, standing, to count
    got_away: float = 0.12               # further than standoff + this from the stance: approach again
    # Kick and after
    settle_s: float = 1.0
    kick_start_s: float = 0.5            # the kick skill hasn't taken over by then: give up on it
    watch_s: float = 6.0                 # watch the ball at most this long...
    watch_min_s: float = 0.5             # ...and at least this long (it may not be rolling yet)
    ball_stopped: float = 0.05           # m/s
    attempt_timeout_s: float = 45.0      # counted from when the duck gets within near_ball
    near_ball: float = 0.6


class KickbackBrain(Brain):
    """`ends`: the line's two ends (world x, y). `feet`: which feet it may kick
    with ("left" / "right"); each needs its skill ("kick_left" / "kick_right") loaded."""

    def __init__(self, ends, params: KickbackParams = KickbackParams(), feet=("left", "right")):
        super().__init__("APPROACH")
        self.ends = [np.asarray(e, float) for e in ends]
        self.p = params
        self.feet = list(feet)
        self.foot = self.feet[0]
        self.attempt_t0 = 0.0
        self.events = {"kicks": [], "retries": [], "orbits": [], "timeouts": [], "knocks": [], "crept": []}
        self._kick_started = False
        self._kick_peak = 0.0
        self._lined_up_t0 = None
        self._orbit_dir = 0.0            # +1 anticlockwise round the ball, -1 clockwise, 0 not orbiting
        # The current aim (a unit vector) and where the ball was when it was picked.
        self.aim = None
        self.aim_from = None
        # Line-up errors, for the status panel and log.csv:
        self.align_err = float("nan")
        self.heading_err = float("nan")

    # -- aiming -----------------------------------------------------------------
    def pick_aim(self, s: Snapshot):
        """Aim at the end of the line behind the last kick (first kick: away from the duck)."""
        ball = np.asarray(s.ball_xy, float)
        if self.events["kicks"]:
            back = -np.asarray(self.events["kicks"][-1]["aim"])
        else:
            back = ball - np.asarray(s.duck_xy, float)
        end = max(self.ends, key=lambda e: float(np.dot(e - ball, back)))
        d = end - ball
        self.aim = d / max(float(np.linalg.norm(d)), 1e-9)
        self.aim_from = ball

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
        r = self.spot_rel(foot)
        return ball_xy - r[0] * self.aim - r[1] * left_of(self.aim)

    def spot_world(self, duck_xy, duck_yaw):
        """The current foot's kick spot, in the world."""
        return from_duck_frame(duck_xy, duck_yaw, self.spot_rel())

    # -- main step -------------------------------------------------------------
    def update(self, s: Snapshot) -> Action:
        p, act = self.p, Action()
        if self.aim is None:
            self.pick_aim(s)
        dt_state = s.t - self.state_t0
        self.plan = []                   # APPROACH and ALIGN fill it in

        # Falls trump everything. (The game loop records them: common/game.py.)
        if s.fallen and self.state != "RECOVER":
            act.stop_skill = s.skill is not None
            self.set_state("RECOVER", s.t, "fell over")
            return act
        if self.state == "RECOVER":
            if not s.fallen:
                self.attempt_t0 = s.t
                self.set_state("APPROACH", s.t, "back up")
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
            return self._new_attempt(s, "taking too long, starting over")
        # Knocked the ball (not with a kick)? Once it's still again, aim afresh from there.
        if (self.state in ("APPROACH", "ALIGN", "SETTLE") and s.ball_speed < p.ball_stopped
                and np.linalg.norm(s.ball_xy - self.aim_from) > p.moved):
            # Knocked by the duck, or just still creeping after WATCH called it stopped
            # (MuJoCo's ball rolls on at 1-2 cm/s for a while)? Both mean aiming again.
            near = float(np.linalg.norm(s.ball_xy - s.duck_xy)) < p.reach
            self.events["knocks" if near else "crept"].append({"t": s.t, "state": self.state})
            return self._new_attempt(s, "ball was knocked")

        if self.state == "APPROACH":
            act.cmd = self._approach(s)
        elif self.state == "ALIGN":
            act = self._align(s, dt_state)
        elif self.state == "SETTLE":
            act = self._settle(s, dt_state)
        elif self.state == "KICK":
            if s.skill is not None:
                self._kick_started = True
            elif self._kick_started:
                self.set_state("WATCH", s.t, "kick done")
            elif dt_state > p.kick_start_s:
                self.events["kicks"].pop()     # it never happened
                self.set_state("APPROACH", s.t, "the kick never started")
        elif self.state == "WATCH":
            if (dt_state > p.watch_min_s and s.ball_speed < p.ball_stopped) or dt_state > p.watch_s:
                k = self._finish_kick(s)
                went = f"went {k['actual_deg']:+.0f}°" if k.get("actual_deg") is not None else "barely moved"
                return self._new_attempt(s, f"aimed {k['aim_deg']:+.0f}°, {went}, {k['travel_m']:.2f} m")
        return act

    def _new_attempt(self, s, why):
        self.pick_aim(s)
        self.attempt_t0 = s.t
        self._orbit_dir = 0.0
        self.set_state("APPROACH", s.t, f"{why}; next aim {self.aim_deg:+.0f}°")
        return Action()

    def _finish_kick(self, s):
        """Record how the last kick went: speed, distance, and where it went vs the aim."""
        k = self.events["kicks"][-1]
        d = np.asarray(s.ball_xy, float) - np.array(k["ball_xy"])
        travel = float(np.linalg.norm(d))
        k["peak_ball_speed"] = round(self._kick_peak, 3)
        k["travel_m"] = round(travel, 3)
        k["end_xy"] = [float(v) for v in s.ball_xy]
        if travel > self.p.moved:     # less than that and its direction means nothing
            actual = math.degrees(math.atan2(d[1], d[0]))
            k["actual_deg"] = round(actual, 1)
            k["aim_err_deg"] = round(math.degrees(wrap(math.radians(actual - k["aim_deg"]))), 1)
        else:
            k["actual_deg"] = k["aim_err_deg"] = None
        return k

    # -- states ------------------------------------------------------------------
    def _approach(self, s: Snapshot):
        p = self.p
        # Pick the foot whose stance is closest to where the duck already is.
        options = {f: self.stance_for(s.ball_xy, f) for f in self.feet}
        self.foot = min(options, key=lambda f: np.linalg.norm(options[f] - s.duck_xy))
        a, stance = self.aim, options[self.foot]
        ball, duck = np.asarray(s.ball_xy, float), np.asarray(s.duck_xy, float)
        pre = stance - p.standoff * a            # where ALIGN takes over
        d, e = duck - ball, pre - ball
        phi = math.atan2(d[1], d[0])             # where the duck is, round the ball
        left = wrap(math.atan2(e[1], e[0]) - phi)  # how far round `pre` still is, the shorter way
        if self._orbit_dir and abs(left) < p.orbit_exit:
            self._orbit_dir = 0.0                # orbited round: straight in from here
        elif not self._orbit_dir:
            # Is the ball in the way? Only if the straight route takes the duck closer to
            # it than it already is (backing off from right next to it is fine).
            gap = _gap(ball, duck, pre)
            if gap < p.clearance and gap < np.linalg.norm(d) - 0.01:
                self._orbit_dir = 1.0 if left >= 0 else -1.0
                self.events["orbits"].append({"t": s.t, "deg": round(math.degrees(abs(left)))})

        if not self._orbit_dir:
            target, route = pre, [pre]
        else:
            # Orbit the shorter way round, but near 180° keep the way already chosen
            # (no dithering), steering at a point `orbit_lead` ahead on the circle.
            if abs(left) < math.radians(150):
                self._orbit_dir = 1.0 if left >= 0 else -1.0
            if left * self._orbit_dir < 0:
                left += self._orbit_dir * 2 * math.pi
            r = float(np.linalg.norm(e)) + p.orbit_margin
            target = _on_circle(ball, r, phi + self._orbit_dir * min(abs(left), p.orbit_lead))
            steps = max(1, int(abs(left) / math.radians(10)))
            route = [_on_circle(ball, r, phi + left * k / steps) for k in range(1, steps + 1)] + [pre]
        self.plan = [duck, *route, stance]

        cmd, _, _ = steer_to(to_duck_frame(duck, s.duck_yaw, target), p.approach)
        if np.linalg.norm(pre - duck) < p.approach.catch_radius:
            self.set_state("ALIGN", s.t, f"behind the ball, {self.foot} foot")
            return np.zeros(3, dtype=np.float32)
        return cmd

    def _errors(self, s: Snapshot):
        rel = to_duck_frame(s.duck_xy, s.duck_yaw, s.ball_xy)
        e = rel - self.spot_rel()
        h = wrap(math.atan2(self.aim[1], self.aim[0]) - s.duck_yaw)
        self.align_err, self.heading_err = float(np.linalg.norm(e)), h
        return e, h

    def _align(self, s: Snapshot, dt_state):
        p, act = self.p, Action()
        e, h = self._errors(s)
        err = float(np.linalg.norm(e))
        # "Got away" is judged in the world (how far the duck is from the kick
        # stance), not in the duck's frame, which swings wildly while turning.
        stance, a = self.stance_for(s.ball_xy, self.foot), self.aim
        self.plan = [np.array(s.duck_xy, float), stance]
        if np.linalg.norm(stance - s.duck_xy) > p.standoff + p.got_away:
            self.set_state("APPROACH", s.t, "ball got away")
            return act
        # Where the trunk is relative to the kick stance, in the world along the
        # aim line: `ahead` > 0 means it still has to walk forward, `lat` > 0
        # that it needs to move left. (Judged in the world, not from the ball in
        # the duck's frame: there, any heading error leaks into the sideways error,
        # and a last turn on the spot swings the kick spot 9 cm ahead sideways.)
        to_stance = stance - s.duck_xy
        ahead, lat = float(np.dot(to_stance, a)), float(np.dot(to_stance, left_of(a)))
        # Lined up: with some margin (0.8 x tolerance) while still walking in,
        # since standing still in SETTLE lets the ball drift by a few mm; once
        # it has walked in as far as it will go, anything inside tolerance will
        # do (the alternative is a ~10 s retry). Either way it has to stay lined
        # up for `lined_up_s` with a zero command: the first step of a walk-in
        # lunges the trunk forward for a moment, which can look lined up and then
        # sway back (seen in MuJoCo: 10 mm -> 17 mm).
        walked_in = ahead <= p.stop_lead
        if abs(h) < p.heading_tol and (err < 0.8 * p.tolerance or (walked_in and err < p.tolerance)):
            if self._lined_up_t0 is None or self._lined_up_t0 < self.state_t0:   # (from an earlier ALIGN)
                self._lined_up_t0 = s.t
            if s.t - self._lined_up_t0 >= p.lined_up_s:
                self.set_state("SETTLE", s.t, f"lined up ({err * 1000:.0f} mm off)")
            return act
        self._lined_up_t0 = None
        if dt_state > p.align_timeout_s:
            return self._retry(s, err, "couldn't line up, trying again")
        # Steer like a car onto the line: aim off the kick direction, towards
        # the side the duck needs to move to, by an angle that shrinks as the
        # sideways error does. `hd` is the heading error from that.
        cut = float(np.clip(math.atan(p.align_lat_gain * lat), -p.align_max_cut, p.align_max_cut))
        hd = wrap(h + cut) if ahead > p.stop_lead else h
        if abs(hd) > p.turn_first:
            act.cmd = self._turn(hd)        # facing well off: turn on the spot first
        elif ahead > p.stop_lead:
            wz = float(np.clip(p.align_turn_gain * hd, -p.align_max_turn, p.align_max_turn))
            act.cmd = np.array([p.walk_in_v, 0.0, wz], dtype=np.float32)
        elif ahead < -p.tolerance:
            # Walked past it. The duck can't step back, so go round again.
            return self._retry(s, err, "overshot the ball")
        elif abs(h) > p.heading_tol:
            act.cmd = self._turn(h)
        else:
            # Right depth, facing along the aim, but the ball's still off the spot:
            # the duck can't shuffle, so retry.
            return self._retry(s, err, f"{err * 1000:.0f} mm off, can't shuffle")
        return act

    def _turn(self, towards):
        """Turn on the spot towards the sign of `towards`, hard enough to beat the dead zone."""
        return np.array([0.0, 0.0, math.copysign(self.p.min_turn, towards)], dtype=np.float32)

    def _retry(self, s, err, why):
        self.events["retries"].append({"t": s.t, "err_m": round(err, 4), "why": why})
        self._orbit_dir = 0.0
        self.set_state("APPROACH", s.t, why)
        return Action()

    def _settle(self, s: Snapshot, dt_state):
        p, act = self.p, Action()
        e, h = self._errors(s)
        if dt_state < p.settle_s:
            return act
        err = float(np.linalg.norm(e))
        # Heading gets a little more slack here (1.5x) than for "lined up":
        # a few degrees of drift while standing is fine.
        if err > p.tolerance or abs(h) > 1.5 * p.heading_tol:
            self.set_state("ALIGN", s.t, f"drifted to {err * 1000:.0f} mm while settling")
            return act
        act.skill = f"kick_{self.foot}"
        self._kick_started = False
        self._kick_peak = 0.0
        rel = to_duck_frame(s.duck_xy, s.duck_yaw, s.ball_xy)
        ball_xy = [float(v) for v in s.ball_xy]
        self.aim_from = np.array(ball_xy)
        self.events["kicks"].append({
            "t": s.t, "foot": self.foot,
            "ball_rel": [round(float(v), 4) for v in rel],
            "err_m": round(err, 4), "heading_err_deg": round(math.degrees(h), 1),
            "ball_xy": ball_xy,
            "aim": [float(v) for v in self.aim], "aim_deg": round(self.aim_deg, 1),
        })
        self.set_state("KICK", s.t, f"{self.foot} foot, aiming {self.aim_deg:+.0f}°, {err * 1000:.0f} mm off")
        return act


# -- the learned brain ---------------------------------------------------------------
OBS_SIZE = 11
TRAIN_MARGIN = 0.8


def observe(s: Snapshot, aim, spot, last_cmd) -> np.ndarray:
    """What the learned brain sees, all in the duck's frame (x ahead, y left), roughly in
    [-1, 1]: where the ball is; how far it is from the kick spot (in 10 cm, so the 15 mm
    that matters is visible); where the trunk has to stand (the stance); the aim as
    sin / cos of the heading error; which foot (+1 left, -1 right); the last command."""
    ball = to_duck_frame(s.duck_xy, s.duck_yaw, s.ball_xy)
    off = np.clip((ball - spot) / 0.1, -3.0, 3.0)
    stance = to_duck_frame(s.duck_xy, s.duck_yaw, s.ball_xy - spot[0] * aim - spot[1] * left_of(aim))
    h = wrap(math.atan2(aim[1], aim[0]) - s.duck_yaw)
    return np.array([*np.clip(ball, -3.0, 3.0), *off, *np.clip(stance, -3.0, 3.0), math.sin(h), math.cos(h),
                     1.0 if spot[1] > 0 else -1.0, last_cmd[0] / MAX_FORWARD, last_cmd[2] / MAX_TURN],
                    dtype=np.float32)


class LearnedKickbackBrain(KickbackBrain):
    """Kickback with a network (kickback/train.py) doing the lining up: in APPROACH it
    walks wherever the network says, 10 times a second, until the ball is on the kick
    spot and it faces along the aim; then it stops by itself, and if that holds for
    `lined_up_s` the hand-written SETTLE / KICK / WATCH take over as usual. (ALIGN, the
    hand-written fine line-up, isn't used: drifting in SETTLE goes back to APPROACH.)
    The foot is picked once per attempt, the nearer stance, and the network is told which.

    `net`: an ONNX path, or anything callable obs -> two outputs in [-1, 1] (training).
    `margin`: "lined up" means within this fraction of the tolerance. Training uses
    TRAIN_MARGIN; playing uses all of it, which leaves room for the real duck's lunge
    and sway (trained and played at 1.0, it froze next to the ball after near misses)."""

    def __init__(self, ends, net, params: KickbackParams = KickbackParams(), feet=("left", "right"), margin=1.0):
        super().__init__(ends, params, feet)
        self.margin = margin
        self.net = Network(net) if isinstance(net, (str, Path)) else net
        self.cmd = np.zeros(3, dtype=np.float32)
        self._attempt = None             # state_t0 of the APPROACH the foot was picked for
        self.ticks = 0

    def _approach(self, s: Snapshot):
        p = self.p
        if self._attempt != self.state_t0:
            self.start_attempt(s)
        e, h = self._errors(s)
        err = float(np.linalg.norm(e))
        self.plan = [np.array(s.duck_xy, float), self.stance_for(s.ball_xy, self.foot)]
        self.ticks += 1
        # Lined up: stand still, and after lined_up_s hand over to SETTLE.
        if abs(h) < p.heading_tol and err < self.margin * p.tolerance:
            if self._lined_up_t0 is None:
                self._lined_up_t0 = s.t
            if s.t - self._lined_up_t0 >= p.lined_up_s - 1e-9:
                self.set_state("SETTLE", s.t, f"lined up ({err * 1000:.0f} mm off), {self.foot} foot")
            return np.zeros(3, dtype=np.float32)
        self._lined_up_t0 = None
        if (self.ticks - 1) % DECIDE_EVERY == 0:
            self.cmd = to_command(self.net(observe(s, self.aim, self.spot_rel(), self.cmd)))
        return self.cmd

    def start_attempt(self, s: Snapshot):
        """A new APPROACH: pick the foot whose stance is nearer, and start deciding afresh."""
        self._attempt, self.ticks, self._lined_up_t0 = self.state_t0, 0, None
        self.foot = min(self.feet, key=lambda f: np.linalg.norm(self.stance_for(s.ball_xy, f) - s.duck_xy))

    def _align(self, s: Snapshot, dt_state):
        self.set_state("APPROACH", s.t, "back to the network")
        return Action()

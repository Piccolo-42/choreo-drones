"""Choreography engine: show file + audio features -> per-frame drone state."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from choreo.analysis import AudioFeatures
from choreo.formations import FORMATIONS
from choreo.motions import MOTIONS, apply_motion

COLORS: dict[str, tuple[float, float, float]] = {
    "white": (1.0, 0.85, 0.6),
    "red": (1.0, 0.12, 0.15),
}
ORDERS = (
    "outside_in",
    "inside_out",
    "bottom_up",
    "top_down",
    "index_asc",
    "index_desc",
)
ASSIGN_MODES = ("optimal", "line")


@dataclass(frozen=True)
class Params:
    rms_mix: float = 0.3  # how much loudness modulates brightness (0 = none)
    rms_window_s: float = 0.4  # loudness smoothing window
    pulse_tau_s: float = 0.12  # pulse decay time constant
    pulse_gain: float = 0.6  # brightness x(1 + gain) on a pulse
    pulse_size: float = 0.6  # marker size x(1 + size) on a pulse
    sweep_ramp_s: float = 0.12  # fade time of one drone in a sweep

DEFAULT_PARAMS = Params()

@dataclass(frozen=True)
class Sweep:
    mode: str  # "on" | "off"
    order: str  # one of ORDERS
    beats: float  # sweep length in beats
    subdiv: int = 1  # steps per beat


@dataclass(frozen=True)
class Segment:
    t: float
    shape: str
    motion: dict | None = None
    level: float = 1.0
    pulse_every: int = 0  # 0 = none, 1 = every beat, 2 = every 2nd beat
    sweep: Sweep | None = None
    fade: str | None = None  # "in" | "out", linear over the whole segment
    fade_from: float = 0.0  # fade "in" starts at this fraction of level
    light_cut: bool = False  # on morph: switch lights instantly, no crossfade
    assign: str = "optimal"  # "optimal" | "line" (keep left-to-right order)
    transition_beats: float | str | None = (
        None  # None = show default, "fill" = whole segment
    )
    color: tuple[float, float, float] = COLORS["white"]


@dataclass(frozen=True)
class Show:
    n_points: int
    transition_beats: float
    segments: list[Segment]


@dataclass(frozen=True)
class Frame:
    positions: np.ndarray  # (n, 2)
    brightness: np.ndarray  # (n,) in [0, 1]
    colors: np.ndarray  # (n, 3) RGB
    size_scale: float


def _parse_color(c: str | list[float]) -> tuple[float, float, float]:
    if isinstance(c, str):
        if c not in COLORS:
            raise ValueError(f"unknown color {c!r}; available: {sorted(COLORS)}")
        return COLORS[c]
    r, g, b = c
    return (float(r), float(g), float(b))


def _parse_segment(raw: dict) -> Segment:
    return Segment(
        t=float(raw["t"]),
        shape=raw["shape"],
        motion=raw.get("motion"),
        level=float(raw.get("level", 1.0)),
        pulse_every=int(raw.get("pulse_every", 0)),
        sweep=Sweep(**raw["sweep"]) if "sweep" in raw else None,
        fade=raw.get("fade"),
        fade_from=float(raw.get("fade_from", 0.0)),
        light_cut=bool(raw.get("light_cut", False)),
        assign=raw.get("assign", "optimal"),
        transition_beats=raw.get("transition_beats"),
        color=_parse_color(raw.get("color", "white")),
    )


def load_show(path: str | Path) -> Show:
    with open(path) as fh:
        raw = json.load(fh)
    return Show(
        n_points=int(raw["n_points"]),
        transition_beats=float(raw.get("transition_beats", 2)),
        segments=[_parse_segment(s) for s in raw["segments"]],
    )


def ease(u: float | np.ndarray) -> float | np.ndarray:
    """Smoothstep: 0 -> 0, 1 -> 1, zero velocity at both ends."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def assign(prev: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Reorder target so the total straight-line travel from prev is minimal."""
    _, cols = linear_sum_assignment(cdist(prev, target))
    return target[cols]


def assign_line(prev: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Map drones left-to-right onto target's path order (closed shapes only),
    trying every start point and both directions, keeping the cheapest."""
    n = len(prev)
    order = np.argsort(prev[:, 0], kind="stable")
    best, best_cost = target, np.inf
    for path in (target, target[::-1]):
        for shift in range(n):
            cand = np.roll(path, -shift, axis=0)
            cost = np.linalg.norm(prev[order] - cand, axis=1).sum()
            if cost < best_cost:
                best, best_cost = cand, cost
    out = np.empty_like(target)
    out[order] = best
    return out


def _needs_morph(prev: Segment, seg: Segment) -> bool:
    return seg.shape != prev.shape or (
        prev.motion is not None and seg.motion != prev.motion
    )


def _rank(pts: np.ndarray, order: str) -> np.ndarray:
    """rank[i] = position of drone i in the sweep order."""
    r = np.hypot(pts[:, 0], pts[:, 1])
    y = pts[:, 1]
    i = np.arange(len(pts))
    key = {
        "outside_in": -r,
        "inside_out": r,
        "bottom_up": y,
        "top_down": -y,
        "index_asc": i,
        "index_desc": -i,
    }[order]
    idx = np.argsort(key, kind="stable")
    rank = np.empty(len(pts), dtype=int)
    rank[idx] = np.arange(len(pts))
    return rank


class Timeline:
    def __init__(
        self,
        show: Show,
        features: AudioFeatures,
        use_assignment: bool = True,
        params: Params = DEFAULT_PARAMS,
    ) -> None:
        segs = show.segments
        if not segs:
            raise ValueError("show has no segments")
        if len(features.beat_times) < 2:
            raise ValueError("need at least 2 detected beats")

        self.show, self.f, self.p = show, features, params
        self.n = show.n_points
        self.beats = features.beat_times
        self.period = float(np.median(np.diff(self.beats)))
        self.duration = show.transition_beats * self.period  # default transition

        self.starts = np.array([segs[0].t] + [self._snap(s.t) for s in segs[1:]])
        if np.any(np.diff(self.starts) <= 0):
            raise ValueError(
                f"segment starts not increasing after beat snap: {np.round(self.starts, 2)}"
            )
        self.ends = np.append(self.starts[1:], features.duration)
        self._validate()
        self.trans = [self._transition_s(k) for k in range(len(segs))]

        self.morph = [False] + [
            _needs_morph(segs[k - 1], segs[k]) for k in range(1, len(segs))
        ]
        for k in range(1, len(segs)):
            if self.morph[k] and self.starts[k] + self.trans[k] > self.ends[k] + 1e-6:
                raise ValueError(
                    f"segment {k} at {self.starts[k]:.2f}s: transition "
                    f"({self.trans[k]:.2f}s) does not fit before the next segment"
                )

        # positions: base shape per segment, drone order kept across the show
        self.keyframes: list[np.ndarray] = []
        self.prev_end: list[np.ndarray | None] = [None] * len(segs)
        self.run_start = np.empty(len(segs))
        for k, seg in enumerate(segs):
            if k == 0:
                base = FORMATIONS[seg.shape](self.n)
                self.run_start[k] = self.starts[k]
            elif self.morph[k]:
                prev_end = self._shape_pos(k - 1, self.starts[k])
                base = FORMATIONS[seg.shape](self.n)
                if use_assignment:
                    fn = assign_line if seg.assign == "line" else assign
                    base = fn(prev_end, base)
                self.prev_end[k] = prev_end
                self.run_start[k] = self.starts[k]
            else:
                base = self.keyframes[k - 1]
                same_motion = seg.motion == segs[k - 1].motion
                self.run_start[k] = (
                    self.run_start[k - 1] if same_motion else self.starts[k]
                )
            self.keyframes.append(base)

        # lights: sweep order and beat-synced step times per segment
        self.ranks = [
            _rank(self.keyframes[k], s.sweep.order) if s.sweep else None
            for k, s in enumerate(segs)
        ]
        self.sweep_times: list[np.ndarray | None] = []
        for k, s in enumerate(segs):
            if s.sweep is None:
                self.sweep_times.append(None)
                continue
            steps = max(1, round(s.sweep.beats * s.sweep.subdiv))
            b0 = self._first_beat(k)
            self.sweep_times.append(
                np.array(
                    [self.beat_time(b0 + i / s.sweep.subdiv) for i in range(steps)]
                )
            )

        # loudness, smoothed and re-normalized
        hop = features.rms_times[1] - features.rms_times[0]
        w = max(1, round(params.rms_window_s / hop))
        smooth = np.convolve(features.rms, np.ones(w) / w, mode="same")
        self.rms_smooth = smooth / smooth.max() if smooth.max() > 0 else smooth

    # ---- validation and beat helpers ----

    def _validate(self) -> None:
        for i, s in enumerate(self.show.segments):
            if s.shape not in FORMATIONS:
                raise ValueError(
                    f"segment {i}: unknown shape {s.shape!r}; available: {sorted(FORMATIONS)}"
                )
            if s.motion is not None:
                mtype = s.motion.get("type")
                if mtype not in MOTIONS:
                    raise ValueError(
                        f"segment {i}: unknown motion {mtype!r}; available: {sorted(MOTIONS)}"
                    )
                if mtype == "scroll_wave" and s.shape != "waves":
                    raise ValueError(
                        f"segment {i}: scroll_wave only works with shape 'waves'"
                    )
            if s.sweep is not None and (
                s.sweep.order not in ORDERS or s.sweep.mode not in ("on", "off")
            ):
                raise ValueError(f"segment {i}: bad sweep {s.sweep}")
            if not 0.0 <= s.level <= 1.0:
                raise ValueError(f"segment {i}: level must be in [0, 1]")
            if s.fade not in (None, "in", "out"):
                raise ValueError(f"segment {i}: fade must be 'in' or 'out'")
            if not 0.0 <= s.fade_from <= 1.0:
                raise ValueError(f"segment {i}: fade_from must be in [0, 1]")
            if s.assign not in ASSIGN_MODES:
                raise ValueError(f"segment {i}: assign must be one of {ASSIGN_MODES}")
            tb = s.transition_beats
            valid_tb = (
                tb is None
                or tb == "fill"
                or (
                    isinstance(tb, (int, float)) and not isinstance(tb, bool) and tb > 0
                )
            )
            if not valid_tb:
                raise ValueError(f"segment {i}: transition_beats must be > 0 or 'fill'")

    def _transition_s(self, k: int) -> float:
        tb = self.show.segments[k].transition_beats
        if tb is None:
            return self.duration
        if tb == "fill":
            return float(self.ends[k] - self.starts[k])
        return float(tb) * self.period

    def _snap(self, t: float) -> float:
        return float(self.beats[np.argmin(np.abs(self.beats - t))])

    def _first_beat(self, k: int) -> int:
        """Index of the first beat at or after the start of segment k."""
        return int(np.searchsorted(self.beats, self.starts[k] - 1e-3))

    def beat_time(self, x: float) -> float:
        """Time of fractional beat index x; extrapolates with the median period."""
        last = len(self.beats) - 1
        if x < 0:
            return float(self.beats[0] + x * self.period)
        if x > last:
            return float(self.beats[-1] + (x - last) * self.period)
        return float(np.interp(x, np.arange(len(self.beats)), self.beats))

    def _pulse(self, k: int, t: float, every: int) -> float:
        """Decaying pulse on every `every`-th beat, counted from segment k's start."""
        if every <= 0:
            return 0.0
        b0 = self._first_beat(k)
        j = int(np.searchsorted(self.beats, t, side="right")) - 1
        if j < b0:
            return 0.0
        j = b0 + ((j - b0) // every) * every
        return float(np.exp(-(t - self.beats[j]) / self.p.pulse_tau_s))

    # ---- per-frame state ----

    def segment_index(self, t: float) -> int:
        return max(0, int(np.searchsorted(self.starts, t, side="right")) - 1)

    def _shape_pos(self, k: int, t: float) -> np.ndarray:
        seg = self.show.segments[k]
        beat_pulse = self._pulse(k, t, 1) if seg.motion else 0.0
        return apply_motion(
            seg.motion, self.keyframes[k], t - self.run_start[k], beat_pulse
        )

    def _light(self, k: int, t: float) -> tuple[np.ndarray, float]:
        seg = self.show.segments[k]
        b = np.full(self.n, seg.level, dtype=float)
        if seg.sweep is not None:
            times = self.sweep_times[k]
            group = math.ceil(self.n / len(times))
            step = self.ranks[k] // group
            ramp = np.clip((t - times[step]) / self.p.sweep_ramp_s, 0.0, 1.0)
            b = b * (ramp if seg.sweep.mode == "on" else 1.0 - ramp)
        if seg.fade is not None:
            frac = np.clip(
                (t - self.starts[k]) / (self.ends[k] - self.starts[k]), 0.0, 1.0
            )
            if seg.fade == "in":
                b = b * (seg.fade_from + (1.0 - seg.fade_from) * frac)
            else:
                b = b * (1.0 - frac)
        pulse = self._pulse(k, t, seg.pulse_every)
        return b * (1.0 + self.p.pulse_gain * pulse), pulse

    def state_at(self, t: float) -> Frame:
        segs = self.show.segments
        k = self.segment_index(t)
        pos = self._shape_pos(k, t)
        bright, pulse = self._light(k, t)
        color = np.asarray(segs[k].color)

        if self.morph[k]:
            e = float(ease((t - self.starts[k]) / self.trans[k]))
            if e < 1.0:
                prev = self.prev_end[k]
                assert prev is not None
                pos = prev + e * (pos - prev)
                if not segs[k].light_cut:
                    prev_b, prev_p = self._light(k - 1, t)
                    bright = prev_b + e * (bright - prev_b)
                    pulse = prev_p + e * (pulse - prev_p)
                    prev_c = np.asarray(segs[k - 1].color)
                    color = prev_c + e * (color - prev_c)

        loud = float(np.interp(t, self.f.rms_times, self.rms_smooth))
        bright = np.clip(
            bright * ((1.0 - self.p.rms_mix) + self.p.rms_mix * loud), 0.0, 1.0
        )
        return Frame(
            positions=pos,
            brightness=bright,
            colors=np.tile(color, (self.n, 1)),
            size_scale=1.0 + self.p.pulse_size * pulse,
        )

    def positions_at(self, t: float) -> np.ndarray:
        return self.state_at(t).positions

"""Choreography timeline: cues + audio features -> point positions at any time t."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from choreo.analysis import AudioFeatures
from choreo.formations import FORMATIONS


@dataclass(frozen=True)
class Cue:
    t: float          # seconds
    formation: str    # key in FORMATIONS


@dataclass(frozen=True)
class Show:
    n_points: int
    transition_beats: float
    cues: list[Cue]


def load_show(path: str | Path) -> Show:
    with open(path) as fh:
        raw = json.load(fh)
    cues = [Cue(t=float(c["t"]), formation=c["formation"]) for c in raw["cues"]]
    return Show(
        n_points=int(raw["n_points"]),
        transition_beats=float(raw["transition_beats"]),
        cues=cues,
    )


def ease(u: float | np.ndarray) -> float | np.ndarray:
    """Smoothstep: 0 -> 0, 1 -> 1, zero velocity at both ends."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def snap_to_beat(t: float, beat_times: np.ndarray) -> float:
    if len(beat_times) == 0:
        return t
    return float(beat_times[np.argmin(np.abs(beat_times - t))])


def assign(prev: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Reorder target so the total straight-line travel from prev is minimal."""
    _, cols = linear_sum_assignment(cdist(prev, target))
    return target[cols]


class Timeline:
    def __init__(
        self, show: Show, features: AudioFeatures, use_assignment: bool = True
    ) -> None:
        if not show.cues:
            raise ValueError("show has no cues")

        self.duration = show.transition_beats * 60.0 / features.tempo

        first, rest = show.cues[0], show.cues[1:]
        starts = [first.t] + [snap_to_beat(c.t, features.beat_times) for c in rest]
        self.starts = np.array(starts)

        gaps = np.diff(self.starts)
        if np.any(gaps < self.duration):
            i = int(np.argmax(gaps < self.duration)) + 1
            raise ValueError(
                f"cue {i} at {self.starts[i]:.2f}s starts before the previous "
                f"transition ({self.duration:.2f}s) ends, or cues are unsorted"
            )

        self.keyframes: list[np.ndarray] = []
        for cue in show.cues:
            if cue.formation not in FORMATIONS:
                raise ValueError(
                    f"unknown formation {cue.formation!r}; available: {sorted(FORMATIONS)}"
                )
            pts = FORMATIONS[cue.formation](show.n_points)
            if self.keyframes and use_assignment:
                pts = assign(self.keyframes[-1], pts)
            self.keyframes.append(pts)

    def positions_at(self, t: float) -> np.ndarray:
        k = int(np.searchsorted(self.starts, t, side="right")) - 1
        if k <= 0:
            return self.keyframes[0].copy()
        u = (t - self.starts[k]) / self.duration
        prev, target = self.keyframes[k - 1], self.keyframes[k]
        return prev + ease(u) * (target - prev)
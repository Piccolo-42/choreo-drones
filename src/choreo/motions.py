"""In-segment motions: (base points, local time, beat pulse, **params) -> points."""

from collections.abc import Callable

import numpy as np

from choreo.formations import semicircle_wave

Motion = Callable[..., np.ndarray]


def spin(
    pts: np.ndarray,
    t: float,
    pulse: float,
    deg_per_s: float = 20.0,
    delay_s: float = 0.0,
) -> np.ndarray:
    a = np.deg2rad(deg_per_s * max(0.0, t - delay_s))
    c, s = np.cos(a), np.sin(a)
    rot = np.array([[c, -s], [s, c]])
    return pts @ rot.T


def breathe(
    pts: np.ndarray, t: float, pulse: float, amount: float = 0.08
) -> np.ndarray:
    return pts * (1.0 + amount * pulse)


def scroll_wave(
    pts: np.ndarray, t: float, pulse: float, speed: float = 0.25
) -> np.ndarray:
    """Pattern travels in x; each drone keeps its x and only moves vertically."""
    x = pts[:, 0]
    return np.column_stack((x, semicircle_wave(x - speed * t)))


MOTIONS: dict[str, Motion] = {
    "spin": spin,
    "breathe": breathe,
    "scroll_wave": scroll_wave,
}


def apply_motion(
    spec: dict | None, pts: np.ndarray, t: float, pulse: float
) -> np.ndarray:
    if spec is None:
        return pts.copy()
    params = {key: val for key, val in spec.items() if key != "type"}
    return MOTIONS[spec["type"]](pts, t, pulse, **params)

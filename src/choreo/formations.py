"""2D formations: name -> (n, 2) point array, centered, within [-1, 1].

Vertical plane facing the audience: x = horizontal, y = altitude.
"""

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

Formation = Callable[[int], np.ndarray]


def _normalize(pts: np.ndarray) -> np.ndarray:
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2
    pts = pts - center
    scale = np.abs(pts).max()
    return pts / scale if scale > 0 else pts


def _polyline(vertices: ArrayLike, n: int, closed: bool = False) -> np.ndarray:
    """n points evenly spaced along a path; closed=True joins last to first."""
    v = np.asarray(vertices, dtype=float)
    if closed:
        v = np.vstack((v, v[:1]))
    seg_len = np.hypot(*np.diff(v, axis=0).T)
    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    s = np.linspace(0.0, cum[-1], n, endpoint=not closed)
    return np.column_stack((np.interp(s, cum, v[:, 0]), np.interp(s, cum, v[:, 1])))


def semicircle_wave(x: np.ndarray, n_arcs: int = 4, amp: float = 0.4) -> np.ndarray:
    """Alternating up/down half-ellipses; n_arcs arcs span x in [-1, 1]."""
    r = 1.0 / n_arcs  # arc half-width
    u = (x + 1.0) / (2 * r)  # position in arc units
    k = np.floor(u)  # arc index
    s = 2.0 * (u - k) - 1.0  # local coordinate in [-1, 1)
    sign = np.where(np.mod(k, 2) == 0, 1.0, -1.0)
    return sign * amp * np.sqrt(np.clip(1.0 - s * s, 0.0, None))


def circle(n: int) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return _normalize(np.column_stack((np.cos(theta), np.sin(theta))))


def grid(n: int) -> np.ndarray:
    rows = max(d for d in range(1, math.isqrt(n) + 1) if n % d == 0)
    cols = n // rows
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
    return _normalize(np.column_stack((xs.ravel(), ys.ravel())).astype(float))


def spiral(n: int, turns: float = 3.0) -> np.ndarray:
    theta = 2 * np.pi * turns * np.sqrt(np.linspace(0, 1, n))
    return _normalize(np.column_stack((theta * np.cos(theta), theta * np.sin(theta))))


def letter(n: int) -> np.ndarray:
    return _normalize(_polyline([(-0.6, 1.0), (0.0, -1.0), (0.6, 1.0)], n))


def upside_down_v(n: int) -> np.ndarray:
    return _normalize(_polyline([(-0.6, -1.0), (0.0, 1.0), (0.6, -1.0)], n))


def lips(n: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, 200)
    upper = (
        0.42 * (1 - x**2) ** 0.7 * (1 - 0.45 * np.exp(-((x / 0.18) ** 2)))
    )  # cupid's bow
    lower = -0.5 * (1 - x**2) ** 0.8
    outline = np.vstack(
        (
            np.column_stack((x, upper)),
            np.column_stack((x[::-1], lower[::-1]))[1:-1],  # skip shared corners
        )
    )
    n_inner = n // 5
    xi = np.linspace(-0.8, 0.8, 50)
    inner = np.column_stack(
        (xi, 0.05 * (1 - (xi / 0.8) ** 2) - 0.02)
    )  # line between lips
    pts = np.vstack(
        (_polyline(outline, n - n_inner, closed=True), _polyline(inner, n_inner))
    )
    return _normalize(pts)


def waves(n: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, n)
    return np.column_stack(
        (x, semicircle_wave(x))
    )  # not normalized: must match scroll_wave


def heart(n: int) -> np.ndarray:
    s = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    x = 16 * np.sin(s) ** 3
    y = 13 * np.cos(s) - 5 * np.cos(2 * s) - 2 * np.cos(3 * s) - np.cos(4 * s)
    return _normalize(_polyline(np.column_stack((x, y)), n, closed=True))


FORMATIONS: dict[str, Formation] = {
    "circle": circle,
    "grid": grid,
    "spiral": spiral,
    "letter": letter,
    "upside_down_v": upside_down_v,
    "lips": lips,
    "waves": waves,
    "heart": heart,
}

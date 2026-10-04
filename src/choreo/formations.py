"""2D formations: name -> (n, 2) point array, centered, within [-1, 1].

Vertical plane facing the audience: x = horizontal, y = altitude.
"""
import math
from collections.abc import Callable, Sequence

import numpy as np

Formation = Callable[[int], np.ndarray]


def _normalize(pts: np.ndarray) -> np.ndarray:
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2
    pts = pts - center
    scale = np.abs(pts).max()
    return pts / scale if scale > 0 else pts


def _polyline(vertices: Sequence[tuple[float, float]], n: int) -> np.ndarray:
    v = np.asarray(vertices, dtype=float)
    seg_len = np.hypot(*np.diff(v, axis=0).T)
    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    s = np.linspace(0.0, cum[-1], n)
    return np.column_stack((np.interp(s, cum, v[:, 0]), np.interp(s, cum, v[:, 1])))


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


_V = [(-0.6, 1.0), (0.0, -1.0), (0.6, 1.0)]


def letter(n: int) -> np.ndarray:
    return _normalize(_polyline(_V, n))


FORMATIONS: dict[str, Formation] = {
    "circle": circle,
    "grid": grid,
    "spiral": spiral,
    "letter": letter,
}

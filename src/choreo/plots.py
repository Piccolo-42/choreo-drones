"""Diagnostic plots (README assets)."""

from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np

from choreo.analysis import AudioFeatures
from choreo.formations import FORMATIONS
from choreo.timeline import Show, Timeline


def plot_features(track: str | Path, f: AudioFeatures, out: str | Path) -> None:
    y, sr = librosa.load(track)
    fig, ax = plt.subplots(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.4, label="waveform")
    ax.vlines(f.beat_times, -1, 1, color="tab:red", linewidth=0.6, label="beats")
    ax.plot(f.rms_times, f.rms, color="tab:orange", label="RMS (normalized)")
    ax.set(xlabel="time [s]", ylim=(-1, 1), title=f"Audio features — {f.tempo:.0f} BPM")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_formations(out: str | Path, n: int = 50) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (name, make) in zip(axes.flat, FORMATIONS.items()):
        pts = make(n)
        ax.scatter(pts[:, 0], pts[:, 1], s=15)
        ax.set(title=name, xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_transition(show: Show, f: AudioFeatures, k: int, out: str | Path) -> None:
    """Paths of transition k (cue k-1 -> cue k), naive order vs. assignment."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, use in zip(axes, (False, True)):
        tl = Timeline(show, f, use_assignment=use)
        a, b = tl.prev_end[k], tl.keyframes[k]
        for (x0, y0), (x1, y1) in zip(a, b):
            ax.plot([x0, x1], [y0, y1], color="0.6", lw=0.6, zorder=1)
        ax.scatter(
            a[:, 0],
            a[:, 1],
            s=12,
            color="0.3",
            label=show.segments[k - 1].shape,
            zorder=2,
        )
        ax.scatter(
            b[:, 0],
            b[:, 1],
            s=12,
            color="tab:blue",
            label=show.segments[k].shape,
            zorder=2,
        )
        total = np.linalg.norm(b - a, axis=1).sum()
        name = "assignment" if use else "naive order"
        ax.set(
            title=f"{name}: total travel {total:.1f}",
            xlim=(-1.1, 1.1),
            ylim=(-1.1, 1.1),
        )
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

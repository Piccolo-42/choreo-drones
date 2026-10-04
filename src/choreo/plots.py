"""Diagnostic plots (README assets)."""
from pathlib import Path

import librosa
import matplotlib.pyplot as plt

from choreo.analysis import AudioFeatures
from choreo.formations import FORMATIONS


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
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for ax, (name, make) in zip(axes.flat, FORMATIONS.items()):
        pts = make(n)
        ax.scatter(pts[:, 0], pts[:, 1], s=15)
        ax.set(title=name, xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
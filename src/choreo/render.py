"""Render a Timeline to MP4 with matplotlib + ffmpeg, then mux in the track audio."""
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from choreo.analysis import AudioFeatures
from choreo.timeline import Timeline

BASE_RGB = np.array([1.0, 0.85, 0.6])  # warm white


@dataclass(frozen=True)
class RenderSettings:
    fps: int = 30
    width: int = 1280
    height: int = 720
    preview_seconds: float | None = None  # render only the first N seconds
    rms_window_s: float = 0.4    # brightness smoothing window
    brightness_floor: float = 0.2
    pulse_tau_s: float = 0.12    # pulse decay time constant
    pulse_size: float = 0.8      # +80 % marker size at a beat
    pulse_glow: float = 0.25     # brightness boost at a beat
    base_size: float = 40.0      # marker area in points^2


def brightness_curve(
    f: AudioFeatures, t: np.ndarray, window_s: float, floor: float
) -> np.ndarray:
    hop_s = f.rms_times[1] - f.rms_times[0]
    w = max(1, round(window_s / hop_s))
    smooth = np.convolve(f.rms, np.ones(w) / w, mode="same")
    peak = smooth.max()
    if peak > 0:
        smooth = smooth / peak  # smoothing lowers peaks: re-normalize
    level = np.interp(t, f.rms_times, smooth)
    return floor + (1.0 - floor) * level


def pulse_curve(beat_times: np.ndarray, t: np.ndarray, tau: float) -> np.ndarray:
    if len(beat_times) == 0:
        return np.zeros_like(t)
    idx = np.searchsorted(beat_times, t, side="right") - 1
    dt = t - beat_times[np.maximum(idx, 0)]
    pulse = np.exp(-dt / tau)
    pulse[idx < 0] = 0.0  # before the first beat
    return pulse


def mux_audio(video: Path, track: str | Path, out: str | Path, duration: float) -> None:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video), "-i", str(track),
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-t", f"{duration:.3f}",
        str(out),
    ]
    subprocess.run(cmd, check=True)


def render(
    tl: Timeline,
    f: AudioFeatures,
    track: str | Path,
    out: str | Path,
    s: RenderSettings = RenderSettings(),
) -> None:
    duration = f.duration if s.preview_seconds is None else min(f.duration, s.preview_seconds)
    n_frames = int(np.ceil(duration * s.fps))
    t = np.arange(n_frames) / s.fps  # frame time from index, never accumulated

    pulse = pulse_curve(f.beat_times, t, s.pulse_tau_s)
    bright = brightness_curve(f, t, s.rms_window_s, s.brightness_floor)
    bright = np.clip(bright + s.pulse_glow * pulse, 0.0, 1.0)
    sizes = s.base_size * (1.0 + s.pulse_size * pulse)

    aspect = s.width / s.height
    fig, ax = plt.subplots(figsize=(s.width / 100, s.height / 100), dpi=100)
    fig.patch.set_facecolor("black")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(-1.2 * aspect, 1.2 * aspect), ylim=(-1.2, 1.2))
    ax.axis("off")

    pts = tl.positions_at(0.0)
    glow = ax.scatter(pts[:, 0], pts[:, 1], linewidths=0)
    core = ax.scatter(pts[:, 0], pts[:, 1], linewidths=0)

    out = Path(out)
    silent = out.with_suffix(".silent.mp4")
    writer = FFMpegWriter(fps=s.fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])

    with writer.saving(fig, str(silent), dpi=100):
        for i, ti in enumerate(t):
            pts = tl.positions_at(ti)
            rgb = BASE_RGB * bright[i]
            core.set_offsets(pts)
            core.set_sizes([sizes[i]])
            core.set_facecolors([(*rgb, 1.0)])
            glow.set_offsets(pts)
            glow.set_sizes([sizes[i] * 6])
            glow.set_facecolors([(*rgb, 0.15)])
            writer.grab_frame()
            if i % s.fps == 0:
                print(f"\rframe {i}/{n_frames}", end="", flush=True)
    print()
    plt.close(fig)

    mux_audio(silent, track, out, duration)
    silent.unlink()

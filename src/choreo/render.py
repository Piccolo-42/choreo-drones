"""Render a Timeline to MP4 with matplotlib + ffmpeg, then mux in the track audio."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from choreo.timeline import Timeline


@dataclass(frozen=True)
class RenderSettings:
    fps: int = 30
    width: int = 1280
    height: int = 720
    preview_start: float = 0.0  # seconds
    preview_seconds: float | None = None  # None = until the end
    base_size: float = 40.0  # marker area in points^2
    glow_scale: float = 6.0
    glow_alpha: float = 0.15

DEFAULT_SETTINGS = RenderSettings()

def mux_audio(
    video: Path, track: str | Path, out: Path, start: float, duration: float
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-ss",
        f"{start:.3f}",
        "-i",
        str(track),
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        f"{duration:.3f}",
        str(out),
    ]
    subprocess.run(cmd, check=True)


def render(
    tl: Timeline,
    track: str | Path,
    out: str | Path,
    s: RenderSettings = DEFAULT_SETTINGS,
) -> None:
    start = s.preview_start
    end = (
        tl.f.duration
        if s.preview_seconds is None
        else min(tl.f.duration, start + s.preview_seconds)
    )
    n_frames = int(np.ceil((end - start) * s.fps))

    aspect = s.width / s.height
    fig, ax = plt.subplots(figsize=(s.width / 100, s.height / 100), dpi=100)
    fig.patch.set_facecolor("black")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(-1.2 * aspect, 1.2 * aspect), ylim=(-1.2, 1.2))
    ax.axis("off")

    pts = tl.state_at(start).positions
    glow = ax.scatter(pts[:, 0], pts[:, 1], linewidths=0)
    core = ax.scatter(pts[:, 0], pts[:, 1], linewidths=0)

    out = Path(out)
    silent = out.with_suffix(".silent.mp4")
    writer = FFMpegWriter(
        fps=s.fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]
    )

    with writer.saving(fig, str(silent), dpi=100):
        for i in range(n_frames):
            fr = tl.state_at(
                start + i / s.fps
            )  # time from frame index, never accumulated
            rgb = fr.colors * fr.brightness[:, None]
            size = s.base_size * fr.size_scale
            core.set_offsets(fr.positions)
            core.set_facecolors(
                np.column_stack((rgb, np.clip(fr.brightness * 5.0, 0.0, 1.0)))
            )
            core.set_sizes([size])
            glow.set_offsets(fr.positions)
            glow.set_facecolors(np.column_stack((rgb, s.glow_alpha * fr.brightness)))
            glow.set_sizes([size * s.glow_scale])
            writer.grab_frame()
            if i % s.fps == 0:
                print(f"\rframe {i}/{n_frames}", end="", flush=True)
    print()
    plt.close(fig)

    mux_audio(silent, track, out, start, n_frames / s.fps)
    silent.unlink()

"""Audio analysis: track file -> time-stamped features."""

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass(frozen=True)
class AudioFeatures:
    duration: float  # seconds
    tempo: float  # BPM
    beat_times: np.ndarray  # (B,) seconds
    rms_times: np.ndarray  # (F,) seconds
    rms: np.ndarray  # (F,) normalized to [0, 1]


def analyze(
    path: str | Path,
    start_bpm: float = 80.0,
    tightness: float = 400.0,
    hop_length: int = 512,
) -> AudioFeatures:
    y, sr = librosa.load(path)
    tempo, beat_times = librosa.beat.beat_track(
        y=y,
        sr=sr,
        hop_length=hop_length,
        start_bpm=start_bpm,
        tightness=tightness,
        units="time",
    )
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
    peak = rms.max()
    if peak > 0:
        rms = rms / peak
    return AudioFeatures(
        duration=len(y) / sr,
        tempo=tempo.item(),
        beat_times=beat_times,
        rms_times=rms_times,
        rms=rms,
    )

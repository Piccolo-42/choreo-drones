# choreo-drones

A Python prototype that turns music into a 2D drone-show choreography. A 51-second section of a rock ballad is analyzed (beats and loudness), and a hand-written show file maps it to 50 drones: formations, motion within formations, and beat-synced light effects. The result is rendered to MP4 with the original audio.

Built in two days as a portfolio project while learning Python (my background is C/C++).

## Preview

![Intro preview](docs/preview.gif)

**▶ Full show with audio:** 

https://github.com/user-attachments/assets/36d5cf1d-fafb-4f5b-be7b-0e00eaf80440



## How it works

```
track.ogg ─► analysis ─► beats, loudness
cues.json ─► timeline ─► positions + lights per frame ─► render ─► show.mp4
```

- **Analysis:** librosa beat tracking, tuned for a live-tempo ballad (78 BPM), plus smoothed RMS loudness.
- **Timeline:** show segments snapped to beats; formation changes eased over a set number of beats; light pulses and drone-by-drone sweeps timed on beats and subdivisions.
- **Transitions:**
  - Hungarian assignment between formations: minimal travel, no crossing paths.
  - Order-preserving mode for the wave → heart morph.
- **Render:** matplotlib frames, encoded with ffmpeg, audio muxed in.

## Usage

Requires Python ≥ 3.10 and `ffmpeg` on PATH.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
choreo chorus   # 16 s preview (also: intro, verse, end)
choreo full     # out/show.mp4
choreo plots    # diagnostic plots in docs/
```

The choreography is defined in `assets/cues.json`.

## Limitations

A choreography and timing prototype, not a flight planner.

- 2D only (a vertical plane facing the audience).
- No drone dynamics: no speed or acceleration limits, wind, or positioning error.
- No collision avoidance or minimum-separation check. Optimal assignment avoids crossing paths, but drones can pass close; the order-preserving morph can cross.
- Beat tracking drifts for a few beats at the verse → chorus transition; parameters were tuned by ear.
- Section times are placed by hand, not detected.
- Validated with plots and renders; limited automated tests.

## Credits

- Music: **"Gone for Good" by Odysseas Williams**, used with permission ([original](https://www.youtube.com/watch?v=IAS7qQDWpFQ)). Not covered by the code license; see `assets/TRACK_LICENSE.md`.
- Code: MIT, see `LICENSE`.
- Development: V0 was built partly with an AI assistant, mainly to understand functions and Python idioms. The final version, with the creative choreography (shapes, timing, light design), was developed with more extensive AI help.

"""CLI: choreo [plots | intro | verse | chorus | end | full]"""

import argparse
from pathlib import Path

from choreo.analysis import analyze
from choreo.plots import plot_features, plot_formations, plot_transition
from choreo.render import RenderSettings, render
from choreo.timeline import Timeline, load_show

PREVIEWS = {
    "intro": RenderSettings(preview_start=0, preview_seconds=16, fps=15),
    "verse": RenderSettings(preview_start=12, preview_seconds=16, fps=15),
    "chorus": RenderSettings(preview_start=26, preview_seconds=16, fps=15),
    "end": RenderSettings(preview_start=36, preview_seconds=16, fps=15),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Music-driven 2D drone-show choreography")
    ap.add_argument(
        "mode", nargs="?", default="full", choices=["plots", *PREVIEWS, "full"]
    )
    ap.add_argument("--track", default="assets/track.ogg")
    ap.add_argument("--show", default="assets/cues.json")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    f = analyze(args.track)
    show = load_show(args.show)
    tl = Timeline(show, f)

    if args.mode == "plots":
        docs = Path("docs")
        docs.mkdir(exist_ok=True)
        plot_features(args.track, f, docs / "analysis.png")
        plot_formations(docs / "formations.png")
        for k, is_morph in enumerate(tl.morph):
            if is_morph:
                plot_transition(show, f, k, docs / f"transition_{k}.png")
        return

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    if args.mode == "full":
        render(tl, args.track, out / "show.mp4")
    else:
        render(tl, args.track, out / f"preview_{args.mode}.mp4", PREVIEWS[args.mode])


if __name__ == "__main__":
    main()

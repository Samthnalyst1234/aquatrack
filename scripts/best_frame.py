"""Save the frame with the most simultaneous detections as a demo image.

Reads a run's detections.csv, finds the busiest frame, and pulls that frame
from the run's annotated.mp4 — so the demo image shows the pipeline at its
fullest rather than a random moment.

Example:
    python scripts/best_frame.py outputs/flamingos --out docs/demo.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Save the busiest frame of a run as an image")
    ap.add_argument("run_dir", help="an outputs/<name> directory")
    ap.add_argument("--out", default=None, help="output image path (default: <run_dir>/best_frame.jpg)")
    args = ap.parse_args(argv)

    run = Path(args.run_dir)
    df = pd.read_csv(run / "detections.csv")
    if df.empty:
        raise SystemExit("No detections in this run.")

    counts = df.groupby("frame").size()
    best_frame = int(counts.idxmax())
    best_count = int(counts.max())

    cap = cv2.VideoCapture(str(run / "annotated.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read frame {best_frame} from annotated.mp4")

    out = Path(args.out) if args.out else run / "best_frame.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(f"Frame {best_frame} has the most detections ({best_count}) -> {out}")


if __name__ == "__main__":
    main()

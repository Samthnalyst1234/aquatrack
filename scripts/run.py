"""AquaTrack command-line entry point.

Example:
    python scripts/run.py data/clip.mp4 --out outputs/clip --depth
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aquatrack.behavior import cluster_behaviors, interaction_events
from aquatrack.pipeline import run_pipeline
from aquatrack.trajectories import per_track_metrics
from aquatrack import viz


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AquaTrack: spatiotemporal animal behaviour tracker")
    p.add_argument("video", help="path to input video")
    p.add_argument("--out", default="outputs/run1", help="output directory")
    p.add_argument("--model", default="yolo11n.pt", help="YOLO weights (.pt)")
    p.add_argument("--classes", default="animals",
                   help="'animals' (COCO animals), 'all', or comma-separated class ids")
    p.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    p.add_argument("--device", default=None, help="e.g. '0' for GPU, 'cpu' for CPU")
    p.add_argument("--depth", action="store_true", help="enable monocular depth estimation")
    p.add_argument("--no-video", dest="write_video", action="store_false",
                   help="skip writing the annotated video")
    p.add_argument("--max-frames", type=int, default=None, help="process at most N frames")
    p.add_argument("--clusters", type=int, default=3, help="k for behaviour clustering")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    out = Path(args.out)

    df, fps = run_pipeline(
        args.video, out,
        model_path=args.model, conf=args.conf, classes=args.classes,
        device=args.device, with_depth=args.depth,
        write_video=args.write_video, max_frames=args.max_frames,
    )

    n_frames = int(df["frame"].nunique()) if not df.empty else 0
    n_tracks = int(df["track_id"].nunique()) if not df.empty else 0
    print(f"Detections: {len(df)} across {n_frames} frames, {n_tracks} tracks")

    metrics = per_track_metrics(df, fps=fps)
    interactions = interaction_events(df)

    if not metrics.empty:
        metrics = cluster_behaviors(metrics, k=args.clusters)
        viz.plot_trajectories(df, out / "trajectories.png")
        viz.plot_speed(metrics, out / "speed.png")

    metrics.to_csv(out / "track_metrics.csv", index=False)
    interactions.to_csv(out / "interactions.csv", index=False)

    summary = dict(
        video=str(args.video),
        frames=n_frames,
        tracks=n_tracks,
        detections=int(len(df)),
        interaction_pairs=int(len(interactions)),
        depth=bool(args.depth),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Wrote outputs to", out)


if __name__ == "__main__":
    main()

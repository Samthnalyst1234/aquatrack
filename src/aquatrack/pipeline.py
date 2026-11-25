"""Single-pass video pipeline: detect + track (+ depth) and write annotated video."""
from __future__ import annotations

import colorsys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .models import COCO_ANIMALS, Detector


def _resolve_classes(classes):
    if classes == "animals":
        return COCO_ANIMALS
    if classes in (None, "all", ""):
        return None
    return [int(x) for x in str(classes).split(",")]


def _color(i: int):
    """Deterministic distinct BGR colour per track id."""
    h = (i * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def _draw(frame, dets, trails):
    img = frame.copy()
    for d in dets:
        tid = d["track_id"]
        color = _color(tid)
        x1, y1, x2, y2 = (int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"]))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img, f'{d["label"]} #{tid}', (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )
        pts = list(trails[tid])
        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], color, 2)
    return img


def run_pipeline(
    video_path,
    out_dir,
    model_path: str = "yolo11n.pt",
    conf: float = 0.25,
    classes="animals",
    tracker: str = "bytetrack.yaml",
    imgsz: int = 640,
    device=None,
    with_depth: bool = False,
    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf",
    write_video: bool = True,
    max_frames: int | None = None,
    trail_len: int = 30,
):
    """Process ``video_path`` and return (detections DataFrame, fps).

    Reads each frame once, runs detection+tracking, optionally monocular depth,
    writes an annotated video, and saves detections.csv in ``out_dir``.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_filter = _resolve_classes(classes)
    detector = Detector(
        model_path=model_path, conf=conf, classes=class_filter,
        tracker=tracker, imgsz=imgsz, device=device,
    )

    depth_est = None
    if with_depth:
        from .models import DepthEstimator
        depth_est = DepthEstimator(model_name=depth_model, device=device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), fourcc, fps, (width, height))

    trails = defaultdict(lambda: deque(maxlen=trail_len))
    rows: list[dict] = []
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            dets = detector.track_frame(frame)

            depth_map = None
            if depth_est is not None:
                from PIL import Image
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                depth_map = depth_est.depth_map(Image.fromarray(rgb))

            for d in dets:
                rec = dict(frame=frame_idx, time_s=frame_idx / fps, **d)
                if depth_map is not None:
                    cx = int(np.clip(d["cx"], 0, depth_map.shape[1] - 1))
                    cy = int(np.clip(d["cy"], 0, depth_map.shape[0] - 1))
                    rec["depth"] = float(depth_map[cy, cx])
                rows.append(rec)
                trails[d["track_id"]].append((int(d["cx"]), int(d["cy"])))

            if writer is not None:
                writer.write(_draw(frame, dets, trails))

            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    df = pd.DataFrame(rows)
    df.attrs["fps"] = fps
    df.to_csv(out_dir / "detections.csv", index=False)
    return df, fps

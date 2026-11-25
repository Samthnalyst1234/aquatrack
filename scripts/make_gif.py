"""Make a short, downscaled GIF from a video (e.g. an annotated run) for READMEs.

Example:
    python scripts/make_gif.py outputs/flamingos/annotated.mp4 --out docs/demo.gif
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Make a short GIF from a video")
    ap.add_argument("video", help="input video (e.g. an annotated.mp4)")
    ap.add_argument("--out", default="docs/demo.gif", help="output .gif path")
    ap.add_argument("--fps", type=int, default=8, help="GIF frame rate")
    ap.add_argument("--width", type=int, default=480, help="GIF width in px (keeps aspect)")
    ap.add_argument("--max-seconds", type=float, default=5.0, help="clip length to use")
    args = ap.parse_args(argv)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(src_fps / args.fps))
    max_frames = int(args.max_seconds * src_fps)

    frames: list[Image.Image] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or idx > max_frames:
            break
        if idx % step == 0:
            h, w = frame.shape[:2]
            new_h = int(h * args.width / w)
            frame = cv2.resize(frame, (args.width, new_h))
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        idx += 1
    cap.release()

    if not frames:
        raise SystemExit("No frames read from the video.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / args.fps),
        loop=0,
        optimize=True,
    )
    print(f"Wrote {out} ({len(frames)} frames at {args.width}px)")


if __name__ == "__main__":
    main()

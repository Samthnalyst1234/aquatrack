# AquaTrack

**A spatiotemporal pipeline for detecting, tracking and analysing animal behaviour in video.**

AquaTrack takes ordinary video of animals (fish, wildlife, anything) and turns it into
structured behavioural data: who is in the scene, where each individual moves over time,
how fast, how they interact, and a rough sense of depth. It is built from pretrained models,
so it runs on a single modest GPU without training anything from scratch.

> This is a personal learning and portfolio project, run on **public video data**. It is not
> production software and is not affiliated with any company. The goal is to demonstrate a
> working spatiotemporal computer-vision pipeline end to end.

## What it demonstrates

| Area | How AquaTrack shows it |
|---|---|
| Object detection | Pretrained YOLO detector, class-filtered to animals |
| Multi-object tracking | Persistent track IDs across frames via ByteTrack |
| Temporal modelling | Per-track trajectories, speed, displacement, path straightness over time |
| Depth / pseudo-3D | Monocular depth (Depth Anything V2) sampled at each animal, giving an approximate Z |
| Behavioural analysis | Proximity-based interaction events and unsupervised clustering of movement patterns |
| Real-world robustness | Works on noisy, uncontrolled footage; configurable confidence and class filters |

## How it works

```
video ─▶ detect + track (YOLO + ByteTrack) ─▶ per-frame detections with stable IDs
                                   │
                 (optional) monocular depth per frame ─▶ pseudo-Z at each animal
                                   │
                 trajectories + speed/displacement  ─▶ per-track metrics
                                   │
                 proximity over time ─▶ interaction events
                 movement features  ─▶ behaviour clusters
                                   │
                 annotated video + trajectory/speed charts + CSVs + summary.json
```

Everything runs in a single pass over the video, so each frame is read once for detection,
tracking, depth and annotation.

## Setup

Python 3.10+. A CUDA GPU is recommended for depth; detection/tracking also run on CPU.

```bash
git clone <your-repo-url> aquatrack
cd aquatrack
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

On Lightning AI (T4/L4), `torch` is already installed with CUDA, so `pip install -e .`
just adds the rest. On a fresh local machine without CUDA you may want to install the
matching `torch` build first from https://pytorch.org.

## Get a test video

Any short video of animals works. Good public sources:

- Pexels / Pixabay free stock video (search "fish", "aquarium", "birds", "herd").
- Roboflow Universe has public fish/animal video datasets.

Drop the file in `data/` (git-ignored). For fish specifically, the default COCO detector
has no "fish" class, so either use wildlife footage (COCO animals work out of the box) or
pass a fish-trained weights file with `--model`. See **Fish footage** below.

## Usage

```bash
# Basic run on wildlife footage (COCO animal classes, no depth, fast):
python scripts/run.py data/herd.mp4 --out outputs/herd

# Full run with depth estimation (needs GPU for reasonable speed):
python scripts/run.py data/aquarium.mp4 --out outputs/aquarium --depth

# Limit frames while testing, pick a device, custom confidence:
python scripts/run.py data/clip.mp4 --out outputs/test --max-frames 300 --device 0 --conf 0.3
```

### Outputs (in `--out`)

- `annotated.mp4` — video with boxes, track IDs and motion trails
- `detections.csv` — every detection: frame, time, track id, label, box, centre, (depth)
- `track_metrics.csv` — per animal: duration, distance, speed, straightness, behaviour cluster
- `interactions.csv` — pairs of animals that came close, and for how long
- `trajectories.png`, `speed.png` — charts
- `summary.json` — run summary

### Fish footage

COCO (the default YOLO training set) has no fish class. To track fish:

1. Grab a fish-trained YOLO weights file (e.g. from Roboflow Universe), or
2. Fine-tune `yolo11n.pt` on a small labelled fish dataset.

Then run with `--model path/to/fish.pt --classes all`.

## Honest limitations

- **Pseudo-3D, not true 3D.** Monocular depth gives a *relative* Z, not calibrated metric
  depth. Real biomass/3D work needs multi-camera calibration, which is out of scope here.
- **Tracking breaks under heavy occlusion** and with look-alike individuals; IDs can switch.
- **Behaviour clustering is unsupervised and heuristic**, a starting point, not a validated
  ethogram.
- **Depth is slow on CPU.** Use a GPU or `--max-frames` while iterating.

## Roadmap

- Multi-camera calibration + triangulation for true 3D positions
- A proper temporal model (sequence classifier) over trajectories instead of static features
- Fish-specific fine-tuned detector shipped with the repo
- Re-ID to reduce ID switches after occlusion

## License

MIT

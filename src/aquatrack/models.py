"""Model wrappers: a YOLO detector+tracker and a monocular depth estimator."""
from __future__ import annotations

import numpy as np

# COCO class ids that are animals (bird, cat, dog, horse, sheep, cow,
# elephant, bear, zebra, giraffe). Handy default for wildlife footage.
COCO_ANIMALS = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]


class Detector:
    """Thin wrapper over an Ultralytics YOLO model running in tracking mode.

    Call ``track_frame`` once per frame, in order. The underlying tracker keeps
    state between calls (``persist=True``) so track IDs stay stable over time.
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf: float = 0.25,
        classes=None,
        tracker: str = "bytetrack.yaml",
        imgsz: int = 640,
        device=None,
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf = conf
        self.classes = classes
        self.tracker = tracker
        self.imgsz = imgsz
        self.device = device
        self.names = self.model.names

    def track_frame(self, frame) -> list[dict]:
        """Return a list of detections for one BGR frame.

        Each detection is a dict with the track id, class, confidence, box
        corners and box centre.
        """
        result = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            classes=self.classes,
            tracker=self.tracker,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]

        dets: list[dict] = []
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return dets

        ids = boxes.id.int().cpu().tolist()
        clss = boxes.cls.int().cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        xyxy = boxes.xyxy.cpu().tolist()

        for tid, cls, cf, (x1, y1, x2, y2) in zip(ids, clss, confs, xyxy):
            dets.append(
                dict(
                    track_id=int(tid),
                    cls=int(cls),
                    label=self.names.get(int(cls), str(cls)),
                    conf=float(cf),
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                    cx=float((x1 + x2) / 2.0), cy=float((y1 + y2) / 2.0),
                )
            )
        return dets


class DepthEstimator:
    """Monocular relative-depth estimator (Depth Anything V2 by default)."""

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device=None,
    ):
        import torch
        from transformers import pipeline

        if device is None:
            hf_device = 0 if torch.cuda.is_available() else -1
        elif device in ("cpu", "-1", -1):
            hf_device = -1
        else:
            # ultralytics accepts "0"; transformers wants an int
            try:
                hf_device = int(device)
            except (TypeError, ValueError):
                hf_device = 0 if torch.cuda.is_available() else -1

        # token=False forces anonymous access, so a stale/expired HF token in
        # the environment can't break downloading this public model.
        self.pipe = pipeline(
            "depth-estimation", model=model_name, device=hf_device, token=False
        )

    def depth_map(self, pil_image) -> np.ndarray:
        """Return a float32 depth map (same HxW as the image). Larger = nearer,
        but values are *relative*, not metric."""
        out = self.pipe(pil_image)
        return np.array(out["depth"], dtype=np.float32)

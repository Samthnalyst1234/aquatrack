"""Turn per-frame detections into per-track trajectory metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def per_track_metrics(df: pd.DataFrame, fps: float | None = None) -> pd.DataFrame:
    """Aggregate detections into one row per tracked individual.

    Metrics: number of detections, duration, total path length, net
    displacement, straightness (net/total), mean and max speed, and mean
    relative depth if available.
    """
    if df.empty:
        return pd.DataFrame()

    fps = fps or df.attrs.get("fps", 25.0)
    records = []

    for tid, g in df.sort_values("frame").groupby("track_id"):
        cx = g["cx"].to_numpy()
        cy = g["cy"].to_numpy()
        frames = g["frame"].to_numpy()

        if len(frames) > 1:
            dx = np.diff(cx)
            dy = np.diff(cy)
            step = np.sqrt(dx ** 2 + dy ** 2)
            dt = np.diff(frames) / fps
            dt[dt == 0] = 1.0 / fps
            speeds = step / dt
            duration = (frames[-1] - frames[0]) / fps
            total_dist = float(step.sum())
            net_disp = float(np.hypot(cx[-1] - cx[0], cy[-1] - cy[0]))
            mean_speed = float(speeds.mean())
            max_speed = float(speeds.max())
        else:
            duration = total_dist = net_disp = mean_speed = max_speed = 0.0

        rec = dict(
            track_id=int(tid),
            label=g["label"].mode().iloc[0],
            n_detections=int(len(g)),
            duration_s=round(duration, 2),
            total_distance_px=round(total_dist, 1),
            net_displacement_px=round(net_disp, 1),
            straightness=round(net_disp / total_dist, 3) if total_dist > 0 else 0.0,
            mean_speed_px_s=round(mean_speed, 1),
            max_speed_px_s=round(max_speed, 1),
        )
        if "depth" in g.columns:
            rec["mean_depth"] = round(float(g["depth"].mean()), 3)

        records.append(rec)

    out = pd.DataFrame(records)
    return out.sort_values("n_detections", ascending=False).reset_index(drop=True)

"""Behavioural analysis: proximity interactions and movement-pattern clusters."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

_CLUSTER_FEATURES = ["mean_speed_px_s", "straightness", "total_distance_px"]


def interaction_events(
    df: pd.DataFrame, dist_thresh: float = 80.0, min_frames: int = 5
) -> pd.DataFrame:
    """Find pairs of individuals that stayed close.

    Two tracks "interact" in a frame when their centres are within
    ``dist_thresh`` pixels. Pairs close for at least ``min_frames`` frames are
    reported, with how many frames they were close and their minimum distance.
    """
    events = []
    for frame, g in df.groupby("frame"):
        recs = g[["track_id", "cx", "cy"]].to_dict("records")
        for a, b in combinations(recs, 2):
            dist = float(np.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]))
            if dist <= dist_thresh:
                lo, hi = sorted((a["track_id"], b["track_id"]))
                events.append((frame, lo, hi, dist))

    cols = ["track_a", "track_b", "n_frames_close", "min_dist_px"]
    if not events:
        return pd.DataFrame(columns=cols)

    e = pd.DataFrame(events, columns=["frame", "track_a", "track_b", "dist"])
    agg = (
        e.groupby(["track_a", "track_b"])
        .agg(n_frames_close=("frame", "nunique"), min_dist_px=("dist", "min"))
        .reset_index()
    )
    agg = agg[agg["n_frames_close"] >= min_frames].copy()
    agg["min_dist_px"] = agg["min_dist_px"].round(1)
    return agg.sort_values("n_frames_close", ascending=False).reset_index(drop=True)


def cluster_behaviors(metrics: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Add a ``behavior_cluster`` column by k-means over movement features.

    Unsupervised and heuristic: it groups individuals with similar speed, path
    straightness and distance travelled (e.g. "fast and straight" vs "slow and
    milling"). Not a validated ethogram, just a starting point.
    """
    m = metrics.copy()
    usable = m.dropna(subset=_CLUSTER_FEATURES)
    if len(usable) < k:
        m["behavior_cluster"] = 0
        return m

    from sklearn.cluster import KMeans

    X = usable[_CLUSTER_FEATURES].to_numpy(dtype=float)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)

    m["behavior_cluster"] = 0
    m.loc[usable.index, "behavior_cluster"] = labels
    return m

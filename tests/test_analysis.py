"""Tests for the analysis layer (no GPU / torch / ultralytics needed).

These exercise trajectory metrics and behaviour analysis on synthetic
detections, so the maths is verified independently of the vision models.
"""
import numpy as np
import pandas as pd

from aquatrack.behavior import cluster_behaviors, interaction_events
from aquatrack.trajectories import per_track_metrics


def _synthetic_detections():
    """Two tracks over 10 frames at 10 fps.

    Track 1 moves in a straight horizontal line at 10 px/frame.
    Track 2 sits almost still near track 1's start.
    """
    rows = []
    for f in range(10):
        rows.append(dict(frame=f, time_s=f / 10, track_id=1, cls=0, label="fish",
                         conf=0.9, x1=f * 10, y1=0, x2=f * 10 + 5, y2=5,
                         cx=f * 10 + 2.5, cy=2.5))
        rows.append(dict(frame=f, time_s=f / 10, track_id=2, cls=0, label="fish",
                         conf=0.9, x1=0, y1=0, x2=5, y2=5,
                         cx=2.5 + 0.1 * f, cy=2.5))
    df = pd.DataFrame(rows)
    df.attrs["fps"] = 10.0
    return df


def test_per_track_metrics_basic():
    df = _synthetic_detections()
    m = per_track_metrics(df, fps=10.0).set_index("track_id")

    # Track 1: 9 steps of 10 px over 0.9 s => ~100 px/s, perfectly straight.
    assert m.loc[1, "n_detections"] == 10
    assert m.loc[1, "mean_speed_px_s"] == 100.0
    assert m.loc[1, "straightness"] == 1.0
    # Track 2 barely moves and is far slower than track 1.
    assert m.loc[2, "mean_speed_px_s"] < m.loc[1, "mean_speed_px_s"]


def test_metrics_single_detection_is_safe():
    df = pd.DataFrame([dict(frame=0, time_s=0, track_id=9, cls=0, label="fish",
                            conf=0.9, x1=0, y1=0, x2=4, y2=4, cx=2, cy=2)])
    df.attrs["fps"] = 10.0
    m = per_track_metrics(df, fps=10.0).set_index("track_id")
    assert m.loc[9, "total_distance_px"] == 0.0
    assert m.loc[9, "mean_speed_px_s"] == 0.0


def test_interaction_events_detects_close_pair():
    df = _synthetic_detections()
    # Tracks start together, so early frames are "close".
    inter = interaction_events(df, dist_thresh=30, min_frames=3)
    assert len(inter) == 1
    assert set(inter.iloc[0][["track_a", "track_b"]]) == {1, 2}


def test_interaction_events_none_when_far():
    df = _synthetic_detections()
    inter = interaction_events(df, dist_thresh=1.0, min_frames=3)
    assert inter.empty


def test_cluster_behaviors_assigns_labels():
    df = _synthetic_detections()
    m = per_track_metrics(df, fps=10.0)
    clustered = cluster_behaviors(m, k=2)
    assert "behavior_cluster" in clustered.columns
    assert len(clustered) == len(m)

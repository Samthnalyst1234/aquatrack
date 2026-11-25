"""Static charts: trajectories and per-track speed."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_trajectories(df, out_path, top_n: int = 12):
    """Plot the paths of the most-seen tracks in image coordinates."""
    fig, ax = plt.subplots(figsize=(8, 6))
    top_ids = df["track_id"].value_counts().head(top_n).index
    for tid in top_ids:
        g = df[df["track_id"] == tid].sort_values("frame")
        ax.plot(g["cx"], g["cy"], lw=1.5, label=f"#{tid}")
    ax.invert_yaxis()  # image y grows downward
    ax.set_title("Animal trajectories")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_speed(metrics, out_path):
    """Bar chart of mean speed per track."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(metrics["track_id"].astype(str), metrics["mean_speed_px_s"])
    ax.set_title("Mean speed per track")
    ax.set_xlabel("track id")
    ax.set_ylabel("px / s")
    plt.xticks(rotation=90, fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

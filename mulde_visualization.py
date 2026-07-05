"""Shared MULDE custom-video visualization: thresholding, segments, and charts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


def format_timestamp(seconds: float) -> str:
    """Format seconds as M:SS.mmm or S.mmm for sub-minute clips."""
    if seconds < 0:
        seconds = 0.0
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    if minutes:
        return f"{minutes}:{remainder:06.3f}"
    return f"{remainder:.3f}s"


def log_likelihood_to_anomaly_score(log_likelihood: np.ndarray) -> np.ndarray:
    """Match training convention: higher score = more anomalous (negative log-likelihood)."""
    return (-np.asarray(log_likelihood, dtype=np.float64)).astype(np.float32)


def compute_anomaly_threshold(
    anomaly_scores: np.ndarray,
    method: Literal["mad", "percentile", "manual"] = "mad",
    *,
    percentile: float = 90.0,
    mad_k: float = 3.0,
    manual_threshold: float | None = None,
) -> float:
    """Return score threshold; frames with score strictly above this are anomalous."""
    scores = np.asarray(anomaly_scores, dtype=np.float64)
    if method == "manual":
        if manual_threshold is None:
            raise ValueError("manual_threshold is required when method='manual'")
        return float(manual_threshold)

    if method == "percentile":
        return float(np.percentile(scores, percentile))

    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    if mad < 1e-12:
        mad = float(np.std(scores))
    if mad < 1e-12:
        mad = 1e-6
    return median + mad_k * mad


def detect_anomaly_segments(
    is_anomaly: np.ndarray,
    fps: float,
    *,
    min_segment_frames: int = 1,
    merge_gap_frames: int = 0,
) -> list[dict]:
    """Find contiguous anomalous frame runs and convert them to time ranges."""
    flags = np.asarray(is_anomaly, dtype=bool)
    if flags.size == 0 or fps <= 0:
        return []

    segments: list[tuple[int, int]] = []
    in_segment = False
    start = 0
    for idx, flag in enumerate(flags):
        if flag and not in_segment:
            start = idx
            in_segment = True
        elif not flag and in_segment:
            segments.append((start, idx - 1))
            in_segment = False
    if in_segment:
        segments.append((start, len(flags) - 1))

    if merge_gap_frames > 0 and segments:
        merged = [segments[0]]
        for seg_start, seg_end in segments[1:]:
            prev_start, prev_end = merged[-1]
            if seg_start - prev_end - 1 <= merge_gap_frames:
                merged[-1] = (prev_start, seg_end)
            else:
                merged.append((seg_start, seg_end))
        segments = merged

    min_len = max(1, int(min_segment_frames))
    segments = [(s, e) for s, e in segments if e - s + 1 >= min_len]

    result = []
    for seg_start, seg_end in segments:
        start_sec = seg_start / fps
        end_sec = seg_end / fps
        duration_sec = (seg_end - seg_start + 1) / fps
        result.append(
            {
                "segment_id": len(result) + 1,
                "start_frame": int(seg_start),
                "end_frame": int(seg_end),
                "num_frames": int(seg_end - seg_start + 1),
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "duration_sec": round(duration_sec, 3),
                "start_time": format_timestamp(start_sec),
                "end_time": format_timestamp(end_sec),
                "time_range": f"{format_timestamp(start_sec)} – {format_timestamp(end_sec)}",
            }
        )
    return result


def build_results_dataframe(
    raw_log_likelihood: np.ndarray,
    smoothed_log_likelihood: np.ndarray,
    fps: float,
    *,
    threshold_method: str = "mad",
    threshold_percentile: float = 90.0,
    threshold_mad_k: float = 3.0,
    manual_threshold: float | None = None,
    min_segment_sec: float = 0.4,
    merge_gap_sec: float = 0.25,
) -> tuple[pd.DataFrame, float, list[dict]]:
    """Build per-frame results, threshold, and anomaly segments."""
    num_frames = len(raw_log_likelihood)
    raw_nll = log_likelihood_to_anomaly_score(raw_log_likelihood)
    smooth_nll = log_likelihood_to_anomaly_score(smoothed_log_likelihood)

    threshold = compute_anomaly_threshold(
        smooth_nll,
        method=threshold_method,  # type: ignore[arg-type]
        percentile=threshold_percentile,
        mad_k=threshold_mad_k,
        manual_threshold=manual_threshold,
    )
    is_anomaly = smooth_nll > threshold

    frame_indices = np.arange(num_frames, dtype=np.int64)
    timestamps = frame_indices / float(fps)

    df = pd.DataFrame(
        {
            "frame_index": frame_indices,
            "timestamp_sec": np.round(timestamps, 4),
            "timestamp": [format_timestamp(t) for t in timestamps],
            "raw_log_likelihood": raw_log_likelihood,
            "smoothed_log_likelihood": smoothed_log_likelihood,
            "anomaly_score_raw": raw_nll,
            "anomaly_score": smooth_nll,
            "is_anomaly": is_anomaly.astype(np.uint8),
            "classification": np.where(is_anomaly, "ANOMALY", "normal"),
        }
    )

    min_segment_frames = max(1, int(round(min_segment_sec * fps)))
    merge_gap_frames = max(0, int(round(merge_gap_sec * fps)))
    segments = detect_anomaly_segments(
        is_anomaly,
        fps,
        min_segment_frames=min_segment_frames,
        merge_gap_frames=merge_gap_frames,
    )
    return df, threshold, segments


def _shade_segments(ax, segments: list[dict], fps: float, *, use_time_axis: bool) -> None:
    for seg in segments:
        if use_time_axis:
            x0, x1 = seg["start_sec"], seg["end_sec"]
        else:
            x0, x1 = seg["start_frame"], seg["end_frame"]
        ax.axvspan(x0, x1, color="#ff6b6b", alpha=0.22, linewidth=0)


def _shade_manual_ranges(ax, ranges: list[tuple[int, int]], fps: float, *, use_time_axis: bool) -> None:
    for start, end in ranges:
        if use_time_axis:
            x0, x1 = start / fps, end / fps
        else:
            x0, x1 = start, end
        ax.axvspan(x0, x1, color="#ffd166", alpha=0.25, linewidth=0)


def generate_anomaly_dashboard(
    df: pd.DataFrame,
    segments: list[dict],
    *,
    video_name: str,
    fps: float,
    threshold: float,
    output_path: str | Path,
    manual_frame_ranges: list[tuple[int, int]] | None = None,
    threshold_method: str = "mad",
    model_name: str = "MULDE",
) -> Path:
    """Save a multi-panel anomaly report figure.

    ``model_name`` only controls the figure title; MULDE is the default and
    preserves the original behavior.
    """
    output_path = Path(output_path)
    times = df["timestamp_sec"].to_numpy()
    raw_nll = df["anomaly_score_raw"].to_numpy()
    smooth_nll = df["anomaly_score"].to_numpy()
    is_anomaly = df["is_anomaly"].to_numpy().astype(bool)
    duration_sec = float(times[-1]) if len(times) else 0.0

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 0.65, 1.2], hspace=0.32)
    ax_score = fig.add_subplot(gs[0])
    ax_strip = fig.add_subplot(gs[1], sharex=ax_score)
    ax_hist = fig.add_subplot(gs[2])

    _shade_segments(ax_score, segments, fps, use_time_axis=True)
    if manual_frame_ranges:
        _shade_manual_ranges(ax_score, manual_frame_ranges, fps, use_time_axis=True)

    ax_score.plot(times, raw_nll, color="#9aa5b1", alpha=0.55, linewidth=0.9, label="Raw anomaly score")
    ax_score.plot(times, smooth_nll, color="#1d4ed8", linewidth=1.8, label="Smoothed anomaly score")
    ax_score.axhline(
        threshold,
        color="#dc2626",
        linestyle="--",
        linewidth=1.4,
        label=f"Threshold ({threshold_method}) = {threshold:.3f}",
    )
    ax_score.set_ylabel("Anomaly score\n(−log-likelihood, ↑ = more anomalous)")
    ax_score.set_title(
        f"{model_name} Anomaly Report — {video_name}\n"
        f"{len(segments)} detected segment(s) · {duration_sec:.2f}s @ {fps:.2f} FPS",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax_score.grid(True, linestyle="--", alpha=0.35)
    ax_score.legend(loc="upper right", fontsize=9)

    strip_colors = np.where(is_anomaly, "#ef4444", "#22c55e")
    ax_strip.scatter(times, np.zeros_like(times), c=strip_colors, s=8, marker="|", linewidths=1.2)
    ax_strip.set_yticks([])
    ax_strip.set_ylabel("Frame\nlabel", rotation=0, labelpad=28, va="center")
    ax_strip.set_ylim(-0.5, 0.5)

    normal_patch = mpatches.Patch(color="#22c55e", label="Normal")
    anomaly_patch = mpatches.Patch(color="#ef4444", label="Anomaly")
    ax_strip.legend(handles=[normal_patch, anomaly_patch], loc="upper right", fontsize=8, ncol=2)

    ax_hist.hist(smooth_nll[~is_anomaly], bins=40, alpha=0.72, color="#22c55e", label="Normal frames", density=True)
    if is_anomaly.any():
        ax_hist.hist(smooth_nll[is_anomaly], bins=40, alpha=0.72, color="#ef4444", label="Anomaly frames", density=True)
    ax_hist.axvline(threshold, color="#dc2626", linestyle="--", linewidth=1.4)
    ax_hist.set_xlabel("Smoothed anomaly score")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Score distribution vs threshold", fontsize=11)
    ax_hist.grid(True, linestyle="--", alpha=0.3)
    ax_hist.legend(loc="upper right", fontsize=9)

    ax_score.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: format_timestamp(x)))
    ax_strip.set_xlabel("Time")
    fig.align_labels()
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_anomaly_artifacts(
    df: pd.DataFrame,
    segments: list[dict],
    *,
    output_dir: str | Path,
    video_name: str,
    fps: float,
    threshold: float,
    threshold_method: str,
    smooth_sigma: float,
    dashboard_path: Path,
    model_name: str = "MULDE",
) -> dict[str, Path]:
    """Write CSV, interval table, and summary JSON.

    The per-frame scores CSV is tagged with the lowercased ``model_name``
    (e.g. ``<video>_mulde_scores.csv``, ``<video>_stgnf_scores.csv``) so
    outputs from different models on the same video do not clobber each
    other. The interval/summary filenames remain shared.
    """
    output_dir = Path(output_dir)
    score_suffix = model_name.lower().replace("-", "")
    paths = {
        "scores_csv": output_dir / f"{video_name}_{score_suffix}_scores.csv",
        "intervals_csv": output_dir / f"{video_name}_anomaly_intervals.csv",
        "summary_json": output_dir / f"{video_name}_anomaly_summary.json",
        "dashboard_png": dashboard_path,
    }

    df.to_csv(paths["scores_csv"], index=False)

    intervals_df = pd.DataFrame(segments)
    if intervals_df.empty:
        intervals_df = pd.DataFrame(
            columns=[
                "segment_id",
                "start_frame",
                "end_frame",
                "num_frames",
                "start_sec",
                "end_sec",
                "duration_sec",
                "start_time",
                "end_time",
                "time_range",
            ]
        )
    intervals_df.to_csv(paths["intervals_csv"], index=False)

    summary = {
        "model_name": model_name,
        "video_name": video_name,
        "fps": float(fps),
        "num_frames": int(len(df)),
        "duration_sec": round(float(df["timestamp_sec"].iloc[-1]) if len(df) else 0.0, 3),
        "num_anomaly_frames": int(df["is_anomaly"].sum()),
        "anomaly_frame_ratio": round(float(df["is_anomaly"].mean()), 4),
        "num_anomaly_segments": len(segments),
        "threshold_method": threshold_method,
        "anomaly_threshold": round(float(threshold), 6),
        "smooth_sigma": float(smooth_sigma),
        "anomaly_intervals": segments,
        "artifacts": {key: str(path) for key, path in paths.items()},
    }
    with open(paths["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return paths


def print_anomaly_report(segments: list[dict], fps: float, threshold: float, threshold_method: str) -> None:
    """Print a human-readable summary to stdout."""
    print(f"\n{'=' * 60}")
    print("MULDE ANOMALY DETECTION SUMMARY")
    print(f"{'=' * 60}")
    print(f"FPS: {fps:.3f}  |  Threshold ({threshold_method}): {threshold:.4f}")
    print(f"Detected segments: {len(segments)}")
    if not segments:
        print("No anomaly segments detected above threshold.")
        return
    print(f"\n{'ID':<4} {'Frames':<14} {'Time range':<28} {'Duration':<10}")
    print("-" * 60)
    for seg in segments:
        frame_range = f"{seg['start_frame']}-{seg['end_frame']}"
        print(
            f"{seg['segment_id']:<4} {frame_range:<14} {seg['time_range']:<28} {seg['duration_sec']:.2f}s"
        )


def parse_frame_ranges(shading: str | None) -> list[tuple[int, int]]:
    """Parse '50-200,340-440' into [(50, 200), (340, 440)]."""
    if not shading:
        return []
    ranges: list[tuple[int, int]] = []
    for part in shading.split(","):
        part = part.strip()
        if not part:
            continue
        start_s, end_s = part.split("-", 1)
        ranges.append((int(start_s.strip()), int(end_s.strip())))
    return ranges

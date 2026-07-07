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


def print_anomaly_report(
    segments: list[dict],
    fps: float,
    threshold: float,
    threshold_method: str,
    *,
    model_name: str = "MULDE",
) -> None:
    """Print a human-readable summary to stdout.

    ``model_name`` only affects the header text; MULDE is the default and
    preserves the original behavior.
    """
    print(f"\n{'=' * 60}")
    print(f"{model_name.upper()} ANOMALY DETECTION SUMMARY")
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


def _render_static_score_graph(
    df: pd.DataFrame,
    segments: list[dict],
    *,
    fps: float,
    threshold: float,
    threshold_method: str,
    model_name: str,
    graph_width_px: int,
    graph_height_px: int,
) -> tuple[np.ndarray, float, float]:
    """Render the full score timeline once and return it as an RGB numpy array.

    The returned image is the static background for the bottom panel of the
    annotated video; the per-frame red playhead is drawn on top by the caller
    using ``cv2.line``.
    """
    times = df["timestamp_sec"].to_numpy()
    smooth_nll = df["anomaly_score"].to_numpy()
    raw_nll = df["anomaly_score_raw"].to_numpy()
    duration_sec = float(times[-1]) if len(times) else 0.0

    # Size the figure so the rendered pixels match the target panel size.
    dpi = 100.0
    fig_w = graph_width_px / dpi
    fig_h = graph_height_px / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_subplot(111)

    _shade_segments(ax, segments, fps, use_time_axis=True)

    ax.plot(times, raw_nll, color="#9aa5b1", alpha=0.55, linewidth=0.8, label="Raw")
    ax.plot(times, smooth_nll, color="#1d4ed8", linewidth=1.6, label="Smoothed")
    ax.axhline(
        threshold,
        color="#dc2626",
        linestyle="--",
        linewidth=1.2,
        label=f"Threshold ({threshold_method}) = {threshold:.3f}",
    )

    ax.set_xlim(0, duration_sec)
    y_pad = max((smooth_nll.max() - smooth_nll.min()) * 0.08, 1e-6)
    ax.set_ylim(min(smooth_nll.min(), raw_nll.min()) - y_pad,
                max(smooth_nll.max(), raw_nll.max()) + y_pad)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Anomaly score", fontsize=9)
    ax.set_title(
        f"{model_name.upper()} — {duration_sec:.1f}s @ {fps:.1f} FPS",
        fontsize=10, loc="left",
    )
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: format_timestamp(v)))
    fig.tight_layout(pad=0.4)

    fig.canvas.draw()
    # Get pixel coordinates of the x-axis limits (0 and duration_sec)
    x_start_px = float(ax.transData.transform((0.0, 0.0))[0])
    x_end_px = float(ax.transData.transform((duration_sec, 0.0))[0])

    graph_rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    graph_rgb = graph_rgba.reshape(int(fig_h * dpi), int(fig_w * dpi), 4)[:, :, :3]
    plt.close(fig)
    return graph_rgb, x_start_px, x_end_px


def detect_fps_for_image_dir(dir_path: str | Path) -> float:
    """Automatically detect the FPS of an image directory.
    
    1. Looks for neighboring video files with matching stems (e.g. .mp4, .avi).
    2. Analyzes file modification times for consistent intervals.
    3. Guesses based on dataset name (ShanghaiTech/Avenue = 25.0, UBnormal = 30.0).
    4. Falls back to 25.0 FPS.
    """
    import cv2
    import re
    dir_path = Path(dir_path)
    stem = dir_path.name
    
    # 1. Search for neighboring video files with matching stems
    video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv')
    parent = dir_path.parent
    
    for ext in video_extensions:
        candidate = parent / f"{stem}{ext}"
        if candidate.is_file():
            cap = cv2.VideoCapture(str(candidate))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps > 0:
                print(f"Automatically detected FPS {fps:.2f} from sibling video: {candidate.name}")
                return float(fps)
                
    if parent.parent.exists():
        for sibling_dir in parent.parent.iterdir():
            if sibling_dir.is_dir() and sibling_dir != parent:
                for ext in video_extensions:
                    candidate = sibling_dir / f"{stem}{ext}"
                    if candidate.is_file():
                        cap = cv2.VideoCapture(str(candidate))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        cap.release()
                        if fps > 0:
                            print(f"Automatically detected FPS {fps:.2f} from cousin video: {candidate}")
                            return float(fps)
                            
    # 2. Try file modification times
    try:
        img_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
        img_files = sorted([f for f in dir_path.iterdir() if f.suffix.lower() in img_extensions])
        if len(img_files) >= 5:
            mtimes = [f.stat().st_mtime for f in img_files[:10]]
            diffs = [mtimes[i+1] - mtimes[i] for i in range(len(mtimes)-1)]
            avg_diff = sum(diffs) / len(diffs)
            if 0.001 < avg_diff < 1.0:
                variance = sum((d - avg_diff)**2 for d in diffs) / len(diffs)
                if variance < 0.01:
                    calculated_fps = round(1.0 / avg_diff, 1)
                    print(f"Automatically calculated FPS {calculated_fps} from image file timestamps")
                    return float(calculated_fps)
    except Exception:
        pass

    # 3. Default fallback based on dataset keywords in the path
    path_lower = str(dir_path).lower()
    if "shanghaitech" in path_lower or "avenue" in path_lower:
        print("Defaulting to 25.0 FPS (standard for ShanghaiTech/Avenue datasets)")
        return 25.0
    elif "ubnormal" in path_lower:
        print("Defaulting to 30.0 FPS (standard for UBnormal dataset)")
        return 30.0
        
    print("Defaulting to standard 25.0 FPS")
    return 25.0


def generate_annotated_video(
    video_path: str | Path,
    df: pd.DataFrame,
    segments: list[dict],
    *,
    output_path: str | Path,
    fps: float | None = None,
    threshold: float,
    threshold_method: str = "mad",
    model_name: str = "MULDE",
    graph_height_ratio: float = 0.30,
) -> Path:
    """Build an MP4 with the original video on top and a scrolling score graph below.

    The score graph shows the full timeline with detected anomaly segments
    shaded pink and a red vertical playhead that advances frame-by-frame so the
    viewer can correlate what the model sees with what is happening in the
    video.

    Parameters
    ----------
    video_path
        Path to the input ``.mp4`` file.
    df
        Per-frame DataFrame from :func:`build_results_dataframe` (must contain
        ``timestamp_sec``, ``anomaly_score``, ``anomaly_score_raw``).
    segments
        Anomaly segments from :func:`build_results_dataframe`.
    output_path
        Destination ``.mp4`` path.
    fps
        Video frames-per-second (used for the output writer and playhead).
    threshold, threshold_method
        Forwarded to the static graph renderer for the threshold line.
    model_name
        Label shown in the graph title.
    graph_height_ratio
        Fraction of the composite frame height reserved for the graph panel
        (default ``0.30`` = 30%).
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fps is None:
        fps = detect_fps_for_image_dir(video_path)

    import cv2  # local import — only needed for video output, not score charts

    cap = None
    is_image_dir = video_path.is_dir()
    if is_image_dir:
        img_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
        img_files = sorted([f for f in video_path.iterdir() if f.suffix.lower() in img_extensions])
        num_frames = len(img_files)
        if num_frames == 0:
            raise FileNotFoundError(f"No images found in directory: {video_path}")
        first_img = cv2.imread(str(img_files[0]))
        if first_img is None:
            raise ValueError(f"Cannot read image: {img_files[0]}")
        video_height, video_width = first_img.shape[:2]
    else:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Graph panel sized to the video width; height proportional to the video
    # frame height so the composite looks balanced.
    graph_width_px = video_width
    graph_height_px = max(120, int(video_height * graph_height_ratio /
                                   (1.0 - graph_height_ratio)))
    composite_height = video_height + graph_height_px

    # Pre-render the static score graph once (the expensive matplotlib step).
    graph_rgb, x_start_px, x_end_px = _render_static_score_graph(
        df, segments,
        fps=fps, threshold=threshold, threshold_method=threshold_method,
        model_name=model_name,
        graph_width_px=graph_width_px, graph_height_px=graph_height_px,
    )
    graph_bgr = cv2.cvtColor(graph_rgb, cv2.COLOR_RGB2BGR)

    times = df["timestamp_sec"].to_numpy()
    duration_sec = float(times[-1]) if len(times) else 0.0

    # MP4V codec; fall back to MJPG if unavailable.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps,
                             (video_width, composite_height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps,
                                 (video_width, composite_height))

    try:
        for idx in range(num_frames):
            if is_image_dir:
                frame = cv2.imread(str(img_files[idx]))
                if frame is None:
                    frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
            else:
                ok, frame = cap.read()
                if not ok or frame is None:
                    # Pad with the last good frame if the decoder short-reads.
                    frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)

            # Draw the moving red playhead on a copy of the static graph.
            graph_frame = graph_bgr.copy()
            if idx < len(times):
                t = times[idx]
            else:
                t = duration_sec
            
            if duration_sec > 0:
                px = int(x_start_px + (t / duration_sec) * (x_end_px - x_start_px))
            else:
                px = int(x_start_px)
            px = max(0, min(graph_width_px - 1, px))
            
            cv2.line(graph_frame, (px, 0), (px, graph_height_px),
                     (0, 0, 255), 2)  # BGR red

            composite = np.vstack([frame, graph_frame])
            writer.write(composite)

            if (idx + 1) % 200 == 0 or idx == num_frames - 1:
                print(f"  annotated video: {idx + 1}/{num_frames} frames", flush=True)
    finally:
        if cap is not None:
            cap.release()
        writer.release()

    print(f"Saved annotated video: {output_path}")
    return output_path

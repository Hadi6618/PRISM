"""Create frame-level manual ground truth for a custom video.

The output uses the same basic conventions as PRISM:

* frame indices are zero-based;
* ``label=1`` means abnormal/anomalous and ``label=0`` means normal;
* the CSV contains one row for every decoded video frame.

Examples::

    # Explosion from 2:34 through 2:36 (time ranges are inclusive).
    python Utils/manual_video_ground_truth.py \\
        --video /content/drive/MyDrive/videos/explosion.mp4 \\
        --time-ranges "2:34-2:36" \\
        --output-dir /content/drive/MyDrive/videos/explosion_ground_truth

    # Several abnormal intervals, using explicit zero-based frame indices.
    python Utils/manual_video_ground_truth.py \\
        --video explosion.mp4 \\
        --frame-ranges "750-810,1200-1260" \\
        --output-dir ground_truth \\
        --extract-frames

The utility intentionally does not inspect the model scores or choose fusion
weights. It only records the manually supplied labels. This keeps annotation
separate from model evaluation and makes the labels reusable for STG-NF,
MULDE, and PRISM.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np


_TIME_RE = re.compile(r"^(?:(?P<hours>\d+):)?(?P<minutes>\d{1,2}):(?P<seconds>\d{1,2}(?:\.\d+)?)$")


@dataclass(frozen=True)
class FrameRange:
    """Inclusive zero-based frame interval."""

    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            raise ValueError(f"Invalid frame range: {self.start_frame}-{self.end_frame}")


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float
    reported_frame_count: int


def parse_timestamp(value: str) -> float:
    """Parse ``SS``, ``M:SS`` or ``H:MM:SS`` into seconds."""
    value = value.strip()
    try:
        # A plain number is seconds, e.g. ``154.5``.
        seconds = float(value)
        if seconds < 0:
            raise ValueError
        return seconds
    except ValueError:
        pass

    match = _TIME_RE.fullmatch(value)
    if match is None:
        raise ValueError(
            f"Invalid timestamp {value!r}; use seconds, M:SS, or H:MM:SS."
        )
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes"))
    seconds = float(match.group("seconds"))
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Invalid timestamp {value!r}.")
    return hours * 3600.0 + minutes * 60.0 + seconds


def _split_ranges(value: Optional[Sequence[str]]) -> list[str]:
    """Support both repeated arguments and comma-separated ranges."""
    result: list[str] = []
    for item in value or []:
        result.extend(part.strip() for part in item.split(",") if part.strip())
    return result


def _split_range(value: str) -> tuple[str, str]:
    # A hyphen is the only supported separator. It also avoids confusing the
    # colon in timestamps such as 2:34-2:36.
    parts = value.split("-", 1)
    if len(parts) != 2 or not all(part.strip() for part in parts):
        raise ValueError(f"Invalid range {value!r}; expected START-END.")
    return parts[0].strip(), parts[1].strip()


def parse_time_ranges(values: Optional[Sequence[str]], fps: float) -> list[FrameRange]:
    """Convert inclusive time intervals to inclusive zero-based frame ranges.

    The start is rounded down and the end is rounded up minus one frame, so a
    range ``2:34-2:36`` covers frames whose timestamps lie in
    ``[154, 156)`` seconds. A range ending exactly on a frame boundary does
    not accidentally include the next frame.
    """
    if fps <= 0:
        raise ValueError("FPS must be positive before parsing time ranges.")
    ranges: list[FrameRange] = []
    for raw in _split_ranges(values):
        start_s, end_s = _split_range(raw)
        start_sec = parse_timestamp(start_s)
        end_sec = parse_timestamp(end_s)
        if end_sec <= start_sec:
            raise ValueError(f"Time range must have end > start: {raw!r}")
        start = int(np.floor(start_sec * fps + 1e-9))
        end = int(np.ceil(end_sec * fps - 1e-9)) - 1
        ranges.append(FrameRange(start, max(start, end)))
    return ranges


def parse_frame_ranges(values: Optional[Sequence[str]]) -> list[FrameRange]:
    """Parse inclusive zero-based frame intervals."""
    ranges: list[FrameRange] = []
    for raw in _split_ranges(values):
        start, end = _split_range(raw)
        try:
            frame_range = FrameRange(int(start), int(end))
        except ValueError as exc:
            raise ValueError(
                f"Invalid frame range {raw!r}; frame numbers must be integers."
            ) from exc
        ranges.append(frame_range)
    return ranges


def merge_ranges(ranges: Iterable[FrameRange]) -> list[FrameRange]:
    """Sort and merge overlapping or directly adjacent ranges."""
    ordered = sorted(ranges, key=lambda item: (item.start_frame, item.end_frame))
    merged: list[FrameRange] = []
    for current in ordered:
        if not merged or current.start_frame > merged[-1].end_frame + 1:
            merged.append(current)
        else:
            merged[-1] = FrameRange(
                merged[-1].start_frame,
                max(merged[-1].end_frame, current.end_frame),
            )
    return merged


def inspect_video(video_path: Path, *, allow_invalid_fps: bool = False) -> VideoInfo:
    """Read metadata and count decoded frames exactly once."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    reported_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    decoded_count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        decoded_count += 1
    cap.release()

    if not np.isfinite(fps) or fps <= 0:
        if not allow_invalid_fps:
            raise RuntimeError(
                "The video decoder did not report a valid FPS. "
                "Convert the video to a constant-frame-rate file first, or provide "
                "a valid FPS manually with --fps."
            )
        fps = 0.0
    if decoded_count == 0:
        raise RuntimeError(f"The video contains no decodable frames: {video_path}")

    return VideoInfo(
        fps=fps,
        frame_count=decoded_count,
        width=width,
        height=height,
        duration_sec=decoded_count / fps if fps > 0 else 0.0,
        reported_frame_count=reported_count,
    )


def _label_array(frame_count: int, ranges: Sequence[FrameRange]) -> np.ndarray:
    labels = np.zeros(frame_count, dtype=np.uint8)
    for frame_range in ranges:
        start = max(0, frame_range.start_frame)
        end = min(frame_count - 1, frame_range.end_frame)
        if start <= end:
            labels[start : end + 1] = 1
    return labels


def write_ground_truth(
    video_path: Path,
    output_dir: Path,
    video_id: str,
    info: VideoInfo,
    ranges: Sequence[FrameRange],
    *,
    extract_frames: bool = False,
) -> dict[str, Path]:
    """Write CSV, JSON metadata, and a reusable pickle label payload."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = _label_array(info.frame_count, ranges)
    frame_indices = np.arange(info.frame_count, dtype=np.int64)
    frame_dir = output_dir / "frames"
    if extract_frames:
        frame_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{video_id}_ground_truth.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["frame_index", "timestamp_sec", "label"]
        if extract_frames:
            fieldnames.append("frame_path")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        cap = cv2.VideoCapture(str(video_path)) if extract_frames else None
        if cap is not None and not cap.isOpened():
            raise RuntimeError(f"Could not reopen video for frame extraction: {video_path}")
        try:
            for frame_index in frame_indices:
                row = {
                    "frame_index": int(frame_index),
                    "timestamp_sec": f"{float(frame_index) / info.fps:.6f}",
                    "label": int(labels[frame_index]),
                }
                if cap is not None:
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(
                            f"Frame extraction stopped at frame {int(frame_index)}."
                        )
                    frame_path = frame_dir / f"frame_{int(frame_index):06d}.jpg"
                    if not cv2.imwrite(str(frame_path), frame):
                        raise RuntimeError(f"Could not write frame: {frame_path}")
                    row["frame_path"] = str(frame_path)
                writer.writerow(row)
        finally:
            if cap is not None:
                cap.release()

    payload = {
        "video_path": str(video_path),
        "video_id": video_id,
        "fps": info.fps,
        "frame_count": info.frame_count,
        "duration_sec": info.duration_sec,
        "frame_index_base": 0,
        "label_convention": "1=abnormal/anomaly, 0=normal",
        "anomaly_ranges": [asdict(item) for item in ranges],
        "ground_truth": {
            video_id: {
                "frame_indices": frame_indices,
                "labels": labels,
            }
        },
    }
    pkl_path = output_dir / f"{video_id}_ground_truth.pkl"
    with pkl_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    metadata_path = output_dir / f"{video_id}_ground_truth.json"
    metadata = dict(payload)
    metadata["ground_truth"] = {
        video_id: {
            "frame_indices": frame_indices.tolist(),
            "labels": labels.tolist(),
        }
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    paths = {"csv": csv_path, "pickle": pkl_path, "json": metadata_path}
    if extract_frames:
        paths["frames_dir"] = frame_dir
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Input video path.")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory for generated files."
    )
    parser.add_argument(
        "--video-id",
        default=None,
        help="ID used in the output payload (default: input filename without extension).",
    )
    parser.add_argument(
        "--time-ranges",
        action="append",
        default=[],
        metavar="START-END",
        help="Abnormal time ranges, e.g. '2:34-2:36'. Repeat or comma-separate.",
    )
    parser.add_argument(
        "--frame-ranges",
        action="append",
        default=[],
        metavar="START-END",
        help="Abnormal inclusive zero-based frame ranges, e.g. '750-810'.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional FPS override for unusual/variable-frame-rate videos. "
        "By default FPS is read from the video metadata.",
    )
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help="Also save every decoded frame as frames/frame_XXXXXX.jpg. "
        "This can require substantial disk space.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        raise SystemExit(f"Video does not exist: {video_path}")
    if not args.time_ranges and not args.frame_ranges:
        raise SystemExit("Provide at least one --time-ranges or --frame-ranges value.")

    info = inspect_video(video_path, allow_invalid_fps=args.fps is not None)
    if args.fps is not None:
        if not np.isfinite(args.fps) or args.fps <= 0:
            raise SystemExit("--fps must be a positive number.")
        info = VideoInfo(
            fps=float(args.fps),
            frame_count=info.frame_count,
            width=info.width,
            height=info.height,
            duration_sec=info.frame_count / float(args.fps),
            reported_frame_count=info.reported_frame_count,
        )

    ranges = merge_ranges(
        parse_time_ranges(args.time_ranges, info.fps)
        + parse_frame_ranges(args.frame_ranges)
    )
    video_id = args.video_id or video_path.stem
    paths = write_ground_truth(
        video_path,
        args.output_dir.expanduser().resolve(),
        video_id,
        info,
        ranges,
        extract_frames=args.extract_frames,
    )

    print(f"Video:             {video_path}")
    print(f"FPS:               {info.fps:.6f}")
    print(f"Decoded frames:    {info.frame_count}")
    print(f"Duration:          {info.duration_sec:.3f} seconds")
    if info.reported_frame_count != info.frame_count:
        print(
            f"Warning: decoder metadata reported {info.reported_frame_count} frames, "
            f"but {info.frame_count} were decoded; labels use decoded frames."
        )
    print("Abnormal ranges:   " + ", ".join(
        f"{item.start_frame}-{item.end_frame}" for item in ranges
    ))
    print(f"Abnormal frames:   {int(_label_array(info.frame_count, ranges).sum())}")
    for name, path in paths.items():
        print(f"Wrote {name}:       {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

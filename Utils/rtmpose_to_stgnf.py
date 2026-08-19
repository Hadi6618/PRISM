"""Convert modern top-down pose estimators (RTMPose / MMPose) into STG-NF's
tracked-person JSON schema.

STG-NF (`dataset.py::gen_dataset` -> `pose_utils.py::single_pose_dict2np`)
expects, for every video clip, one JSON file whose name ends in
``tracked_person.json`` and whose first two underscore-separated tokens are the
``scene_id`` and ``clip_id`` (e.g. ``01_0001_*_tracked_person.json``). The body
is a dict:

    {
      "<person_id>": {
        "<frame_key>": {"keypoints": [x, y, c, ... x17, y17, c17],  # flat 17*3
                        "scores":    [c1, ..., c17]},               # 17 confidences
        ...
      },
      ...
    }

Keypoints are standard COCO-17 order, which is exactly what RTMPose/MMPose
``coco`` outputs, and exactly what STG-NF's ``keypoints17_to_coco18`` consumes
(the 17->18 neck insertion + reorder happens downstream, so we must NOT reorder
here).

This module only does the schema conversion; it imports nothing heavier than
``json``/``numpy`` so it can be unit-tested without torch or any pose library.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

# Standard COCO-17 order (matches MMPose ``coco`` and AlphaPose COCO output).
COCO17_KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
NUM_KEYPOINTS = 17


def _frame_key(frame_index, num_digits=4):
    """Normalise a frame index to a zero-padded string key.

    ``num_digits=4`` matches the PRISM notebook's STG-NF pose cell
    (``.zfill(4)``), which guarantees lexicographic order == numeric order even
    past frame 999.
    """
    return str(int(frame_index)).zfill(num_digits)


def rtmpose_to_tracked_person(
    keypoints: Sequence[Sequence[Sequence[float]]],
    keypoint_scores: Sequence[Sequence[float]],
    track_ids: Sequence[int],
    frame_indices: Sequence[int],
    num_digits: int = 4,
) -> Dict[str, Dict[str, Dict[str, list]]]:
    """Convert RTMPose top-down outputs into the STG-NF tracked-person dict.

    Parameters
    ----------
    keypoints:
        Shape ``[N, 17, 2]`` -- ``pred_instances.keypoints`` from
        ``mmpose.apis.inference_topdown`` (already in image pixel space).
    keypoint_scores:
        Shape ``[N, 17]`` -- ``pred_instances.keypoint_scores``.
    track_ids:
        Shape ``[N]`` -- per-instance track id (int). Instances sharing a track
        id are grouped into one person across frames.
    frame_indices:
        Shape ``[N]`` -- 0-based frame index for each instance.
    num_digits:
        Zero-padding width for frame keys (default 4).

    Returns
    -------
    The ``{person_id: {frame_key: {"keypoints": [...], "scores": [...]}}}`` dict.
    """
    keypoints = np.asarray(keypoints, dtype=np.float64)
    keypoint_scores = np.asarray(keypoint_scores, dtype=np.float64)
    track_ids = np.asarray(track_ids)
    frame_indices = np.asarray(frame_indices)

    if keypoints.ndim != 3 or keypoints.shape[1] != NUM_KEYPOINTS or keypoints.shape[2] != 2:
        raise ValueError(f"keypoints must be [N, {NUM_KEYPOINTS}, 2], got {keypoints.shape}")
    if keypoint_scores.shape != keypoints.shape[:2]:
        raise ValueError(
            f"keypoint_scores shape {keypoint_scores.shape} != keypoints[:2] {keypoints.shape[:2]}"
        )
    if not (keypoints.shape[0] == keypoint_scores.shape[0] == track_ids.shape[0] == frame_indices.shape[0]):
        raise ValueError("keypoints/keypoint_scores/track_ids/frame_indices must have equal length")

    tracked: Dict[str, Dict[str, Dict[str, list]]] = {}
    for i in range(keypoints.shape[0]):
        pid = str(int(track_ids[i]))
        fk = _frame_key(frame_indices[i], num_digits)

        kp_flat: List[float] = []
        for (x, y), c in zip(keypoints[i], keypoint_scores[i]):
            kp_flat.extend([float(x), float(y), float(c)])

        tracked.setdefault(pid, {})[fk] = {
            "keypoints": kp_flat,
            "scores": [float(c) for c in keypoint_scores[i]],
        }
    return tracked


def items_to_tracked_person(
    items: Iterable[dict],
    num_digits: int = 4,
) -> Dict[str, Dict[str, Dict[str, list]]]:
    """Convert an AlphaPose-style list of per-frame detections to tracked-person.

    Each item is a dict with ``idx`` (person/track id), ``image_id`` (frame
    name such as ``frame_0000.jpg`` or ``0.jpg``), ``keypoints`` (flat
    ``[x, y, c] * 17`` list) and ``score`` (17 confidences). This mirrors the
    conversion the PRISM notebook already does for AlphaPose, so a RTMPose
    exporter that emits the same list format can reuse it unchanged.
    """
    tracked: Dict[str, Dict[str, Dict[str, list]]] = {}
    for item in items:
        pid = str(item["idx"])
        stem = Path(str(item["image_id"])).stem
        digits = "".join(ch for ch in stem[::-1] if ch.isdigit())[::-1]
        fk = (digits or stem).zfill(num_digits)
        tracked.setdefault(pid, {})[fk] = {
            "keypoints": list(item["keypoints"]),
            "scores": list(item["score"]),
        }
    return tracked


def write_tracked_person(tracked: dict, path: str | Path) -> Path:
    """Serialize a tracked-person dict to JSON (STG-NF reads it back with ``json.load``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tracked, f)
    return path


def load_tracked_person(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


__all__ = [
    "COCO17_KEYPOINT_NAMES",
    "NUM_KEYPOINTS",
    "rtmpose_to_tracked_person",
    "items_to_tracked_person",
    "write_tracked_person",
    "load_tracked_person",
]

"""Per-video alignment of STG-NF and MULDE score streams.

The two streams come from independent pipelines with different conventions:

* **Video IDs** — STG-NF uses ``01_0021`` (scene_clip), MULDE uses the bare
  clip index ``21``. :func:`_build_video_id_map` auto-detects and remaps.
* **Frame indices** — STG-NF is 0-based, MULDE is 1-based on some exports.
  :func:`align_per_video` supports an explicit ``stgnf_frame_offset`` and
  can auto-detect it by maximising STG-NF's standalone Micro AUC.
* **Score polarity** — STG-NF's original repo reports *normality* scores
  on ShanghaiTech, while the fusion code expects *anomaly* scores.
  ``stgnf_score_mode="auto"`` tests both polarities and keeps the better one.
* **Labels** — one stream may use ``0=anomaly`` while the other uses
  ``1=anomaly``. :func:`_check_label_inversion` warns when this happens
  and the code falls back to MULDE's labels (1=anomaly convention).

The output is a list of :class:`AlignedVideo` records with raw,
un-normalized scores per video. Apply :func:`apply_normalization` afterwards.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Video-ID aliasing
# ---------------------------------------------------------------------------


def _video_id_alias(video_id: str) -> str:
    """Map a video ID to a canonical short form so two streams that use
    different naming conventions can be intersected.

    Handles two conventions:

    * **Avenue**: STG-NF uses ``01_0021`` (scene 01, clip 21) while MULDE
      uses the clip index ``21``. We therefore treat the *second* token
      (clip index) as the alias: ``01_0021 -> 21``.
    * **ShanghaiTech**: both sides already use the same ``01_0014``
      convention, so no remap is needed (handled by the caller via the
      direct-overlap short-circuit).
    * **Bare integers** (e.g. ``01``, ``21``): the integer itself, zero-padded
      to at least 2 digits, is the alias.
    """
    if "_" in video_id:
        # scene_clip form, e.g. "01_0021" -> clip index "21"
        parts = video_id.split("_", 1)
        clip = parts[-1]
        try:
            return f"{int(clip):02d}"
        except ValueError:
            return clip
    # Bare form: zero-pad if numeric.
    try:
        return f"{int(video_id):02d}"
    except ValueError:
        return video_id


def _build_video_id_map(stgnf: Dict[str, dict], mulde: Dict[str, dict]) -> Dict[str, str]:
    """Return ``{mulde_video_id: stgnf_video_id}`` mapping when the two
    streams use different ID conventions (detected via alias overlap).

    If the two streams already share IDs (e.g. ShanghaiTech), an empty dict is
    returned and the caller should intersect directly.  Otherwise the mapping
    lets us relabel one side to the other before alignment.
    """
    direct = set(stgnf.keys()) & set(mulde.keys())
    if direct:
        return {}

    # Try aliasing: build alias -> stgnf_id and alias -> mulde_id, then match.
    stgnf_by_alias = {}
    for vid in stgnf:
        stgnf_by_alias.setdefault(_video_id_alias(vid), vid)
    mulde_by_alias = {}
    for vid in mulde:
        mulde_by_alias.setdefault(_video_id_alias(vid), vid)

    mapping = {}
    for alias, mvid in mulde_by_alias.items():
        if alias in stgnf_by_alias:
            mapping[mvid] = stgnf_by_alias[alias]
    return mapping


def _apply_video_id_map(scores: Dict[str, dict], mapping: Dict[str, str]) -> Dict[str, dict]:
    """Return a copy of *scores* with video IDs renamed per *mapping*.

    Only keys present in *mapping* are renamed; others are left as-is.
    """
    if not mapping:
        return scores
    out = {}
    for vid, entry in scores.items():
        out[mapping.get(vid, vid)] = entry
    return out


# ---------------------------------------------------------------------------
# Per-video aligned record
# ---------------------------------------------------------------------------


@dataclass
class AlignedVideo:
    video_id: str
    frame_indices: np.ndarray  # int64
    stgnf_scores: np.ndarray   # float32
    mulde_scores: np.ndarray   # float32
    labels: np.ndarray         # uint8 (0/1)


# ---------------------------------------------------------------------------
# Frame intersection
# ---------------------------------------------------------------------------


def _intersect_with_offset(
    s_frames: np.ndarray,
    m_frames: np.ndarray,
    s_scores: np.ndarray,
    m_scores: np.ndarray,
    offset: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect STG-NF and MULDE by frame_index with a configurable offset.

    ``offset`` is added to ``s_frames`` (i.e. STG-NF's frame_index) before the
    intersection. Use this to compensate for STG-NF and MULDE using different
    frame-numbering conventions (most commonly 0-based vs 1-based).
    """
    shifted = s_frames + int(offset)
    common_frames = np.intersect1d(shifted, m_frames)
    if common_frames.size == 0:
        return common_frames, np.empty(0, dtype=s_scores.dtype), np.empty(0, dtype=m_scores.dtype)
    s_idx = np.searchsorted(shifted, common_frames)
    m_idx = np.searchsorted(m_frames, common_frames)
    return common_frames, s_scores[s_idx], m_scores[m_idx]


def _apply_stgnf_polarity(scores: np.ndarray, stgnf_score_mode: str) -> np.ndarray:
    """Convert STG-NF scores into anomaly polarity expected by the fusion code.

    STG-NF's original repository reports *normality* scores for ShanghaiTech:
    larger values indicate more normal frames. The fusion pipeline, however,
    expects *anomaly* scores where larger values indicate more abnormal frames.
    """
    if stgnf_score_mode == "anomaly":
        return scores
    if stgnf_score_mode == "normality":
        return -scores
    raise ValueError(f"Unknown STG-NF score mode: {stgnf_score_mode!r}")


def _resolve_labels(
    m_entry: dict,
    s_entry: dict,
    common_frames: np.ndarray,
) -> Optional[np.ndarray]:
    """Prefer MULDE labels (anomaly convention: 1=abnormal) but accept STG-NF."""
    for entry in (m_entry, s_entry):
        if "labels" not in entry:
            continue
        arr = np.asarray(entry["labels"], dtype=np.uint8)
        if arr.shape[0] == common_frames.shape[0]:
            return arr
        src_frames = np.asarray(entry["frame_indices"], dtype=np.int64)
        if src_frames.shape[0] == arr.shape[0]:
            src_idx = np.searchsorted(src_frames, common_frames)
            if src_idx.size and (src_idx < arr.shape[0]).all():
                return arr[src_idx]
    return None


def _check_label_inversion(
    stgnf: Dict[str, dict],
    mulde: Dict[str, dict],
) -> bool:
    """Return True if STG-NF and MULDE labels are exact inverses for every video.

    Emits a single RuntimeWarning when inversion is detected so the user is
    aware that the two models use different label conventions (e.g. STG-NF
    labels 0=abnormal while MULDE labels 1=abnormal).
    """
    common_videos = sorted(set(stgnf.keys()) & set(mulde.keys()))
    n_checked = 0
    n_inverted = 0
    total_frames = 0
    for vid in common_videos:
        s = stgnf[vid]
        m = mulde[vid]
        if "labels" not in s or "labels" not in m:
            continue
        s_lbl = np.asarray(s["labels"], dtype=np.uint8)
        m_lbl = np.asarray(m["labels"], dtype=np.uint8)
        if s_lbl.shape[0] != m_lbl.shape[0]:
            continue
        n_checked += 1
        if np.all(s_lbl == 1 - m_lbl):
            n_inverted += 1
            total_frames += int(s_lbl.shape[0])

    if n_inverted == n_checked and n_checked > 0:
        warnings.warn(
            f"Label inversion detected: STG-NF and MULDE labels are exact "
            f"inverses for ALL {n_inverted} checked videos ({total_frames} "
            f"frames). One model uses 0=abnormal while the other uses "
            f"1=abnormal. Using MULDE labels (1=anomaly convention).",
            RuntimeWarning,
            stacklevel=2,
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Single-video alignment
# ---------------------------------------------------------------------------


def _align_one_video(
    video_id: str,
    s_entry: dict,
    m_entry: dict,
    stgnf_frame_offset: int,
    stgnf_score_mode: str,
) -> Optional[AlignedVideo]:
    s_frames = np.asarray(s_entry["frame_indices"], dtype=np.int64)
    m_frames = np.asarray(m_entry["frame_indices"], dtype=np.int64)
    s_scores_raw = np.asarray(s_entry["anomaly_scores"], dtype=np.float32)
    m_scores_raw = np.asarray(m_entry["anomaly_scores"], dtype=np.float32)

    common_frames, s_aligned, m_aligned = _intersect_with_offset(
        s_frames, m_frames, s_scores_raw, m_scores_raw, stgnf_frame_offset,
    )
    if common_frames.size:
        s_aligned = _apply_stgnf_polarity(s_aligned, stgnf_score_mode)
    if common_frames.size == 0:
        return AlignedVideo(
            video_id=video_id,
            frame_indices=common_frames,
            stgnf_scores=s_aligned,
            mulde_scores=m_aligned,
            labels=np.empty(0, dtype=np.uint8),
        )

    labels = _resolve_labels(m_entry, s_entry, common_frames)
    if labels is None:
        return None

    return AlignedVideo(
        video_id=video_id,
        frame_indices=common_frames,
        stgnf_scores=s_aligned,
        mulde_scores=m_aligned,
        labels=labels,
    )


# ---------------------------------------------------------------------------
# Auto-detect offset / polarity
# ---------------------------------------------------------------------------


def _align_with_auto_offset(
    stgnf: Dict[str, dict],
    mulde: Dict[str, dict],
    offset_candidates: Tuple[int, ...],
    stgnf_score_mode: str = "auto",
    id_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[List[AlignedVideo], dict]:
    """Search the supplied offset candidates and return the best one.

    The chosen offset maximises STG-NF's single-model Micro AUC on the
    intersected frames. This is robust to the 0-based vs 1-based
    ``frame_index`` mismatch that typically appears when the STG-NF and MULDE
    pipelines were written by different authors.

    ``id_mapping`` (optional) is recorded into the returned stats so callers
    know whether video IDs were remapped.
    """
    best: Optional[Tuple[int, str, List[AlignedVideo], dict, float, int]] = None
    candidate_stats: Dict[str, dict] = {}
    modes = ("anomaly", "normality") if stgnf_score_mode == "auto" else (stgnf_score_mode,)
    for off in offset_candidates:
        for mode in modes:
            aligned: List[AlignedVideo] = []
            key = f"{off:+d}|{mode}"
            per_candidate_stats = {
                "videos_aligned": 0,
                "frames_total": 0,
                "micro_auc_stgnf": None,
                "stgnf_score_mode": mode,
            }
            for video_id in sorted(set(stgnf.keys()) & set(mulde.keys())):
                av = _align_one_video(
                    video_id,
                    stgnf[video_id],
                    mulde[video_id],
                    off,
                    mode,
                )
                if av is None or av.frame_indices.size == 0:
                    continue
                aligned.append(av)
            per_candidate_stats["videos_aligned"] = len(aligned)
            per_candidate_stats["frames_total"] = sum(v.frame_indices.size for v in aligned)
            if aligned:
                all_s = np.concatenate([v.stgnf_scores for v in aligned])
                all_y = np.concatenate([v.labels for v in aligned])
                if len(np.unique(all_y)) >= 2:
                    per_candidate_stats["micro_auc_stgnf"] = float(roc_auc_score(all_y, all_s))
            candidate_stats[key] = per_candidate_stats
            score = per_candidate_stats["micro_auc_stgnf"]
            frames_total = int(per_candidate_stats["frames_total"])
            if score is None or frames_total <= 0:
                continue
            # Prefer the candidate with the largest valid overlap first, then
            # the highest STG-NF AUC inside that overlap. This avoids picking a
            # tiny accidental intersection (e.g. offset -2 with 6 frames) over
            # the true alignment that preserves nearly all frames.
            if (
                best is None
                or frames_total > best[5]
                or (frames_total == best[5] and score > best[4])
            ):
                best = (off, mode, aligned, per_candidate_stats, score, frames_total)

    if best is None:
        # Fall back to offset 0 with an empty alignment rather than crash.
        aligned, stats = align_per_video(
            stgnf,
            mulde,
            stgnf_frame_offset=0,
            stgnf_score_mode="normality",
            apply_id_alias=False,
        )
        if id_mapping:
            stats["video_id_mapping_applied"] = len(id_mapping)
        return aligned, stats

    chosen_offset, chosen_mode, aligned, _, best_auc, _ = best
    aligned, stats = align_per_video(
        stgnf,
        mulde,
        stgnf_frame_offset=chosen_offset,
        stgnf_score_mode=chosen_mode,
        apply_id_alias=False,
    )
    stats["stgnf_frame_offset"] = int(chosen_offset)
    stats["stgnf_score_mode"] = chosen_mode
    if id_mapping:
        stats["video_id_mapping_applied"] = len(id_mapping)
    stats["auto_detect"] = {
        "candidates": list(offset_candidates),
        "stgnf_micro_auc_per_candidate": candidate_stats,
        "chosen_stgnf_micro_auc": float(best_auc),
    }
    return aligned, stats


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def align_per_video(
    stgnf: Dict[str, dict],
    mulde: Dict[str, dict],
    stgnf_frame_offset: int = 0,
    auto_detect_offset: bool = False,
    stgnf_score_mode: str = "auto",
    offset_candidates: Tuple[int, ...] = (-2, -1, 0, 1, 2),
    apply_id_alias: bool = True,
) -> Tuple[List[AlignedVideo], dict]:
    """Intersect STG-NF and MULDE per video without applying normalization.

    Returns the aligned list (with raw, un-normalized scores) and a stats dict
    describing skipped/empty videos. Use :func:`apply_normalization` afterwards
    to scale the per-model scores.

    ``stgnf_frame_offset`` is added to STG-NF's ``frame_index`` before the
    intersection. When ``auto_detect_offset`` is True the function searches
    ``offset_candidates`` and picks the combination of frame offset and
    score polarity that maximises STG-NF's single-model Micro AUC on the
    intersected frames.

    ``stgnf_score_mode`` controls STG-NF polarity:

    * ``"anomaly"``  - larger STG-NF values mean more abnormal.
    * ``"normality"`` - larger STG-NF values mean more normal and are inverted.
    * ``"auto"``      - try both and keep whichever yields the higher AUC.

    ``apply_id_alias`` detects when STG-NF and MULDE use different video-ID
    conventions (e.g. Avenue: STG-NF ``01_0001`` vs MULDE ``01``) and remaps
    MULDE IDs to STG-NF IDs before alignment. Set to False to disable.
    """
    if stgnf_score_mode not in ("auto", "anomaly", "normality"):
        raise ValueError(
            "Unknown stgnf_score_mode: "
            f"{stgnf_score_mode!r}. Valid options: ('auto', 'anomaly', 'normality')"
        )

    # Auto-detect and apply video-ID aliases (e.g. Avenue 01_0001 <-> 01).
    id_mapping = {}
    if apply_id_alias:
        id_mapping = _build_video_id_map(stgnf, mulde)
    if id_mapping:
        mulde = _apply_video_id_map(mulde, id_mapping)

    if auto_detect_offset or stgnf_score_mode == "auto":
        return _align_with_auto_offset(
            stgnf,
            mulde,
            offset_candidates,
            stgnf_score_mode=stgnf_score_mode,
            id_mapping=id_mapping,
        )

    aligned: List[AlignedVideo] = []
    stats = {
        "videos_in_stgnf": 0,
        "videos_in_mulde": 0,
        "videos_aligned": 0,
        "videos_skipped_no_overlap": [],
        "videos_skipped_no_labels": [],
        "videos_skipped_constant": {"stgnf": [], "mulde": []},
        "stgnf_frame_offset": int(stgnf_frame_offset),
        "stgnf_score_mode": stgnf_score_mode,
    }

    common_videos = sorted(set(stgnf.keys()) & set(mulde.keys()))
    stats["videos_in_stgnf"] = len(stgnf)
    stats["videos_in_mulde"] = len(mulde)

    # Cross-check label conventions between STG-NF and MULDE before aligning.
    stats["label_inversion_detected"] = _check_label_inversion(stgnf, mulde)

    for video_id in common_videos:
        aligned_v = _align_one_video(
            video_id,
            stgnf[video_id],
            mulde[video_id],
            stgnf_frame_offset,
            stgnf_score_mode,
        )
        if aligned_v is None:
            continue
        if aligned_v.frame_indices.size == 0:
            stats["videos_skipped_no_overlap"].append(video_id)
            continue
        if aligned_v.stgnf_scores.max() == aligned_v.stgnf_scores.min():
            stats["videos_skipped_constant"]["stgnf"].append(video_id)
        if aligned_v.mulde_scores.max() == aligned_v.mulde_scores.min():
            stats["videos_skipped_constant"]["mulde"].append(video_id)
        aligned.append(aligned_v)

    stats["videos_aligned"] = len(aligned)
    return aligned, stats


__all__ = [
    "AlignedVideo",
    "align_per_video",
]

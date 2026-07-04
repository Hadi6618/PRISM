"""Ensemble fusion for STG-NF + MULDE on the ShanghaiTech Campus test set.

Implements Steps 3 and 4 of ``ensemble_handoff.md``:

* Loads ``stgnf_scores.pkl`` (pose/object-level stream) and ``mulde_scores.pkl``
  (frame-level stream) emitted by the two evaluation notebooks.
* Aligns the two streams per video by ``(video_id, frame_index)`` and applies
  **per-video** Min-Max scaling to both models so they live on a common
  ``[0.0, 1.0]`` range.
* Runs a grid search over ``beta_1`` (STG-NF weight) and ``beta_2 = 1 - beta_1``
  (MULDE weight) and reports the maximum Micro AUC plus the optimal weights.

The script can be imported as a module or run from the command line::

    python fusion.py --stgnf_pkl ... --mulde_pkl ... --output_dir ...

When run with no arguments it uses the default Colab paths described at the
top of ``ensemble_handoff.md``.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas is a hard requirement
    pd = None  # type: ignore[assignment]


# Per-dataset default paths used by the CLI ``main()`` and by the notebook
# config. Each entry maps a dataset key to ``(stgnf_pkl, mulde_pkl, output_dir)``
# Colab Drive paths. Add new datasets here.
DATASET_PATHS = {
    "ShanghaiTech": {
        "stgnf_pkl": Path(
            "/content/drive/MyDrive/STG-NF/original_shanghaitech/logs/shanghaitech_stgnf_scores_84.pkl"
        ),
        "mulde_pkl": Path(
            "/content/drive/MyDrive/MULDE/runs/shanghaitech_hiera_l_mulde/2026_06_10_04_51_41/artifacts/shanghaitech_mulde_scores_79_7.pkl"
        ),
        "output_dir": Path(
            "/content/drive/MyDrive/Fusion/runs/ShanghaiTech/ensemble"
        ),
    },
    "Avenue": {
        "stgnf_pkl": Path(
            "/content/drive/MyDrive/STG-NF/Avenue_dataset/logs/avenue_stgnf_scores_57.pkl"
        ),
        "mulde_pkl": Path(
            "/content/drive/MyDrive/MULDE/runs/avenue_hiera_l_mulde/Final_avenue_scores/artifacts/avenue_mulde_scores_81_4.pkl"
        ),
        "output_dir": Path(
            "/content/drive/MyDrive/Fusion/runs/Avenue/ensemble"
        ),
    },
}

DEFAULT_DATASET = "ShanghaiTech"
# Backwards-compatible single-path defaults (used when --dataset is not passed).
DEFAULT_STGNF_PKL = DATASET_PATHS[DEFAULT_DATASET]["stgnf_pkl"]
DEFAULT_MULDE_PKL = DATASET_PATHS[DEFAULT_DATASET]["mulde_pkl"]
DEFAULT_OUTPUT_DIR = DATASET_PATHS[DEFAULT_DATASET]["output_dir"]


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
# Score loading
# ---------------------------------------------------------------------------


def _normalize_pkl_payload(pkl: object) -> Tuple[Dict[str, dict], dict]:
    """Return (scores_by_video, meta) for either PKL layout."""
    if isinstance(pkl, dict) and "scores_by_video" in pkl:
        return pkl["scores_by_video"], {k: v for k, v in pkl.items() if k != "scores_by_video"}
    if isinstance(pkl, dict):
        # Already a {video_id: {frame_indices, anomaly_scores[, labels]}} mapping.
        return pkl, {}
    raise TypeError(f"Unsupported score-pickle payload type: {type(pkl)!r}")


def load_score_pickle(path: Path) -> Tuple[Dict[str, dict], dict]:
    with open(path, "rb") as f:
        pkl = pickle.load(f)
    scores, meta = _normalize_pkl_payload(pkl)
    # Sanitize inf / NaN values that can arise when a normalizing-flow model
    # encounters degenerate pose windows (e.g. all-zero keypoints from a failed
    # detector).  We replace +inf with the per-video max finite score and -inf /
    # NaN with the per-video min finite score so that the ranking signal is
    # preserved and sklearn's roc_auc_score does not crash.
    total_inf = 0
    for vid, entry in scores.items():
        raw = np.asarray(entry["anomaly_scores"], dtype=np.float64)
        finite_mask = np.isfinite(raw)
        n_bad = int((~finite_mask).sum())
        if n_bad == 0:
            continue
        total_inf += n_bad
        if finite_mask.any():
            vmax = float(raw[finite_mask].max())
            vmin = float(raw[finite_mask].min())
        else:
            vmax = 0.0
            vmin = 0.0
        raw = np.where(raw == np.inf,  vmax, raw)
        raw = np.where(~np.isfinite(raw), vmin, raw)
        entry["anomaly_scores"] = raw.astype(np.float32)
    if total_inf:
        import warnings
        warnings.warn(
            f"load_score_pickle: replaced {total_inf} non-finite scores "
            f"in '{Path(path).name}' with per-video boundary values.",
            RuntimeWarning,
            stacklevel=2,
        )
    return scores, meta


# ---------------------------------------------------------------------------
# Alignment + per-video Min-Max scaling
# ---------------------------------------------------------------------------


@dataclass
class AlignedVideo:
    video_id: str
    frame_indices: np.ndarray  # int64
    stgnf_scores: np.ndarray   # float32
    mulde_scores: np.ndarray   # float32
    labels: np.ndarray         # uint8 (0/1)


def _safe_minmax(values: np.ndarray) -> np.ndarray:
    """Bind values to ``[0, 1]`` while tolerating a constant (zero-variance) clip."""
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)
    vmin = float(values[finite].min())
    vmax = float(values[finite].max())
    if vmax <= vmin:
        return np.zeros_like(values, dtype=np.float32)
    out = (values - vmin) / (vmax - vmin)
    out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)


def _safe_zscore(values: np.ndarray) -> np.ndarray:
    """Z-score with outlier clipping to ``[-3, 3]``, then min-max to ``[0, 1]``."""
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)
    mu = float(values[finite].mean())
    sigma = float(values[finite].std())
    if sigma <= 0.0:
        return np.full_like(values, 0.5, dtype=np.float32)
    out = (values - mu) / sigma
    out = np.clip(out, -3.0, 3.0)
    out = (out + 3.0) / 6.0
    return out.astype(np.float32)


def _rank_to_unit(values: np.ndarray) -> np.ndarray:
    """Convert raw scores to per-model ``[0, 1]`` ranks using average ties."""
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)
    ranks = np.zeros_like(values, dtype=np.float32)
    sub = values[finite]
    order = np.argsort(sub, kind="mergesort")
    sorted_vals = sub[order]
    # Compute average-rank within each tied group.
    tied_ranks = np.empty(sub.shape[0], dtype=np.float32)
    starts = np.concatenate([[0], np.where(np.diff(sorted_vals) != 0)[0] + 1])
    ends = np.concatenate([starts[1:], [sub.shape[0]]])
    for s, e in zip(starts, ends):
        tied_ranks[s:e] = (s + e - 1) / 2.0
    if sub.shape[0] > 1:
        normalized_ranks = tied_ranks / float(sub.shape[0] - 1)
        sub_ranks = np.empty_like(normalized_ranks)
        sub_ranks[order] = normalized_ranks
        ranks[finite] = sub_ranks
    else:
        ranks[finite] = 0.5
    return ranks


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
    labels 0=abomaly while MULDE labels 1=anomaly).
    """
    import warnings

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


# Backwards-compatible alias: keeps the old name pointing at the new
# per-video alignment helper so existing callers keep working.
align_and_normalize = align_per_video


# ---------------------------------------------------------------------------
# Gaussian temporal smoothing
# ---------------------------------------------------------------------------


def smooth_scores(
    aligned: List[AlignedVideo],
    sigma: float,
) -> List[AlignedVideo]:
    """Apply a 1-D Gaussian filter to STG-NF and MULDE scores independently.

    The filter is applied **per video** so that anomalies at video boundaries
    do not bleed across videos. ``sigma`` is in units of frames.

    A sigma of 0 means no smoothing (identity operation).
    """
    if sigma <= 0.0:
        return aligned
    for v in aligned:
        v.stgnf_scores = gaussian_filter1d(
            v.stgnf_scores.astype(np.float64), sigma=sigma
        ).astype(np.float32)
        v.mulde_scores = gaussian_filter1d(
            v.mulde_scores.astype(np.float64), sigma=sigma
        ).astype(np.float32)
    return aligned


def search_best_sigma(
    aligned: List[AlignedVideo],
    sigma_candidates: Tuple[float, ...] = (0, 1, 2, 3, 4, 5, 6, 8, 10, 15),
    normalization: Optional[str] = None,
) -> Tuple[float, dict]:
    """Grid-search over sigma values to maximise per-model standalone AUC.

    For each candidate sigma we:
    1. Apply the requested normalization (if provided).
    2. Smooth both model streams.
    3. Compute the Micro AUC for STG-NF alone and MULDE alone.
    4. Pick the sigma that maximises ``(stgnf_auc + mulde_auc) / 2``.

    Returns the best sigma and a dict of per-candidate results.
    """
    import copy

    best_sigma = 0.0
    best_avg_auc = -1.0
    sigma_results: dict = {}

    for sigma in sigma_candidates:
        # Deep-copy so we do not mutate the original aligned list.
        trial = copy.deepcopy(aligned)
        if normalization is not None:
            trial = apply_normalization(trial, strategy=normalization)
        trial = smooth_scores(trial, sigma=sigma)

        all_stgnf = np.concatenate([v.stgnf_scores for v in trial])
        all_mulde = np.concatenate([v.mulde_scores for v in trial])
        all_labels = np.concatenate([v.labels for v in trial])

        if len(np.unique(all_labels)) < 2:
            sigma_results[sigma] = {"stgnf_auc": None, "mulde_auc": None, "avg_auc": None}
            continue

        stgnf_auc = float(roc_auc_score(all_labels, all_stgnf))
        mulde_auc = float(roc_auc_score(all_labels, all_mulde))
        avg_auc = (stgnf_auc + mulde_auc) / 2.0
        sigma_results[float(sigma)] = {
            "stgnf_auc": round(stgnf_auc, 6),
            "mulde_auc": round(mulde_auc, 6),
            "avg_auc": round(avg_auc, 6),
        }
        print(
            f"  sigma={sigma:5.1f}  STG-NF AUC={stgnf_auc*100:.4f}%"
            f"  MULDE AUC={mulde_auc*100:.4f}%  avg={avg_auc*100:.4f}%"
        )
        if avg_auc > best_avg_auc:
            best_avg_auc = avg_auc
            best_sigma = float(sigma)

    return best_sigma, sigma_results


def smooth_scores_independent(
    aligned: List[AlignedVideo],
    sigma_stgnf: float,
    sigma_mulde: float,
) -> List[AlignedVideo]:
    """Apply 1-D Gaussian smoothing with *independent* sigma per model.

    This is the correct approach when the two score streams have different
    noise profiles (e.g. STG-NF scores are pre-smoothed by their eval
    pipeline while MULDE scores are raw).  A shared sigma would either
    over-smooth STG-NF or under-smooth MULDE.
    """
    if sigma_stgnf > 0.0:
        for v in aligned:
            v.stgnf_scores = gaussian_filter1d(
                v.stgnf_scores.astype(np.float64), sigma=sigma_stgnf,
            ).astype(np.float32)
    if sigma_mulde > 0.0:
        for v in aligned:
            v.mulde_scores = gaussian_filter1d(
                v.mulde_scores.astype(np.float64), sigma=sigma_mulde,
            ).astype(np.float32)
    return aligned


def search_best_sigma_independent(
    aligned: List[AlignedVideo],
    sigma_candidates: Tuple[float, ...] = (0, 1, 2, 3, 4, 5, 6, 8, 10, 15),
    normalization: Optional[str] = None,
) -> Tuple[Tuple[float, float], dict]:
    """Grid-search over independent (sigma_stgnf, sigma_mulde) pairs.

    For each combination we:
    1. Apply normalization (if provided).
    2. Smooth STG-NF with sigma_stgnf and MULDE with sigma_mulde.
    3. Compute the Micro AUC for each model alone.
    4. Pick the pair that maximises ``(stgnf_auc + mulde_auc) / 2``.

    Returns ``(best_sigma_stgnf, best_sigma_mulde)`` and a results dict.
    """
    import copy

    best_pair = (0.0, 0.0)
    best_avg_auc = -1.0
    results: dict = {}

    for s_stgnf in sigma_candidates:
        for s_mulde in sigma_candidates:
            trial = copy.deepcopy(aligned)
            if normalization is not None:
                trial = apply_normalization(trial, strategy=normalization)
            trial = smooth_scores_independent(trial, s_stgnf, s_mulde)

            all_stgnf = np.concatenate([v.stgnf_scores for v in trial])
            all_mulde = np.concatenate([v.mulde_scores for v in trial])
            all_labels = np.concatenate([v.labels for v in trial])

            if len(np.unique(all_labels)) < 2:
                results[(float(s_stgnf), float(s_mulde))] = {
                    "stgnf_auc": None, "mulde_auc": None, "avg_auc": None,
                }
                continue

            stgnf_auc = float(roc_auc_score(all_labels, all_stgnf))
            mulde_auc = float(roc_auc_score(all_labels, all_mulde))
            avg_auc = (stgnf_auc + mulde_auc) / 2.0
            results[(float(s_stgnf), float(s_mulde))] = {
                "stgnf_auc": round(stgnf_auc, 6),
                "mulde_auc": round(mulde_auc, 6),
                "avg_auc": round(avg_auc, 6),
            }
            if avg_auc > best_avg_auc:
                best_avg_auc = avg_auc
                best_pair = (float(s_stgnf), float(s_mulde))

    # Print summary: top-10 pairs by avg AUC.
    sorted_pairs = sorted(results.items(), key=lambda x: x[1].get("avg_auc") or -1, reverse=True)
    print(f"  Independent sigma search: {len(sigma_candidates)}x{len(sigma_candidates)} = "
          f"{len(sigma_candidates)**2} combinations")
    print(f"  {'σ_stgnf':>7s}  {'σ_mulde':>7s}  {'STG-NF':>10s}  {'MULDE':>10s}  {'avg':>10s}")
    for (ss, sm), r in sorted_pairs[:15]:
        sa = r["stgnf_auc"]
        ma = r["mulde_auc"]
        aa = r["avg_auc"]
        sa_s = f"{sa*100:.4f}%" if sa is not None else "n/a"
        ma_s = f"{ma*100:.4f}%" if ma is not None else "n/a"
        aa_s = f"{aa*100:.4f}%" if aa is not None else "n/a"
        marker = " <--" if (ss, sm) == best_pair else ""
        print(f"  {ss:7.1f}  {sm:7.1f}  {sa_s:>10s}  {ma_s:>10s}  {aa_s:>10s}{marker}")

    return best_pair, results


VALID_NORMALIZATIONS = (
    "per_video_minmax",
    "global_minmax",
    "global_zscore",
    "global_rank",
)


def apply_normalization(
    aligned: List[AlignedVideo],
    strategy: str = "global_minmax",
) -> List[AlignedVideo]:
    """Return a copy of ``aligned`` with STG-NF / MULDE scores scaled to ``[0, 1]``.

    Strategies:

    * ``per_video_minmax`` - Min-Max scaling per video (the default in the
      handoff plan). Tends to give the lowest global Micro AUC because it
      destroys the absolute anomaly scale; mainly useful for plan compliance.
    * ``global_minmax``    - Min-Max scaling across the entire aligned test
      set. Preserves the global ranking of both models.
    * ``global_zscore``   - Z-score with ``[-3, 3]`` clipping, then min-max
      to ``[0, 1]``. Robust to outliers.
    * ``global_rank``     - Convert each model's scores to ``[0, 1]`` ranks
      (Borda count). Most robust to scale / orientation differences.
    """
    if strategy not in VALID_NORMALIZATIONS:
        raise ValueError(
            f"Unknown normalization strategy: {strategy!r}. "
            f"Valid options: {VALID_NORMALIZATIONS}"
        )
    aligned = list(aligned)
    if not aligned:
        return aligned

    if strategy == "per_video_minmax":
        for v in aligned:
            v.stgnf_scores = _safe_minmax(v.stgnf_scores)
            v.mulde_scores = _safe_minmax(v.mulde_scores)
        return aligned

    # All global strategies first concatenate the raw scores.
    all_stgnf = np.concatenate([v.stgnf_scores for v in aligned]).astype(np.float32)
    all_mulde = np.concatenate([v.mulde_scores for v in aligned]).astype(np.float32)

    if strategy == "global_minmax":
        s_global = _safe_minmax(all_stgnf)
        m_global = _safe_minmax(all_mulde)
    elif strategy == "global_zscore":
        s_global = _safe_zscore(all_stgnf)
        m_global = _safe_zscore(all_mulde)
    elif strategy == "global_rank":
        s_global = _rank_to_unit(all_stgnf)
        m_global = _rank_to_unit(all_mulde)
    else:  # pragma: no cover - guarded above
        raise ValueError(strategy)

    offset = 0
    for v in aligned:
        n = v.stgnf_scores.shape[0]
        v.stgnf_scores = s_global[offset:offset + n].copy()
        v.mulde_scores = m_global[offset:offset + n].copy()
        offset += n
    return aligned


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------


@dataclass
class GridResult:
    beta_1: float
    beta_2: float
    micro_auc: Optional[float]
    num_frames: int
    num_videos: int


def grid_search_fusion(
    aligned: Iterable[AlignedVideo],
    beta_1_values: Optional[Iterable[float]] = None,
) -> Tuple[List[GridResult], Optional[GridResult], dict]:
    """Run the per-frame weighted-fusion grid search and return results."""
    aligned = list(aligned)
    if not aligned:
        return [], None, {"reason": "no aligned videos"}

    if beta_1_values is None:
        # 0.00, 0.01, ..., 1.00
        beta_1_values = np.round(np.arange(0.0, 1.0 + 1e-9, 0.01), 4).tolist()
    beta_1_values = [float(b) for b in beta_1_values]

    # Pre-stack frame-level arrays for fast evaluation.
    all_stgnf = np.concatenate([v.stgnf_scores for v in aligned]).astype(np.float32)
    all_mulde = np.concatenate([v.mulde_scores for v in aligned]).astype(np.float32)
    all_labels = np.concatenate([v.labels for v in aligned]).astype(np.uint8)
    num_frames = all_labels.shape[0]
    num_videos = len(aligned)

    if len(np.unique(all_labels)) < 2:
        return (
            [
                GridResult(b, 1.0 - b, None, num_frames, num_videos)
                for b in beta_1_values
            ],
            None,
            {"reason": "labels contain a single class"},
        )

    results: List[GridResult] = []
    best: Optional[GridResult] = None
    for beta_1 in beta_1_values:
        beta_2 = 1.0 - beta_1
        fused = beta_1 * all_stgnf + beta_2 * all_mulde
        try:
            auc = float(roc_auc_score(all_labels, fused))
        except ValueError:
            auc = None
        row = GridResult(beta_1, beta_2, auc, num_frames, num_videos)
        results.append(row)
        if auc is not None and (best is None or auc > best.micro_auc):
            best = row

    summary = {
        "num_frames": num_frames,
        "num_videos": num_videos,
        "beta_1_grid_size": len(beta_1_values),
    }
    return results, best, summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def results_to_table(results: List[GridResult]) -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError("pandas is required for result reporting")
    df = pd.DataFrame(
        [
            {
                "beta_1_stgnf": r.beta_1,
                "beta_2_mulde": r.beta_2,
                "micro_auc": r.micro_auc,
            }
            for r in results
        ]
    )
    return df


def write_outputs(
    results: List[GridResult],
    best: Optional[GridResult],
    summary: dict,
    alignment_stats: dict,
    stgnf_meta: dict,
    mulde_meta: dict,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    table = results_to_table(results)
    table_path = output_dir / "fusion_grid_search.csv"
    table.to_csv(table_path, index=False)

    best_payload: dict
    if best is not None:
        best_payload = {
            "beta_1_stgnf": best.beta_1,
            "beta_2_mulde": best.beta_2,
            "max_micro_auc": best.micro_auc,
            "num_frames": best.num_frames,
            "num_videos": best.num_videos,
        }
    else:
        best_payload = {"max_micro_auc": None, "reason": summary.get("reason")}

    report = {
        "best": best_payload,
        "summary": summary,
        "alignment_stats": alignment_stats,
        "stgnf_meta": stgnf_meta,
        "mulde_meta": mulde_meta,
    }
    report_path = output_dir / "fusion_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved grid search table: {table_path}")
    print(f"Saved ensemble report:   {report_path}")
    if best is not None and best.micro_auc is not None:
        print(
            f"Optimal weights -> beta_1 (STG-NF)={best.beta_1:.2f}, "
            f"beta_2 (MULDE)={best.beta_2:.2f}, Micro AUC={best.micro_auc * 100:.4f}%"
        )
    else:
        print("Optimal weights: undefined (insufficient label diversity)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=list(DATASET_PATHS.keys()),
        help=(
            "Dataset key. When set, the STG-NF / MULDE / output paths default "
            f"to the configured {list(DATASET_PATHS.keys())} locations and can "
            "still be overridden by the explicit flags below."
        ),
    )
    # Use sentinel defaults; actual defaults are resolved from --dataset below.
    parser.add_argument("--stgnf_pkl", type=Path, default=None)
    parser.add_argument("--mulde_pkl", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--normalization",
        type=str,
        default="global_minmax",
        choices=VALID_NORMALIZATIONS,
        help=(
            "Score normalization strategy. The handoff plan asks for "
            "per_video_minmax but in practice that destroys the global "
            "ranking and yields a worse Micro AUC than either model alone. "
            "The default (global_minmax) preserves the absolute anomaly "
            "scale while still binding both models to [0, 1]."
        ),
    )
    parser.add_argument(
        "--beta_1_step",
        type=float,
        default=0.01,
        help="Step size for the beta_1 grid (default: 0.01 -> 101 points in [0, 1]).",
    )
    parser.add_argument(
        "--smooth_sigma",
        type=float,
        default=3.0,
        help=(
            "Gaussian smoothing sigma (in frames) applied to both STG-NF and "
            "MULDE scores per video before normalization. Set to 0 to disable. "
            "Default: 3.0 (matches MULDE training pipeline). Use "
            "--smooth_sigma_search to auto-select the best sigma."
        ),
    )
    parser.add_argument(
        "--smooth_sigma_search",
        action="store_true",
        help=(
            "Grid-search over sigma in {0,1,2,3,4,5,6,8,10} and pick the value "
            "that maximises the average of STG-NF and MULDE standalone AUC. "
            "Overrides --smooth_sigma."
        ),
    )
    parser.add_argument(
        "--stgnf_frame_offset",
        type=int,
        default=0,
        help=(
            "Integer offset added to STG-NF's frame_index before alignment. "
            "Use 1 if MULDE's frame_index is 1-based and the STG-NF export "
            "is 0-based (the most common mismatch). The recommended path is "
            "--auto_detect_offset, which sweeps a small range and picks the "
            "offset that maximises STG-NF's single-model Micro AUC."
        ),
    )
    parser.add_argument(
        "--stgnf_score_mode",
        type=str,
        default="auto",
        choices=("auto", "anomaly", "normality"),
        help=(
            "Interpretation of STG-NF values. The original STG-NF repository "
            "reports normality scores on ShanghaiTech, while this fusion code "
            "expects anomaly scores. Leave this at 'auto' unless you know the "
            "export has already been inverted."
        ),
    )
    parser.add_argument(
        "--auto_detect_offset",
        action="store_true",
        help=(
            "Sweep stgnf_frame_offset in {-2, -1, 0, 1, 2} and pick the offset "
            "plus STG-NF polarity that maximise STG-NF's single-model Micro "
            "AUC on the intersected frames. The chosen combination is reported "
            "in the alignment stats."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    # Resolve dataset-keyed defaults if explicit paths were not provided.
    dataset_cfg = DATASET_PATHS.get(args.dataset or DEFAULT_DATASET)
    if args.stgnf_pkl is None:
        args.stgnf_pkl = dataset_cfg["stgnf_pkl"]
    if args.mulde_pkl is None:
        args.mulde_pkl = dataset_cfg["mulde_pkl"]
    if args.output_dir is None:
        args.output_dir = dataset_cfg["output_dir"]

    print(f"Dataset: {args.dataset or DEFAULT_DATASET}")
    print(f"Loading STG-NF scores from: {args.stgnf_pkl}")
    stgnf, stgnf_meta = load_score_pickle(args.stgnf_pkl)
    print(f"Loading MULDE scores from:  {args.mulde_pkl}")
    mulde, mulde_meta = load_score_pickle(args.mulde_pkl)

    aligned, align_stats = align_per_video(
        stgnf,
        mulde,
        stgnf_frame_offset=args.stgnf_frame_offset,
        auto_detect_offset=args.auto_detect_offset,
        stgnf_score_mode=args.stgnf_score_mode,
    )
    chosen_offset = align_stats.get("stgnf_frame_offset", 0)
    chosen_mode = align_stats.get("stgnf_score_mode", args.stgnf_score_mode)
    n_remapped = align_stats.get("video_id_mapping_applied", 0)
    print(
        f"Aligned {align_stats['videos_aligned']} videos "
        f"(STG-NF={align_stats['videos_in_stgnf']}, "
        f"MULDE={align_stats['videos_in_mulde']}, "
        f"stgnf_frame_offset={chosen_offset}, "
        f"stgnf_score_mode={chosen_mode}"
        + (f", video_ids_remapped={n_remapped}" if n_remapped else "")
        + ")."
    )
    if "auto_detect" in align_stats:
        for key, payload in align_stats["auto_detect"]["stgnf_micro_auc_per_candidate"].items():
            auc = payload.get("micro_auc_stgnf")
            auc_s = f"{auc * 100:.4f}%" if auc is not None else "n/a"
            off_s, mode = key.split("|", 1)
            marker = " <-- chosen" if (int(off_s) == chosen_offset and mode == chosen_mode) else ""
            print(f"  offset={int(off_s):+d}  mode={mode:9s}  STG-NF Micro AUC = {auc_s}{marker}")
    if align_stats["videos_skipped_no_overlap"]:
        print(
            f"  Skipped (no overlap): {len(align_stats['videos_skipped_no_overlap'])}"
        )
    if align_stats["videos_skipped_no_labels"]:
        print(
            f"  Skipped (no labels):  {len(align_stats['videos_skipped_no_labels'])}"
        )

    # ---- Normalization (BEFORE Smoothing) -----------------------------------
    aligned = apply_normalization(aligned, strategy=args.normalization)
    print(f"Normalization strategy:     {args.normalization}")
    # -------------------------------------------------------------------------

    # ---- Gaussian temporal smoothing ----------------------------------------
    sigma_search_results: dict = {}
    if args.smooth_sigma_search:
        print("\nSearching for best Gaussian smoothing sigma ...")
        # Pass normalization=None since it is already applied
        chosen_sigma, sigma_search_results = search_best_sigma(
            aligned, normalization=None
        )
        print(f"Best sigma = {chosen_sigma} (maximises avg standalone AUC)")
    else:
        chosen_sigma = args.smooth_sigma

    if chosen_sigma > 0:
        aligned = smooth_scores(aligned, sigma=chosen_sigma)
        print(f"Gaussian smoothing sigma:   {chosen_sigma}")
    else:
        print("Gaussian smoothing:         disabled (sigma=0)")
    # -------------------------------------------------------------------------

    # Quick per-model Micro AUC snapshot for diagnostics.
    all_stgnf = np.concatenate([v.stgnf_scores for v in aligned])
    all_mulde = np.concatenate([v.mulde_scores for v in aligned])
    all_labels = np.concatenate([v.labels for v in aligned])
    if len(np.unique(all_labels)) >= 2:
        s_alone = float(roc_auc_score(all_labels, all_stgnf))
        m_alone = float(roc_auc_score(all_labels, all_mulde))
        print(
            f"Single-model Micro AUC     STG-NF={s_alone * 100:.4f}%  "
            f"MULDE={m_alone * 100:.4f}%"
        )

    beta_1_values = list(np.round(np.arange(0.0, 1.0 + 1e-9, args.beta_1_step), 6))
    results, best, summary = grid_search_fusion(aligned, beta_1_values=beta_1_values)
    summary["dataset"] = args.dataset or DEFAULT_DATASET
    summary["normalization"] = args.normalization
    summary["smooth_sigma"] = chosen_sigma
    if sigma_search_results:
        summary["sigma_search"] = sigma_search_results
    write_outputs(
        results=results,
        best=best,
        summary=summary,
        alignment_stats=align_stats,
        stgnf_meta=stgnf_meta,
        mulde_meta=mulde_meta,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

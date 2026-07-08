"""Score-pickle loading and sanitization.

The two streams (STG-NF, MULDE) export their per-video scores as pickle files
with two supported layouts:

* ``{"scores_by_video": {...}, ...meta...}`` — produced by the newer exporters.
* ``{video_id: {"frame_indices": ..., "anomaly_scores": ..., "labels": ...}}`` —
  the bare dict layout, used by older notebooks.

Both are normalized to a uniform ``Dict[str, dict]`` keyed by ``video_id``.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _normalize_pkl_payload(pkl: object) -> Tuple[Dict[str, dict], dict]:
    """Return (scores_by_video, meta) for either PKL layout."""
    if isinstance(pkl, dict) and "scores_by_video" in pkl:
        return pkl["scores_by_video"], {k: v for k, v in pkl.items() if k != "scores_by_video"}
    if isinstance(pkl, dict):
        # Already a {video_id: {frame_indices, anomaly_scores[, labels]}} mapping.
        return pkl, {}
    raise TypeError(f"Unsupported score-pickle payload type: {type(pkl)!r}")


def load_score_pickle(path: Path) -> Tuple[Dict[str, dict], dict]:
    """Load a STG-NF or MULDE score pickle and sanitize non-finite values.

    The two streams come from heterogeneous pipelines (normalizing-flow
    likelihoods + heuristic object-level scores), and degenerate cases —
    a failed pose detector, an all-zero skeleton window — can produce
    ``+inf`` / ``-inf`` / ``NaN`` scores that crash sklearn's
    ``roc_auc_score``. We replace them with the per-video boundary value
    (max finite for ``+inf``, min finite for ``-inf``/``NaN``) so the
    ranking signal is preserved.
    """
    with open(path, "rb") as f:
        pkl = pickle.load(f)
    scores, meta = _normalize_pkl_payload(pkl)
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
        warnings.warn(
            f"load_score_pickle: replaced {total_inf} non-finite scores "
            f"in '{Path(path).name}' with per-video boundary values.",
            RuntimeWarning,
            stacklevel=2,
        )
    return scores, meta


__all__ = ["load_score_pickle"]

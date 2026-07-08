"""Score normalization strategies.

The two streams emit scores on incompatible scales:

* STG-NF: per-frame **normality** log-likelihood from a normalizing flow
  (already inverted to anomaly polarity by :mod:`prism_alignment`).
  Heavily heavy-tailed; can produce ``+inf`` after sanitization.
* MULDE: per-frame **negative log-likelihood** from a 5-component GMM fit on
  16-dim DSM signatures. Much smoother distribution.

All strategies collapse both streams to ``[0.0, 1.0]`` so the linear
combination ``beta_1 * stgnf + beta_2 * mulde`` is meaningful. The default
in the CLI is ``global_rank`` (Borda count) because it is the most robust
to the heavy tails of normalizing-flow likelihoods.
"""

from __future__ import annotations

from typing import List

import numpy as np

from prism_alignment import AlignedVideo


VALID_NORMALIZATIONS = (
    "per_video_minmax",
    "global_minmax",
    "global_zscore",
    "global_rank",
)


# ---------------------------------------------------------------------------
# Low-level scaling helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


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


__all__ = ["VALID_NORMALIZATIONS", "apply_normalization"]

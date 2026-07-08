"""1-D Gaussian temporal smoothing (with optional independent-sigma search).

The two streams have different noise profiles:

* STG-NF scores are pre-smoothed by the STG-NF evaluation pipeline
  (windowed likelihood), so additional smoothing is usually mild (``σ ≈ 0-3``).
* MULDE scores are raw per-frame GMM log-likelihoods, which can spike
  briefly when the Hiera-L feature jumps. Heavier smoothing
  (``σ ≈ 10-15``) typically helps.

A **shared** sigma would either over-smooth STG-NF or under-smooth MULDE,
so the default is to search ``(σ_stgnf, σ_mulde)`` independently. Use
:func:`smooth_scores_independent` with explicit sigmas when you already know
the right values (e.g. fixed via the ``--smooth_sigma_*`` CLI flags).
"""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score

from prism_alignment import AlignedVideo
from prism_normalization import apply_normalization


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

    Returns ``(best_sigma_stgnf, best_sigma_mulde)`` and a results dict whose
    keys are ``"sigma_stgnf|sigma_mulde"`` strings (JSON-safe).
    """
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

    # Print summary: top-15 pairs by avg AUC.
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

    # JSON cannot serialize tuple keys, so return a parallel dict with
    # string keys of the form "σ_stgnf|σ_mulde" for reporting.
    results_serializable = {
        f"{ss}|{sm}": r for (ss, sm), r in results.items()
    }
    return best_pair, results_serializable


__all__ = [
    "smooth_scores_independent",
    "search_best_sigma_independent",
]

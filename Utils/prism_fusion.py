"""Weighted-fusion grid search and result container.

Given per-video aligned STG-NF and MULDE scores (already normalized and
smoothed by upstream stages), the fusion is the convex combination

    fused[t] = beta_1 * stgnf[t] + beta_2 * mulde[t]
             = beta_1 * stgnf[t] + (1 - beta_1) * mulde[t]

so there is only **one** free parameter (``beta_1``). We sweep it across
``[0, 1]`` and pick the value that maximises the frame-level Micro AUC
(mean per-video Macro AUC is computed alongside and can select the best
row via ``selection_metric='macro'``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from prism_alignment import AlignedVideo
from prism_metrics import AucEvaluator


@dataclass
class GridResult:
    beta_1: float
    beta_2: float
    micro_auc: Optional[float]
    num_frames: int
    num_videos: int
    macro_auc: Optional[float] = None  # mean per-video AUC (one-class videos skipped)


def grid_search_fusion(
    aligned: Iterable[AlignedVideo],
    beta_1_values: Optional[Iterable[float]] = None,
    selection_metric: str = "micro",
) -> Tuple[List[GridResult], Optional[GridResult], dict]:
    """Run the per-frame weighted-fusion grid search and return results.

    Micro AUC picks the best beta_1 by default; ``selection_metric='macro'``
    selects the mean per-video AUC instead. Both metrics are computed for
    every candidate either way.
    """
    if selection_metric not in {"micro", "macro"}:
        raise ValueError("selection_metric must be 'micro' or 'macro'")
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

    evaluator = AucEvaluator(aligned)

    def sort_key(r: GridResult) -> float:
        value = r.micro_auc if selection_metric == "micro" else r.macro_auc
        return -1.0 if value is None else value

    results: List[GridResult] = []
    best: Optional[GridResult] = None
    for beta_1 in beta_1_values:
        beta_2 = 1.0 - beta_1
        fused = beta_1 * all_stgnf + beta_2 * all_mulde
        try:
            auc = float(roc_auc_score(all_labels, fused))
        except ValueError:
            auc = None
        macro_auc = evaluator.macro(fused) if auc is not None else None
        row = GridResult(beta_1, beta_2, auc, num_frames, num_videos, macro_auc)
        results.append(row)
        if auc is not None and (best is None or sort_key(row) > sort_key(best)):
            best = row

    summary = {
        "num_frames": num_frames,
        "num_videos": num_videos,
        "beta_1_grid_size": len(beta_1_values),
        "selection_metric": selection_metric,
    }
    return results, best, summary


__all__ = ["GridResult", "grid_search_fusion"]

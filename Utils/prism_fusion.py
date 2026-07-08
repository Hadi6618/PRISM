"""Weighted-fusion grid search and result container.

Given per-video aligned STG-NF and MULDE scores (already normalized and
smoothed by upstream stages), the fusion is the convex combination

    fused[t] = beta_1 * stgnf[t] + beta_2 * mulde[t]
             = beta_1 * stgnf[t] + (1 - beta_1) * mulde[t]

so there is only **one** free parameter (``beta_1``). We sweep it across
``[0, 1]`` and pick the value that maximises the frame-level Micro AUC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from prism_alignment import AlignedVideo


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


__all__ = ["GridResult", "grid_search_fusion"]

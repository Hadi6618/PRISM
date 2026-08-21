"""Micro / Macro AUC helpers shared by the two-stream and three-stream fusion.

Macro AUC convention (identical to ``MUDLE.ipynb::compute_micro_macro_auc``):
compute ``roc_auc_score`` inside each video over that video's own frames, then
average across videos. Videos whose labels contain a single class (fully
normal test clips) cannot yield a per-video AUC and are skipped, but they are
reported so the effective video count is always visible.

The per-video AUC is computed with the Mann-Whitney rank statistic (average
ranks for ties), which is exactly what ``sklearn.metrics.roc_auc_score``
computes, but cheap enough to call for every weight combination of a fusion
grid search.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata


def rank_auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """AUC via the Mann-Whitney rank statistic; None when one class is absent."""
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape[0] != s.shape[0]:
        raise ValueError("labels and scores must have the same length")
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(s)  # average ranks -> tie handling matches sklearn
    pos_rank_sum = float(ranks[y == 1].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def per_video_auc_rows(
    video_ids: Sequence[str],
    labels_per_video: Sequence[np.ndarray],
    scores_per_video: Sequence[np.ndarray],
) -> Tuple[Optional[float], List[dict], List[str]]:
    """Per-video AUC rows + their mean (macro AUC), skipping one-class videos."""
    rows: List[dict] = []
    skipped: List[str] = []
    aucs: List[float] = []
    for vid, y, s in zip(video_ids, labels_per_video, scores_per_video):
        y = np.asarray(y)
        auc = rank_auc(y, s)
        rows.append(
            {
                "video_id": vid,
                "num_frames": int(y.shape[0]),
                "num_anomaly_frames": int((y == 1).sum()),
                "auc": auc,
            }
        )
        if auc is None:
            skipped.append(vid)
        else:
            aucs.append(auc)
    macro_auc = float(np.mean(aucs)) if aucs else None
    return macro_auc, rows, skipped


def macro_auc_from_scores_by_video(scores_by_video: Dict[str, dict]) -> dict:
    """Macro AUC for a score-pickle payload ``{video_id: {...}}`` mapping.

    Each entry must provide parallel ``anomaly_scores`` and ``labels`` arrays
    (the layout produced by ``stgnf_export_scores.py`` and the MULDE export
    cell). Used to backfill/report macro AUC without re-running a model.
    """
    video_ids = list(scores_by_video.keys())
    labels_list = []
    scores_list = []
    for entry in scores_by_video.values():
        labels_list.append(np.asarray(entry["labels"]))
        scores_list.append(np.asarray(entry["anomaly_scores"], dtype=np.float64))
    macro_auc, rows, skipped = per_video_auc_rows(video_ids, labels_list, scores_list)
    return {
        "macro_auc": macro_auc,
        "num_macro_videos": len(rows) - len(skipped),
        "skipped_macro_videos_one_class": skipped,
        "per_video_rows": rows,
    }


class AucEvaluator:
    """Repeated micro+macro evaluation over one aligned per-video record list.

    ``records`` are duck-typed objects exposing ``labels`` (0/1 array) and
    ``video_id`` — both :class:`prism_alignment.AlignedVideo` and
    :class:`prism_three_way.TripleAligned` qualify. Build once, then call
    :meth:`micro_macro` for every fused score vector of a grid search; the
    per-video label slices are precomputed so each evaluation is a single
    rank pass per video.
    """

    def __init__(self, records: Iterable):
        self.records = list(records)
        if not self.records:
            raise ValueError("AucEvaluator requires at least one record")
        self.video_ids: List[str] = [r.video_id for r in self.records]
        self.labels: np.ndarray = np.concatenate(
            [np.asarray(r.labels) for r in self.records]
        )
        self._slices: List[Tuple[str, np.ndarray, slice]] = []
        start = 0
        for vid, rec in zip(self.video_ids, self.records):
            y = np.asarray(rec.labels)
            stop = start + y.shape[0]
            self._slices.append((vid, y, slice(start, stop)))
            start = stop
        if start != self.labels.shape[0]:
            raise ValueError("record labels length mismatch")

    def micro(self, scores: np.ndarray) -> Optional[float]:
        return rank_auc(self.labels, scores)

    def macro(self, scores: np.ndarray) -> Optional[float]:
        macro, _, _ = self._macro_detail(scores)
        return macro

    def micro_macro(self, scores: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        return self.micro(scores), self.macro(scores)

    def per_video_rows(self, scores: np.ndarray) -> List[dict]:
        _, rows, _ = self._macro_detail(scores)
        return rows

    def _macro_detail(
        self, scores: np.ndarray
    ) -> Tuple[Optional[float], List[dict], List[str]]:
        scores = np.asarray(scores, dtype=np.float64)
        labels_list = [y for _, y, _ in self._slices]
        scores_list = [scores[sl] for _, _, sl in self._slices]
        return per_video_auc_rows(self.video_ids, labels_list, scores_list)


__all__ = [
    "AucEvaluator",
    "rank_auc",
    "per_video_auc_rows",
    "macro_auc_from_scores_by_video",
]

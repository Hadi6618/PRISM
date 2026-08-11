"""Three-stream late fusion: STG-NF + MULDE + third-model CSV.

Avenue third-model CSV layout (friend)::

    clip_id,frame_id,raw_score,smoothed_score
    1,0,0.2203,0.4637
    ...

Pipeline (same philosophy as two-stream PRISM):

1. Load STG-NF / MULDE pickles + third CSV into a common per-video dict.
2. Map video IDs (Avenue: ``1`` / ``01`` / ``01_0001``).
3. Intersect frames; use MULDE labels (``1 = anomaly``).
4. Auto-fix polarity per stream vs labels (keep orientation with higher AUC).
5. Independent ``global_rank`` normalization per stream (not shared scale).
6. Optional Gaussian smooth per stream (skip third if using pre-smoothed column).
7. Grid-search convex weights ``w1+w2+w3=1`` for Micro AUC.

Example (local Avenue)::

    python Utils/prism_three_way.py \\
        --stgnf_pkl Others/Results/avenue_stgnf_scores_64.2.pkl \\
        --mulde_pkl Others/Results/avenue_mulde_scores_81_4.pkl \\
        --third_csv Others/Results/frame_anomaly_scores.csv \\
        --third_score_col smoothed_score \\
        --output_dir Others/Results/three_way_avenue \\
        --normalization global_rank
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

# Allow ``python Utils/prism_three_way.py`` from repo root.
_UTILS = Path(__file__).resolve().parent
if str(_UTILS) not in sys.path:
    sys.path.insert(0, str(_UTILS))

from prism_io import load_score_pickle  # noqa: E402
from prism_normalization import _rank_to_unit, _safe_minmax, _safe_zscore  # noqa: E402


# ---------------------------------------------------------------------------
# Video-ID helpers (Avenue-centric, also works if IDs already match)
# ---------------------------------------------------------------------------


def _alias(video_id: str) -> str:
    """Canonical clip alias: ``01_0001`` / ``01`` / ``1`` -> ``1`` (int string)."""
    s = str(video_id).strip()
    if "_" in s:
        s = s.split("_")[-1]
    try:
        return str(int(s))
    except ValueError:
        return s


def _index_by_alias(scores: Dict[str, dict]) -> Dict[str, str]:
    """alias -> original key (first wins)."""
    out: Dict[str, str] = {}
    for vid in scores:
        a = _alias(vid)
        out.setdefault(a, vid)
    return out


# ---------------------------------------------------------------------------
# Third-model CSV -> pickle-like dict
# ---------------------------------------------------------------------------


def load_third_csv(
    path: Path,
    score_col: str = "smoothed_score",
) -> Dict[str, dict]:
    """Load friend CSV into ``{video_id: {frame_indices, anomaly_scores}}``.

    ``video_id`` is the bare clip index as string (``\"1\"`` ... ``\"21\"``).
    """
    path = Path(path)
    by_clip: Dict[str, List[Tuple[int, float]]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {path}")
        required = {"clip_id", "frame_id", score_col}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV missing columns {missing}. Found: {reader.fieldnames}"
            )
        for row in reader:
            clip = _alias(row["clip_id"])
            frame = int(float(row["frame_id"]))
            score = float(row[score_col])
            by_clip.setdefault(clip, []).append((frame, score))

    out: Dict[str, dict] = {}
    for clip, pairs in by_clip.items():
        pairs = sorted(pairs, key=lambda t: t[0])
        frames = np.asarray([p[0] for p in pairs], dtype=np.int64)
        scores = np.asarray([p[1] for p in pairs], dtype=np.float32)
        # If duplicate frame_ids appear, keep last.
        if frames.size and np.any(np.diff(frames) == 0):
            uniq: Dict[int, float] = {}
            for fr, sc in pairs:
                uniq[fr] = sc
            frames = np.asarray(sorted(uniq.keys()), dtype=np.int64)
            scores = np.asarray([uniq[int(fr)] for fr in frames], dtype=np.float32)
        out[clip] = {
            "frame_indices": frames,
            "anomaly_scores": scores,
        }
    return out


# ---------------------------------------------------------------------------
# Alignment + polarity
# ---------------------------------------------------------------------------


@dataclass
class TripleAligned:
    video_id: str  # canonical alias, e.g. "1"
    frame_indices: np.ndarray
    stgnf: np.ndarray
    mulde: np.ndarray
    third: np.ndarray
    labels: np.ndarray  # 1 = anomaly


def _lookup_on_frames(
    frames: np.ndarray,
    scores: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    """Return scores at query frames; frames must be sorted unique."""
    idx = np.searchsorted(frames, query)
    # safety
    idx = np.clip(idx, 0, len(frames) - 1)
    if not np.array_equal(frames[idx], query):
        raise ValueError("Frame lookup failed — frames not aligned.")
    return scores[idx].astype(np.float32)


def _best_polarity(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, str, float]:
    """Return (oriented_scores, mode, auc) with mode in {anomaly, normality}."""
    y = labels.astype(np.uint8)
    if len(np.unique(y)) < 2:
        return scores.astype(np.float32), "anomaly", float("nan")
    s = scores.astype(np.float64)
    auc_pos = float(roc_auc_score(y, s))
    auc_neg = float(roc_auc_score(y, -s))
    if auc_neg > auc_pos:
        return (-s).astype(np.float32), "normality->flipped", auc_neg
    return s.astype(np.float32), "anomaly", auc_pos


def align_three(
    stgnf: Dict[str, dict],
    mulde: Dict[str, dict],
    third: Dict[str, dict],
) -> Tuple[List[TripleAligned], dict]:
    """Intersect three streams on common (clip, frame); labels from MULDE."""
    stg_alias = _index_by_alias(stgnf)
    mul_alias = _index_by_alias(mulde)
    thr_alias = _index_by_alias(third)

    common_clips = sorted(
        set(stg_alias) & set(mul_alias) & set(thr_alias),
        key=lambda x: int(x) if x.isdigit() else x,
    )
    aligned: List[TripleAligned] = []
    skipped = []

    for alias in common_clips:
        s_key, m_key, t_key = stg_alias[alias], mul_alias[alias], thr_alias[alias]
        s_e, m_e, t_e = stgnf[s_key], mulde[m_key], third[t_key]

        s_f = np.asarray(s_e["frame_indices"], dtype=np.int64)
        m_f = np.asarray(m_e["frame_indices"], dtype=np.int64)
        t_f = np.asarray(t_e["frame_indices"], dtype=np.int64)
        s_s = np.asarray(s_e["anomaly_scores"], dtype=np.float32)
        m_s = np.asarray(m_e["anomaly_scores"], dtype=np.float32)
        t_s = np.asarray(t_e["anomaly_scores"], dtype=np.float32)

        if "labels" not in m_e:
            skipped.append((alias, "mulde missing labels"))
            continue
        m_lab = np.asarray(m_e["labels"], dtype=np.uint8)
        if m_lab.shape[0] != m_f.shape[0]:
            skipped.append((alias, "mulde labels length mismatch"))
            continue

        # Prefer offset 0; try small offsets if needed (MULDE sometimes 1-based).
        best_common = np.array([], dtype=np.int64)
        best_off = 0
        for off in (0, 1, -1, 2, -2):
            common = np.intersect1d(np.intersect1d(s_f + off, m_f), t_f)
            if common.size > best_common.size:
                best_common = common
                best_off = off
        common = best_common
        if common.size == 0:
            skipped.append((alias, "no common frames"))
            continue

        # Map STG frames with offset: score at frame f comes from s_f index of (f - off)
        s_frames_shifted = s_f + best_off
        order_s = np.argsort(s_frames_shifted)
        s_frames_shifted = s_frames_shifted[order_s]
        s_s_ord = s_s[order_s]

        order_m = np.argsort(m_f)
        m_f_ord, m_s_ord, m_lab_ord = m_f[order_m], m_s[order_m], m_lab[order_m]
        order_t = np.argsort(t_f)
        t_f_ord, t_s_ord = t_f[order_t], t_s[order_t]

        try:
            stg_on = _lookup_on_frames(s_frames_shifted, s_s_ord, common)
            mul_on = _lookup_on_frames(m_f_ord, m_s_ord, common)
            thr_on = _lookup_on_frames(t_f_ord, t_s_ord, common)
            lab_on = _lookup_on_frames(m_f_ord, m_lab_ord.astype(np.float32), common).astype(
                np.uint8
            )
        except ValueError as exc:
            skipped.append((alias, str(exc)))
            continue

        aligned.append(
            TripleAligned(
                video_id=alias,
                frame_indices=common.astype(np.int64),
                stgnf=stg_on,
                mulde=mul_on,
                third=thr_on,
                labels=lab_on,
            )
        )

    stats = {
        "clips_common_ids": len(common_clips),
        "clips_aligned": len(aligned),
        "frames_total": int(sum(v.frame_indices.size for v in aligned)),
        "skipped": skipped,
        "id_examples": {
            "stgnf": {a: stg_alias[a] for a in common_clips[:3]},
            "mulde": {a: mul_alias[a] for a in common_clips[:3]},
            "third": {a: thr_alias[a] for a in common_clips[:3]},
        },
    }
    return aligned, stats


def apply_polarity(aligned: List[TripleAligned]) -> Tuple[List[TripleAligned], dict]:
    """Flip any stream whose raw orientation is worse than its negation vs labels."""
    if not aligned:
        return aligned, {}
    all_y = np.concatenate([v.labels for v in aligned])
    streams = {
        "stgnf": np.concatenate([v.stgnf for v in aligned]),
        "mulde": np.concatenate([v.mulde for v in aligned]),
        "third": np.concatenate([v.third for v in aligned]),
    }
    report = {}
    oriented = {}
    for name, raw in streams.items():
        fixed, mode, auc = _best_polarity(raw, all_y)
        oriented[name] = fixed
        report[name] = {"mode": mode, "micro_auc_after_polarity": auc}

    # Write back per video
    offset = 0
    out: List[TripleAligned] = []
    for v in aligned:
        n = v.frame_indices.size
        out.append(
            TripleAligned(
                video_id=v.video_id,
                frame_indices=v.frame_indices,
                stgnf=oriented["stgnf"][offset : offset + n].copy(),
                mulde=oriented["mulde"][offset : offset + n].copy(),
                third=oriented["third"][offset : offset + n].copy(),
                labels=v.labels,
            )
        )
        offset += n
    return out, report


# ---------------------------------------------------------------------------
# Normalization (independent per stream — same as PRISM global_*)
# ---------------------------------------------------------------------------


def normalize_three(
    aligned: List[TripleAligned],
    strategy: str = "global_rank",
) -> List[TripleAligned]:
    if not aligned:
        return aligned
    all_s = np.concatenate([v.stgnf for v in aligned]).astype(np.float32)
    all_m = np.concatenate([v.mulde for v in aligned]).astype(np.float32)
    all_t = np.concatenate([v.third for v in aligned]).astype(np.float32)

    if strategy == "global_rank":
        ns, nm, nt = _rank_to_unit(all_s), _rank_to_unit(all_m), _rank_to_unit(all_t)
    elif strategy == "global_minmax":
        ns, nm, nt = _safe_minmax(all_s), _safe_minmax(all_m), _safe_minmax(all_t)
    elif strategy == "global_zscore":
        ns, nm, nt = _safe_zscore(all_s), _safe_zscore(all_m), _safe_zscore(all_t)
    elif strategy == "none":
        ns, nm, nt = all_s, all_m, all_t
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    out: List[TripleAligned] = []
    offset = 0
    for v in aligned:
        n = v.frame_indices.size
        out.append(
            TripleAligned(
                video_id=v.video_id,
                frame_indices=v.frame_indices,
                stgnf=ns[offset : offset + n].copy(),
                mulde=nm[offset : offset + n].copy(),
                third=nt[offset : offset + n].copy(),
                labels=v.labels,
            )
        )
        offset += n
    return out


def _gaussian_smooth_1d(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma is None or sigma <= 0:
        return x.astype(np.float32)
    radius = max(1, int(3 * sigma))
    t = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(x.astype(np.float64), kernel, mode="same").astype(np.float32)


def smooth_three(
    aligned: List[TripleAligned],
    sigma_stgnf: float = 0.0,
    sigma_mulde: float = 0.0,
    sigma_third: float = 0.0,
) -> List[TripleAligned]:
    out = []
    for v in aligned:
        out.append(
            TripleAligned(
                video_id=v.video_id,
                frame_indices=v.frame_indices,
                stgnf=_gaussian_smooth_1d(v.stgnf, sigma_stgnf),
                mulde=_gaussian_smooth_1d(v.mulde, sigma_mulde),
                third=_gaussian_smooth_1d(v.third, sigma_third),
                labels=v.labels,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Fusion grid on simplex w1+w2+w3=1
# ---------------------------------------------------------------------------


@dataclass
class TripleGridResult:
    w_stgnf: float
    w_mulde: float
    w_third: float
    micro_auc: float


def grid_search_three(
    aligned: List[TripleAligned],
    step: float = 0.05,
) -> Tuple[List[TripleGridResult], Optional[TripleGridResult]]:
    if not aligned:
        return [], None
    s = np.concatenate([v.stgnf for v in aligned]).astype(np.float64)
    m = np.concatenate([v.mulde for v in aligned]).astype(np.float64)
    t = np.concatenate([v.third for v in aligned]).astype(np.float64)
    y = np.concatenate([v.labels for v in aligned]).astype(np.uint8)
    if len(np.unique(y)) < 2:
        return [], None

    results: List[TripleGridResult] = []
    best: Optional[TripleGridResult] = None
    # w1, w2 on grid; w3 = 1 - w1 - w2 >= 0
    vals = np.round(np.arange(0.0, 1.0 + 1e-9, step), 6)
    for w1 in vals:
        for w2 in vals:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9:
                continue
            w3 = max(0.0, float(w3))
            fused = w1 * s + w2 * m + w3 * t
            try:
                auc = float(roc_auc_score(y, fused))
            except ValueError:
                continue
            row = TripleGridResult(float(w1), float(w2), float(w3), auc)
            results.append(row)
            if best is None or auc > best.micro_auc:
                best = row
    results.sort(key=lambda r: r.micro_auc, reverse=True)
    return results, best


def standalone_aucs(aligned: List[TripleAligned]) -> dict:
    y = np.concatenate([v.labels for v in aligned])
    out = {}
    for name in ("stgnf", "mulde", "third"):
        s = np.concatenate([getattr(v, name) for v in aligned])
        out[name] = float(roc_auc_score(y, s)) if len(np.unique(y)) >= 2 else None
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stgnf_pkl", type=Path, required=True)
    p.add_argument("--mulde_pkl", type=Path, required=True)
    p.add_argument("--third_csv", type=Path, required=True)
    p.add_argument(
        "--third_score_col",
        type=str,
        default="smoothed_score",
        choices=("smoothed_score", "raw_score"),
        help="Which CSV column to fuse. Prefer smoothed_score; then keep sigma_third=0.",
    )
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument(
        "--normalization",
        type=str,
        default="global_rank",
        choices=("global_rank", "global_minmax", "global_zscore", "none"),
    )
    p.add_argument("--weight_step", type=float, default=0.05, help="Simplex grid step.")
    p.add_argument("--sigma_stgnf", type=float, default=0.0)
    p.add_argument("--sigma_mulde", type=float, default=0.0)
    p.add_argument(
        "--sigma_third",
        type=float,
        default=0.0,
        help="Extra smooth for third model. Use 0 if CSV already smoothed.",
    )
    args = p.parse_args(argv)

    stgnf, _ = load_score_pickle(args.stgnf_pkl)
    mulde, _ = load_score_pickle(args.mulde_pkl)
    third = load_third_csv(args.third_csv, score_col=args.third_score_col)

    aligned, align_stats = align_three(stgnf, mulde, third)
    print("Align:", json.dumps({k: align_stats[k] for k in align_stats if k != "skipped"}, indent=2))
    if align_stats["skipped"]:
        print("Skipped clips:", align_stats["skipped"][:10])

    aligned, pol = apply_polarity(aligned)
    print("Polarity (vs MULDE labels, 1=anomaly):")
    for k, v in pol.items():
        print(f"  {k}: {v}")

    raw_auc = standalone_aucs(aligned)
    print("Standalone Micro AUC after polarity (before norm):", raw_auc)

    aligned = normalize_three(aligned, strategy=args.normalization)
    aligned = smooth_three(
        aligned,
        sigma_stgnf=args.sigma_stgnf,
        sigma_mulde=args.sigma_mulde,
        sigma_third=args.sigma_third,
    )
    norm_auc = standalone_aucs(aligned)
    print(
        f"Standalone Micro AUC after {args.normalization} (+smooth):",
        norm_auc,
    )

    results, best = grid_search_three(aligned, step=args.weight_step)
    if best is None:
        print("Fusion failed (no valid AUC).")
        return 1

    print(
        f"BEST fusion: w_stgnf={best.w_stgnf:.3f}  w_mulde={best.w_mulde:.3f}  "
        f"w_third={best.w_third:.3f}  Micro AUC={best.micro_auc * 100:.2f}%"
    )
    print("Top-5 weight combos:")
    for row in results[:5]:
        print(
            f"  ({row.w_stgnf:.2f}, {row.w_mulde:.2f}, {row.w_third:.2f}) -> "
            f"{row.micro_auc * 100:.2f}%"
        )

    # Pairwise ablations (set one weight to 0 on the simplex via 2-way re-eval)
    s = np.concatenate([v.stgnf for v in aligned])
    m = np.concatenate([v.mulde for v in aligned])
    t = np.concatenate([v.third for v in aligned])
    y = np.concatenate([v.labels for v in aligned])

    def best_two(a, b, name):
        best_a, best_auc = 0.0, -1.0
        for w in np.linspace(0, 1, 101):
            auc = float(roc_auc_score(y, w * a + (1 - w) * b))
            if auc > best_auc:
                best_auc, best_a = auc, w
        print(f"  best {name}: w_first={best_a:.2f} AUC={best_auc * 100:.2f}%")

    print("Pairwise (normalized scores):")
    best_two(s, m, "STG+MULDE")
    best_two(s, t, "STG+third")
    best_two(m, t, "MULDE+third")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "align_stats": align_stats,
        "polarity": pol,
        "standalone_auc_after_polarity": raw_auc,
        "standalone_auc_after_norm": norm_auc,
        "normalization": args.normalization,
        "third_score_col": args.third_score_col,
        "sigmas": {
            "stgnf": args.sigma_stgnf,
            "mulde": args.sigma_mulde,
            "third": args.sigma_third,
        },
        "best": {
            "w_stgnf": best.w_stgnf,
            "w_mulde": best.w_mulde,
            "w_third": best.w_third,
            "micro_auc": best.micro_auc,
        },
        "top5": [
            {
                "w_stgnf": r.w_stgnf,
                "w_mulde": r.w_mulde,
                "w_third": r.w_third,
                "micro_auc": r.micro_auc,
            }
            for r in results[:5]
        ],
        "paths": {
            "stgnf_pkl": str(args.stgnf_pkl),
            "mulde_pkl": str(args.mulde_pkl),
            "third_csv": str(args.third_csv),
        },
    }
    # JSON-friendly skipped
    report["align_stats"] = dict(align_stats)
    report["align_stats"]["skipped"] = [
        {"clip": a, "reason": b} for a, b in align_stats.get("skipped", [])
    ]

    with (out_dir / "three_way_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with (out_dir / "three_way_grid.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["w_stgnf", "w_mulde", "w_third", "micro_auc"])
        for r in results:
            w.writerow([r.w_stgnf, r.w_mulde, r.w_third, r.micro_auc])
    print(f"Wrote {out_dir / 'three_way_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

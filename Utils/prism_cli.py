"""Command-line interface for the PRISM ensemble.

Usage example::

    python PRISM.py --dataset ShanghaiTech \\
        --normalization global_rank \\
        --smooth_sigma_search \\
        --auto_detect_offset

All real work happens in the per-stage modules (:mod:`prism_io`,
:mod:`prism_alignment`, :mod:`prism_normalization`, :mod:`prism_smoothing`,
:mod:`prism_fusion`, :mod:`prism_reporting`). This module is the glue
that wires them together and resolves the per-dataset default paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from prism_alignment import align_per_video
from prism_config import (
    DATASET_PATHS,
    DEFAULT_DATASET,
)
from prism_fusion import grid_search_fusion
from prism_io import load_score_pickle
from prism_normalization import VALID_NORMALIZATIONS, apply_normalization
from prism_reporting import write_outputs
from prism_smoothing import (
    search_best_sigma_independent,
    smooth_scores_independent,
)


__all__ = ["_parse_args", "main"]


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
        default="global_rank",
        choices=VALID_NORMALIZATIONS,
        help=(
            "Score normalization strategy. global_rank (default) converts each "
            "model's raw scores to [0, 1] percentiles (Borda-style) and is the "
            "most robust to the heavy-tailed normalizing-flow likelihoods. See "
            "VALID_NORMALIZATIONS for alternatives."
        ),
    )
    parser.add_argument(
        "--beta_1_step",
        type=float,
        default=0.01,
        help="Step size for the beta_1 grid (default: 0.01 -> 101 points in [0, 1]).",
    )
    parser.add_argument(
        "--smooth_sigma_search",
        action="store_true",
        help=(
            "Grid-search the independent (sigma_stgnf, sigma_mulde) pair that "
            "maximises the average of STG-NF and MULDE standalone AUC. This is "
            "the recommended mode: the two streams have different noise profiles "
            "(STG-NF scores are pre-smoothed by the eval pipeline, MULDE scores "
            "are raw) so a single shared sigma is a compromise."
        ),
    )
    parser.add_argument(
        "--smooth_sigma_stgnf",
        type=float,
        default=0.0,
        help=(
            "Manual Gaussian sigma (frames) for the STG-NF stream. Ignored when "
            "--smooth_sigma_search is set. Set to 0 for no smoothing."
        ),
    )
    parser.add_argument(
        "--smooth_sigma_mulde",
        type=float,
        default=0.0,
        help=(
            "Manual Gaussian sigma (frames) for the MULDE stream. Ignored when "
            "--smooth_sigma_search is set. Set to 0 for no smoothing."
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
        print("\nSearching for best Gaussian smoothing (sigma_stgnf, sigma_mulde) ...")
        # Pass normalization=None since it is already applied above.
        chosen_pair, sigma_search_results = search_best_sigma_independent(
            aligned, normalization=None
        )
        chosen_sigma_stgnf, chosen_sigma_mulde = chosen_pair
        print(
            f"Best independent sigma = ({chosen_sigma_stgnf}, {chosen_sigma_mulde})"
            " (maximises avg standalone AUC)"
        )
    else:
        chosen_sigma_stgnf = float(args.smooth_sigma_stgnf)
        chosen_sigma_mulde = float(args.smooth_sigma_mulde)

    aligned = smooth_scores_independent(
        aligned, sigma_stgnf=chosen_sigma_stgnf, sigma_mulde=chosen_sigma_mulde,
    )
    print(
        f"Gaussian smoothing sigma:   STG-NF={chosen_sigma_stgnf}  MULDE={chosen_sigma_mulde}"
    )
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
    summary["smooth_sigma_stgnf"] = chosen_sigma_stgnf
    summary["smooth_sigma_mulde"] = chosen_sigma_mulde
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


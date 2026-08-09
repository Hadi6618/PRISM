"""Result persistence: CSV grid-search table + JSON ensemble report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from prism_fusion import GridResult


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


__all__ = ["results_to_table", "write_outputs"]

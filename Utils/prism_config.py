"""Per-dataset default paths and shared constants for PRISM.

Kept dependency-free (only stdlib) so any other module can import it
without creating circular import risk.
"""

from __future__ import annotations

from pathlib import Path

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


__all__ = [
    "DATASET_PATHS",
    "DEFAULT_DATASET",
    "DEFAULT_STGNF_PKL",
    "DEFAULT_MULDE_PKL",
    "DEFAULT_OUTPUT_DIR",
]

"""Ensemble PRISM for STG-NF + MULDE (PRISM) on the ShanghaiTech Campus
and Avenue test sets.

This module is a thin **backwards-compatible shim**. The implementation
has been split across focused sub-modules (one per pipeline stage). This
file re-exports the public API and serves as the ``python PRISM.py``
entry point.

The full pipeline:

* Loads a STG-NF score pickle (pose/object-level stream) and a MULDE score
  pickle (frame-level stream), then:
* Aligns the two streams per video by ``(video_id, frame_index)``. Video-ID
  conventions are auto-detected and remapped (Avenue: ``01_0001 <-> 21``).
* Applies the requested score normalization so both streams live on
  ``[0.0, 1.0]``.
* Grid-searches an **independent** Gaussian smoothing sigma per model
  (``sigma_stgnf``, ``sigma_mulde``). This is preferable to a single shared
  sigma because the two streams have different noise profiles: STG-NF scores
  are pre-smoothed by their eval pipeline while MULDE scores are raw.
* Runs a grid search over ``beta_1`` (STG-NF weight) and ``beta_2 = 1 - beta_1``
  (MULDE weight) and reports the maximum Micro AUC plus the optimal weights.

The script can be imported as a module or run from the command line::

    python PRISM.py --dataset ShanghaiTech --smooth_sigma_search --auto_detect_offset
    python PRISM.py --dataset Avenue      --smooth_sigma_search --auto_detect_offset

Public API (re-exported from sub-modules):

* :data:`DATASET_PATHS`, :data:`DEFAULT_DATASET`,
  :data:`DEFAULT_STGNF_PKL`, :data:`DEFAULT_MULDE_PKL`,
  :data:`DEFAULT_OUTPUT_DIR`
* :func:`load_score_pickle` (from :mod:`prism_io`)
* :class:`AlignedVideo`, :func:`align_per_video` (from :mod:`prism_alignment`)
* :data:`VALID_NORMALIZATIONS`, :func:`apply_normalization`
  (from :mod:`prism_normalization`)
* :func:`smooth_scores_independent`, :func:`search_best_sigma_independent`
  (from :mod:`prism_smoothing`)
* :class:`GridResult`, :func:`grid_search_fusion` (from :mod:`prism_fusion`)
* :func:`results_to_table`, :func:`write_outputs` (from :mod:`prism_reporting`)
* :func:`main` (from :mod:`prism_cli`)
"""

from __future__ import annotations

# --- Bootstrap: ensure the Utils/ directory is on sys.path --------------------
# When this file is loaded by the PRISM_Runner.ipynb via
# ``importlib.util.spec_from_file_location('prism', 'PRISM.py')``, the
# containing directory is NOT automatically added to ``sys.path``. The
# ``prism_*.py`` modules live in the sibling ``Utils/`` folder, so we add
# it to ``sys.path`` before importing any of them. This keeps both
# notebook-style dynamic loading and ordinary ``python PRISM.py``
# execution working with the same source.
import sys
import pathlib as _pathlib

_SHIM_DIR = str(_pathlib.Path(__file__).resolve().parent)
_UTILS_DIR = str(_pathlib.Path(__file__).resolve().parent / "Utils")
for _dir in (_UTILS_DIR, _SHIM_DIR):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

# --- Public re-exports --------------------------------------------------------
# Configuration
from prism_config import (  # noqa: E402  - bootstrap above
    DATASET_PATHS,
    DEFAULT_DATASET,
    DEFAULT_MULDE_PKL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_STGNF_PKL,
)

# Score-pickle loading
from prism_io import load_score_pickle  # noqa: E402

# Alignment
from prism_alignment import (  # noqa: E402
    AlignedVideo,
    align_per_video,
)

# Score normalization
from prism_normalization import (  # noqa: E402
    VALID_NORMALIZATIONS,
    apply_normalization,
)

# Temporal smoothing
from prism_smoothing import (  # noqa: E402
    search_best_sigma_independent,
    smooth_scores_independent,
)

# Weighted fusion
from prism_fusion import (  # noqa: E402
    GridResult,
    grid_search_fusion,
)

# Reporting
from prism_reporting import (  # noqa: E402
    results_to_table,
    write_outputs,
)

# CLI
from prism_cli import main, _parse_args  # noqa: E402

# --- CLI entry point ----------------------------------------------------------
if __name__ == "__main__":
    raise SystemExit(main())

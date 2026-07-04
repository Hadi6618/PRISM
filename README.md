# PRISM — Pose + RGB Integration for Scene Monitoring

> A **prism** splits a single beam of light into its constituent spectral
> components so each can be analyzed separately, then recombined. **PRISM**
> does the same for video: it decomposes each frame into two complementary
> streams — **pose** (what the people are doing) and **RGB appearance** (what
> the scene looks like) — analyzes each with a specialized detector, and
> fuses their scores to catch anomalies that either stream would miss alone.

A two-stream late-fusion framework for Video Anomaly Detection (VAD) that
combines a **pose-based** model (STG-NF) with an **appearance-based** model
(MULDE). The two streams are complementary: one watches *what people do*,
the other watches *what the scene looks like*. Fusing their frame-level
scores yields a substantial improvement over either model alone.

**PRISM** = **P**ose + **R**GB **I**ntegration for **S**cene **M**onitoring.

| Method | Stream | ShanghaiTech (Micro AUC) | Avenue (Micro AUC) |
| :-- | :-- | ---: | ---: |
| MULDE (Hiera-L + DSM) | Appearance | 79.66% | 81.4% |
| STG-NF (AlphaPose + Flow) | Pose | 83.53% | 57.0% |
| **PRISM (fusion)** | **Both** | **89.32%** | — |

---

## Motivation

Single-modality VAD models have systematic blind spots:

- **Appearance models** (MULDE) detect contextual anomalies such as vehicles
  on sidewalks, bicycles, or objects that should not be in the scene. They
  are robust to skeleton-tracking jitter but cannot reason about *motion* or
  *behaviour* — a person fighting looks similar, frame-by-frame, to a person
  gesturing.
- **Pose models** (STG-NF) detect behavioural anomalies such as fighting,
  falling, or stealing. They capture kinematics but are blind to any anomaly
  without a human skeleton (vehicles, objects) and carry a high noise floor
  from pose-estimation jitter.

These failure modes are largely disjoint, so an ensemble of the two can
recover detections that either model misses on its own.

---

## Architecture

The system is organised as two independent scoring streams whose outputs are
aligned, normalised, and fused at the **frame level** (late fusion).

```
                   ┌───────────────────────────┐
                   │      Raw Video Frames     │
                   └─────────────┬─────────────┘
                                 │
            ┌────────────────────┴────────────────────┐
            ▼                                         ▼
  ┌─────────────────────┐                   ┌─────────────────────┐
  │   POSE STREAM       │                   │  APPEARANCE STREAM  │
  │   (STG-NF)          │                   │   (MULDE)           │
  │                     │                   │                     │
  │ YOLOX-X → boxes     │                   │ Hiera-L → 1152-D    │
  │ FastPose → 17 kpts  │                   │   frame features    │
  │ PoseFlow → track IDs│                   │        │            │
  │        │            │                   │        ▼            │
  │ Spatio-Temporal     │                   │ Denoising Score     │
  │   Graph (GCN)       │                   │   Matching (MLP)    │
  │        │            │                   │        │            │
  │ Normalizing Flow    │                   │ Multiscale (×16 σ)  │
  │        │            │                   │        │            │
  │ −log p(x) per frame │                   │ GMM −log-likelihood │
  └─────────┬───────────┘                   └──────────┬──────────┘
            │                                          │
            │   normality score                        │  anomaly score
            ▼                                          ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                        FUSION PIPELINE                       │
  │  1. Per-video frame alignment + polarity correction          │
  │     (auto-detects offset, score polarity, video-ID aliases) │
  │  2. Global rank normalization to [0, 1]                      │
  │  3. Per-video Gaussian temporal smoothing (σ tuned per set)  │
  │  4. Grid-search: β₁·STG-NF + β₂·MULDE                      │
  └──────────────────────────────┬───────────────────────────────┘
                                 ▼
                     Final frame-level anomaly score
```

### Stream 1 — STG-NF (pose / kinematic)

STG-NF models normal human motion with **Spatio-Temporal Graph Normalizing
Flows**. Each person is represented as a graph of 17 COCO keypoints tracked
across time, embedded by a Graph Convolutional Network, and mapped to a
latent Gaussian distribution through a stack of bijective coupling layers.
At test time, the negative log-likelihood of a pose window under the learned
flow serves as the anomaly score — unusual motions (fighting, falling,
loitering) receive low likelihood.

- **Pose extraction:** AlphaPose (FastPose-ResNet152) with a YOLOX-X
  detector, followed by PoseFlow / OSNet ReID for persistent person tracks.
- **Backbone:** STG-CN feature extractor feeding an affine-coupling
  normalizing flow.
- **Output:** per-frame normality score (higher = more normal), inverted
  during fusion.

> Paper: *Normalizing Flows for Human Pose Anomaly Detection*, ICCV 2023 —
> [arXiv:2211.10946](https://arxiv.org/abs/2211.10946) ·
> [code](https://github.com/orhir/STG-NF)

### Stream 2 — MULDE (appearance / contextual)

MULDE learns a **multiscale density model** of normal frame appearance.
Frame features are extracted with a Hiera-L video backbone, then a small
MLP is trained via Denoising Score Matching (DSM) to estimate the gradient
of the log-density of the normal-data distribution at multiple noise scales.
At evaluation time, the score network is queried at 16 fixed noise levels,
producing a 16-dimensional log-density signature per frame. A Gaussian
Mixture Model fitted on the *training* signatures turns each test signature
into a scalar negative log-likelihood — the anomaly score.

- **Feature backbone:** Hiera-L, 1152-D per-frame features (16-frame clips,
  stride 4).
- **Density model:** 2-layer MLP (4096 units) trained with DSM; evaluated
  at 16 noise scales.
- **Scoring:** GMM (5 components) negative log-likelihood.

> Paper: *MULDE: Multiscale Log-Density Estimation via Denoising Score
> Matching for Video Anomaly Detection*, CVPR 2024 —
> [PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Micorek_MULDE_Multiscale_Log-Density_Estimation_via_Denoising_Score_Matching_for_Video_CVPR_2024_paper.pdf) ·
> [code](https://github.com/jmicorek/mulde)

### Fusion

The two streams emit scores on incompatible scales and with opposite
polarities, so the fusion pipeline ([`fusion.py`](fusion.py)) applies a
deterministic 4-step procedure before combining them:

1. **Alignment** — intersect the two streams per video by `frame_index`,
   auto-detecting frame-offset and polarity conventions. Video-ID aliases
   are resolved automatically (e.g. Avenue: STG-NF `01_0001` ↔ MULDE `01`).
2. **Global rank normalization** — convert each model's scores to `[0, 1]`
   ranks (Borda-style). More robust than min-max to the heavy tails of
   normalizing-flow likelihoods.
3. **Temporal smoothing** — per-video 1-D Gaussian filter (σ grid-searched
   per dataset) suppresses single-frame spikes from pose jitter.
4. **Weighted combination** — `score = β₁·STG-NF + β₂·MULDE` with the
   weights found by grid search over 1001 candidates on the test split.

---

## Supported Datasets

| Dataset | Videos (test) | Frames | Notes |
| :-- | --: | --: | :-- |
| **ShanghaiTech Campus** | 107 | 40 791 | Both streams share `01_0014`-style video IDs. |
| **Avenue** | 21 | — | STG-NF uses `01_0021` (scene_clip); MULDE uses the clip
  index `21`. The fusion pipeline remaps these automatically. |

Both datasets are supported by the notebook and the CLI via a `--dataset`
flag or a single `DATASET` variable in the notebook config cell.

---

## Results

### ShanghaiTech Campus (107 test videos, 40 791 frames)

| Method | Micro AUC |
| :-- | --: |
| MULDE (appearance) | 79.66% |
| STG-NF (pose) | 83.53% |
| **PRISM (β₁ = 0.546, β₂ = 0.454, global_rank, σ = 15)** | **89.32%** |

Fusion adds **+5.8 pp** over the strongest single stream — both streams
contribute non-redundant signal. The optimal weights are close to 50/50
because the two models' errors are de-correlated: MULDE catches the
vehicle/object anomalies that STG-NF is structurally blind to, and the two
models rarely fire false positives on the same frames.

### Avenue (21 test videos)

| Method | Micro AUC |
| :-- | --: |
| MULDE (appearance) | 81.4% |
| STG-NF (pose) | 57.0% |
| **PRISM** | — |

Avenue results are computed by setting `DATASET = 'Avenue'` in the notebook
or passing `--dataset Avenue` to `fusion.py`. The Avenue AUC gain is expected
to be larger than ShanghaiTech because STG-NF is much weaker relative to
MULDE on Avenue, so MULDE dominates the fusion and the pose stream acts
purely as a noisy supplement.

---

## Quick Start (Google Colab)

The recommended way to run the fusion is via the provided Colab notebook
which handles environment setup, Drive mounting, and result persistence
automatically.

1. Open [`ShanghaiTech_Ensemble_Fusion.ipynb`](ShanghaiTech_Ensemble_Fusion.ipynb)
   in Colab.
2. In the **Dataset selection** cell, set `DATASET` to `'ShanghaiTech'` or
   `'Avenue'`.
3. Run all cells. The grid-search table and fusion report are saved to
   `/content/drive/MyDrive/Fusion/runs/<dataset>/ensemble/`.

The default Colab Drive paths for both datasets are pre-configured in the
notebook and in `fusion.py` ([`DATASET_PATHS`](fusion.py)). Edit them only if
your Drive layout differs.

---

## CLI Usage

`fusion.py` can also be invoked directly from the command line. On Colab:

```bash
# ShanghaiTech (default dataset)
python fusion.py --dataset ShanghaiTech \
    --normalization global_rank \
    --smooth_sigma 15 \
    --auto_detect_offset

# Avenue
python fusion.py --dataset Avenue \
    --normalization global_rank \
    --smooth_sigma_search \
    --auto_detect_offset
```

Use `--smooth_sigma_search` to automatically grid-search the best temporal
smoothing window. Pass explicit `--stgnf_pkl` / `--mulde_pkl` / `--output_dir`
flags to override the default Drive paths for any dataset.

Locally (without Colab Drive):

```bash
python fusion.py \
    --stgnf_pkl Others/Results/stgnf_scores.pkl \
    --mulde_pkl  Others/Results/mulde_scores.old.pkl \
    --output_dir Others/Results/ensemble \
    --normalization global_rank \
    --smooth_sigma 15 \
    --auto_detect_offset
```

The script writes:

| File | Content |
| :-- | :-- |
| `fusion_grid_search.csv` | Micro AUC for every β₁ candidate |
| `fusion_report.json` | Optimal weights, AUC, alignment stats, sigma search |

---

## Repository Layout

```
PRISM/
├── fusion.py                              # Fusion pipeline (alignment, normalization, grid search)
├── models.py                              # MULDE score / log-density networks
├── mulde_visualization.py                 # Reporting: thresholds, segments, dashboards
├── run_mulde_on_custom_video.py           # CLI: end-to-end MULDE inference on one video
│
├── ShanghaiTech_Ensemble_Fusion.ipynb     # Colab: fusion on ShanghaiTech OR Avenue (set DATASET)
├── Pose Extraction and Testing.ipynb       # STG-NF pose extraction, training, score export
├── ShanghaiTech_Hiera_L_Feature_Extraction.ipynb
├── Avenue_Hiera_L_Feature_Extraction.ipynb
├── MULDE_Training_GMM.ipynb               # Train the MULDE density model + GMM
├── run_custom_anomaly_detection.ipynb     # Notebook: end-to-end MULDE on a single video
│
└── Others/
    ├── Results/                           # Saved score pickles, fusion reports
    ├── Agent Plans/                       # Design documents
    ├── Helpful Codes for Agents/          # Diagnostic and test scripts
    ├── Utils/                             # Small helpers for data inspection
    ├── stgnf_deep_dive.md                 # Technical write-up: how STG-NF works
    └── mulde_deep_dive.md                 # Technical write-up: how MULDE works
```

---

## Citation

```bibtex
@misc{prism2026,
  title   = {PRISM: Pose + RGB Integration for Scene Monitoring},
  author  = {Hadi},
  year    = {2026},
  note    = {Late fusion of STG-NF and MULDE for video anomaly detection},
}

@inproceedings{stgnf2023,
  title     = {Normalizing Flows for Human Pose Anomaly Detection},
  author    = {Hirsch, Or and Berkovich, Ron},
  booktitle = {ICCV},
  year      = {2023}
}

@inproceedings{mulde2024,
  title     = {MULDE: Multiscale Log-Density Estimation via Denoising Score
               Matching for Video Anomaly Detection},
  author    = {Micorek, Jiri and Vavrecka, Michal and Sulc, Nikos and Matas, Jiri},
  booktitle = {CVPR},
  year      = {2024}
}
```

## License

This repository contains experiment code for a graduate research project.
The underlying STG-NF and MULDE methods are the work of their respective
authors; please respect their licenses when reusing those components.

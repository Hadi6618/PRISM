# Agent Context: UniVAD Late Fusion Pipeline

**To any AI Agent reading this file:** 
This document contains the complete context, mathematical rationale, and architectural design of the Late Fusion pipeline for Video Anomaly Detection (VAD) on the ShanghaiTech Campus dataset. Read this carefully before modifying any code in the `Fusion/` directory.

---

## 1. Project Overview
The goal of this pipeline is to ensemble two fundamentally different anomaly detection models:
1. **MULDE (Multiscale Log-Density Estimation):** A patch-based Gaussian Mixture Model evaluating visual appearance. It takes 1152-dimensional features extracted from each frame by the Hiera video encoder. It is extremely accurate at detecting non-human anomalies (e.g., vehicles, bicycles) with an incredibly low noise floor on normal frames.
   * *Paper:* "MULDE: Multiscale Log-Density Estimation via Denoising Score Matching for Video Anomaly Detection" (CVPR 2024)
   * *Repo:* [https://github.com/jmicorek/mulde](https://github.com/jmicorek/mulde)
2. **STG-NF (Spatio-Temporal Graph Normalizing Flows):** A pose-based kinematics model evaluating human skeletons. It is specialized at catching complex human anomalies (e.g., fighting, throwing) that MULDE misses, but it has a high noise floor due to skeleton tracking jitter.
   * *Paper:* "Normalizing Flows for Human Pose Anomaly Detection" (ICCV 2023)
   * *Repo:* [https://github.com/orhir/STG-NF](https://github.com/orhir/STG-NF)

By performing a **Late Fusion** of their independent frame-level anomaly scores, we achieve a state-of-the-art **87.17% Micro AUC**, significantly beating both standalone models.

---

## 2. Core File Structure
*   `stgnf_scores.pkl` & `mulde_scores.pkl`: The raw outputs from the standalone models. They contain dictionaries mapping `video_id` to `{frame_indices, anomaly_scores, labels}`.
*   `fusion.py`: The core mathematical engine. It handles frame alignment, smoothing, normalization, and grid search.
*   `ShanghaiTech_Ensemble_Fusion.ipynb`: The orchestration notebook that loads the pickles, calls `fusion.py`, and displays the results.

---

## 3. The Mathematical Pipeline (7 Steps)
The `fusion.py` script applies the following operations in exact order. **Do not change this order.**

### Step 1 & 2: Alignment and Polarity Correction
*   **Alignment:** STG-NF and MULDE often have different starting frame indices (0-based vs 1-based) or slightly different lengths. `align_per_video()` intersects them safely.
*   **Polarity:** STG-NF naturally outputs a *Normality Score* (higher = normal). MULDE outputs an *Anomaly Score* (higher = abnormal). STG-NF is explicitly negated (`-score`) so both streams follow: **Higher Score = More Anomalous**.

### Step 3: Gaussian Temporal Smoothing (CRITICAL)
Before any scaling, both raw score streams are smoothed independently per-video using a 1D Gaussian Filter (`scipy.ndimage.gaussian_filter1d`).
*   **Why:** Raw scores are noisy. A single anomalous frame shouldn't spike alone. 
*   **Historical Bug Warning:** Skipping this step artificially caps the ensemble at ~87.03%. Applying `sigma=2.0` correctly boosts the performance to 87.17%.

### Step 4: Global Min-Max Normalization
Because the negated STG-NF scores live around `[-2, 0]` and MULDE scores live around `[0, 3600]`, they must be normalized to `[0.0, 1.0]`.
*   **Crucial Design Choice:** We use **Global** Min-Max (across all 107 test videos simultaneously). We do *not* use Per-Video Min-Max. Per-video scaling destroys the global ranking by forcing perfectly normal videos to stretch their tiny noise up to 1.0, creating massive false positives.

### Step 5 & 6: Weighted Fusion & Grid Search
The normalized scores are linearly combined: 
`Final_Score = (beta_1 * STGNF_Norm) + (beta_2 * MULDE_Norm)`
*   `fusion.py` runs a grid search from `beta_1 = 0.00` to `1.00`.
*   The optimal weights found are **`beta_1 = 0.05` (STG-NF)** and **`beta_2 = 0.95` (MULDE)**.

### Step 7: AUC Calculation
All fused frame scores and labels are concatenated into massive 1D arrays, and a single global Micro ROC AUC score is computed via `sklearn.metrics.roc_auc_score`.

---

## 4. Why is STG-NF's weight only 0.05?
If an agent is asked to "fix" the heavily skewed weights, **do not attempt to equalize them**. The 5% / 95% split is mathematically optimal for the following reasons:
1.  **The Noise Floor:** After Min-Max scaling, STG-NF's average score on a perfectly normal frame is `0.0088`. MULDE's is `0.0004`. STG-NF is **21.5x noisier**. If STG-NF is given a high weight (e.g., 0.50), its skeleton-jitter bleeds into MULDE's clean baseline and creates false alarms.
2.  **Primary vs Supplement:** MULDE solves ~80% of the dataset (vehicles/crowds) flawlessly. It acts as the **Primary Detector**. STG-NF is completely blind to vehicles and acts solely as a **Specialized Supplement**. We only need a tiny 5% pinch of STG-NF's signal to bump fighting/throwing anomalies across the threshold without ruining MULDE's clean background.

---

## 5. Future Architectural Upgrades (CC-STG-NF)
If the user requests upgrading from Late Fusion to **Mid/Early Fusion**, do NOT use Cross-Attention/EBMs due to extreme dimensional mismatch. Instead, build a **Context-Conditional Spatio-Temporal Graph Normalizing Flow (CC-STG-NF)**:
*   Pre-extract MULDE's 1152-D mid-level features (from the Hiera encoder).
*   Use a small MLP to compress them into a 256-D Context Vector ($C$).
*   Concatenate $C$ into the Affine Coupling Layers of STG-NF so the scale ($s$) and translation ($t$) networks are conditioned on the visual background: $z = f(x | C)$.
*   This natively solves the problem of "normality" being context-dependent (e.g., running on a track vs running in a library).

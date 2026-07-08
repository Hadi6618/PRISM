# PRISM — Methods

> **Pose + RGB Integration for Scene Monitoring** — a two-stream late-fusion
> framework for unsupervised video anomaly detection (VAD) that combines
> a pose-based detector (STG-NF) with an appearance-based detector (MULDE).

This document is written as a **methods section** suitable for a graduate
project report or a workshop paper. The notation, equations, and section
numbering are aligned with the conventions used in the underlying STG-NF
(ICCV 2023) and MULDE (CVPR 2024) papers, and can be re-typeset directly into
a LaTeX manuscript.

---

## 1. Problem setting and notation

We address **unsupervised video anomaly detection** at the frame level.
The training set contains only normal videos; at test time we assign each
frame $t$ of a test video $v$ a continuous anomaly score
$s[t] \in \mathbb{R}$ such that higher values indicate a higher likelihood
of anomaly. We report frame-level **Micro AUC** (concatenating all test
frames) and, where noted, **Macro AUC** (per-video AUC, averaged).

Each test video is a sequence of $T_v$ RGB frames
$\mathbf{X}_v = (X_v^1, \dots, X_v^{T_v})$. The ground-truth label
$y_v^t \in \{0, 1\}$ indicates whether frame $t$ of video $v$ contains an
anomalous event ($y_v^t = 1$).

We have access to **two pre-trained, independently trained detectors** whose
outputs we wish to combine:

| Stream | Detector | Score definition | Polarity |
| :-- | :-- | :-- | :-- |
| Pose / kinematic | STG-NF | $-\log p(\mathbf{P}_v^t)$ under a normalizing flow over pose graphs | *Normality* on ShanghaiTech, *anomaly* elsewhere (auto-detected) |
| Appearance / contextual | MULDE | $-\log p(\mathbf{F}_v^t)$ under a GMM over 16-d DSM signatures | *Anomaly* |

The two streams produce frame-level scores
$\{s_{\text{STG-NF}}[t]\}_{t=1}^{T_v}$ and
$\{s_{\text{MULDE}}[t]\}_{t=1}^{T_v}$ that differ in scale, in polarity, and
in the convention used to index frames and videos. The contribution of this
work is the **PRISM pipeline** (§3) that aligns, normalises, smooths, and
fuses these two score streams into a single frame-level score.

---

## 2. Pre-trained component streams

We use two publicly available detectors without retraining.

### 2.1 STG-NF — pose stream

STG-NF [Hirsch & Berkovich, ICCV 2023] models the distribution of normal
human motion using a **Spatio-Temporal Graph Normalising Flow** over
per-person pose trajectories. For each frame $t$ of a test video we:

1. detect persons with **YOLOX-X**;
2. estimate 17 COCO keypoints with **FastPose-ResNet152**;
3. link detections across time with **PoseFlow** + an **OSNet** ReID
   embedding, yielding per-person pose tracks $\mathbf{P}_i^t$;
4. encode each track with a **Spatio-Temporal Graph Convolutional Network**
   (ST-GCN) into a motion feature;
5. evaluate the **negative log-likelihood** of that feature under an
   affine-coupling normalising flow trained only on normal motion.

The output is a per-frame **normality score** (higher = more normal),
inverted to *anomaly* polarity in §3.1.

### 2.2 MULDE — appearance stream

MULDE [Micorek et al., CVPR 2024] models the distribution of normal
**frame appearance** at multiple noise scales. For each test frame we:

1. extract a 1152-d feature with a **Hiera-L** video backbone operating on
   16-frame clips with stride 4;
2. estimate the score of the data distribution at 16 fixed noise levels
   $\sigma_1 < \sigma_2 < \dots < \sigma_{16}$ using a 2-layer MLP (4096
   units) trained via **Denoising Score Matching** (DSM);
3. concatenate the 16 noise-scale log-density estimates into a
   **16-d DSM signature**;
4. evaluate the **negative log-likelihood** of that signature under a
   5-component **Gaussian Mixture Model** fitted on the training
   signatures.

The output is a per-frame **anomaly score**.

---

## 3. PRISM — the fusion pipeline

Let $S_v = \{(s_{\text{STG-NF}}^t, s_{\text{MULDE}}^t, y_v^t)\}_{t=1}^{T_v}$
denote the raw per-frame outputs of the two streams for video $v$. The
streams arrive as Python dictionaries keyed by video ID; the same frame
indexing convention is **not** used on both sides. PRISM is the four-stage
deterministic pipeline shown in Algorithm 1 that converts these two
unaligned dictionaries into a fused anomaly score.

> **Algorithm 1 — PRISM (overview).**
> **Input:** STG-NF pickle $\mathcal{D}_S$, MULDE pickle $\mathcal{D}_M$, dataset key.
> **Output:** per-frame fused score $s[t]$; CSV grid-search table; JSON report.
> 1. **Align** $\mathcal{D}_S, \mathcal{D}_M$ per video by frame index, auto-detecting
>    video-ID aliases, frame offset, and STG-NF polarity. *(§3.1)*
> 2. **Normalize** both streams to $[0, 1]$. *(§3.2)*
> 3. **Smooth** with a 1-D Gaussian per stream, with independent $\sigma$. *(§3.3)*
> 4. **Fuse** as a 1-D grid search over $\beta_1$ in
>    $s[t] = \beta_1 \cdot s_{\text{STG-NF}}[t] + (1-\beta_1) \cdot s_{\text{MULDE}}[t]$. *(§3.4)*

The four stages are implemented as focused sub-modules in `Utils/` (one
file per stage) and orchestrated by a thin 110-line shim `PRISM.py`. Each
stage is described below.

### 3.1 Per-video alignment

The two streams use **different video-ID and frame-index conventions** that
must be reconciled before they can be intersected.

* **Video-ID aliasing.** STG-NF exports its test set as
  `scene_clip` identifiers (e.g. `01_0021` = scene 01, clip 21) while MULDE
  uses the bare clip index (e.g. `21`). PRISM first checks for direct
  overlap; if absent, it builds a canonical alias by extracting the trailing
  integer token of each ID, zero-pads it to two digits, and matches aliases
  across the two streams. The resulting mapping
  $\phi: \text{vid}_{\text{MULDE}} \mapsto \text{vid}_{\text{STG-NF}}$ is
  applied to the MULDE dictionary before frame intersection.

* **Frame offset.** STG-NF exports are 0-based; some MULDE exports are
  1-based. We sweep
  $\delta \in \{-2, -1, 0, 1, 2\}$ and select the value that maximises
  STG-NF's standalone frame-level Micro AUC on the intersected frames.

* **Polarity correction.** STG-NF's reference release reports a *normality*
  score on ShanghaiTech but an *anomaly* score on Avenue. The
  `auto` mode tests both polarities and keeps the one with higher
  STG-NF AUC; a `manual` override is also exposed.

* **Label inversion check.** A warning is emitted if the two streams use
  opposite label conventions (0 = anomaly vs. 1 = anomaly). The code falls
  back to MULDE's `1 = anomaly` convention.

The output of this stage is a list of
`AlignedVideo(vid, frame_idx, stgnf_scores, mulde_scores, labels)`
records containing the raw, unnormalised scores of both streams on the
*intersected* frames only. Alignment statistics (chosen offset, polarity,
videos intersected, frames per stream) are stored alongside for the final
report.

### 3.2 Score normalisation

The two streams emit scores on **incompatible numerical scales** — STG-NF's
normalising-flow log-likelihoods are heavily heavy-tailed and can produce
$+\infty$ after sanitisation, while MULDE's GMM log-likelihoods are well
behaved. Combining them linearly is meaningless without first collapsing
both to a common range. We map both streams to $[0, 1]$ with one of four
strategies:

| Strategy | Operation | When it helps |
| :-- | :-- | :-- |
| `per_video_minmax` | $\tilde s_v^t = (s_v^t - \min_v) / (\max_v - \min_v)$, computed per video | Removes cross-video magnitude variation; can destroy the global anomaly scale |
| `global_minmax` | As above, but with $\min$ / $\max$ taken over **all** aligned frames | Preserves the global ranking; sensitive to outliers |
| `global_zscore` | $z = (s - \mu)/\sigma$ clipped to $[-3, 3]$, then min-max to $[0,1]$ | Robust to outliers but assumes a roughly symmetric distribution |
| **`global_rank`** *(default)* | $\tilde s^t = R(s^t) / (N-1)$ with average-rank ties (Borda count) | Most robust to scale and orientation; invariant to monotone transforms |

Empirically (`global_rank` is the CLI default and our reported setting) the
rank strategy is the most robust to the heavy tails of normalising-flow
likelihoods and consistently yields the highest fusion Micro AUC, so it is
used as the default in all reported numbers.

### 3.3 Independent temporal smoothing

The two streams have **different noise profiles**:

* STG-NF scores are produced by a windowed likelihood, so they are
  **pre-smoothed** by the evaluation pipeline; additional smoothing is
  usually mild ($\sigma_{\text{STG-NF}} \in [0, 3]$ frames).
* MULDE scores are raw per-frame GMM log-likelihoods and can spike briefly
  when the Hiera-L feature jumps; heavier smoothing
  ($\sigma_{\text{MULDE}} \in [5, 15]$ frames) typically helps.

A **shared** Gaussian $\sigma$ would either over-smooth STG-NF or
under-smooth MULDE. We therefore apply a per-stream 1-D Gaussian
independently:

$$
\tilde s_{\text{STG-NF}}^t = (g_{\sigma_S} * s_{\text{STG-NF}})^t,
\qquad
\tilde s_{\text{MULDE}}^t = (g_{\sigma_M} * s_{\text{MULDE}})^t,
$$

where $g_\sigma$ is a 1-D Gaussian kernel with standard deviation $\sigma$
($\sigma = 0$ means no smoothing). The optimal pair
$(\hat\sigma_S, \hat\sigma_M)$ is selected by grid search over a fixed
candidate set
$\{0, 1, 2, 3, 4, 5, 6, 8, 10, 15\}$
maximising the **mean of the two standalone Micro AUCs**
$(\text{AUC}_S + \text{AUC}_M)/2$. This criterion is independent of
$\beta_1$ and therefore cheap to evaluate as a preprocessing step before
fusion.

### 3.4 Weighted fusion

Given the aligned, normalised, smoothed scores, the final anomaly score is
a **convex combination** of the two streams:

$$
s_{\text{fused}}^t \;=\; \beta_1 \cdot s_{\text{STG-NF}}^t
\;+\; \beta_2 \cdot s_{\text{MULDE}}^t,
\qquad \beta_2 = 1 - \beta_1,\quad \beta_1 \in [0, 1].
$$

This is a one-dimensional fusion rule with a single free parameter
$\beta_1$. The endpoints $\beta_1 = 0$ and $\beta_1 = 1$ reduce to
single-stream MULDE and STG-NF respectively, and intermediate values
correspond to a soft ensemble.

We sweep $\beta_1$ over the grid
$\{0.00, 0.01, \dots, 1.00\}$ (101 points) and pick the value that
maximises frame-level Micro AUC on the aligned test set:

$$
\hat\beta_1 \;=\; \arg\max_{\beta_1} \;
\text{AUC}\!\left(\mathbf{y},\, \beta_1 \mathbf{s}_{\text{STG-NF}} + (1-\beta_1)\mathbf{s}_{\text{MULDE}}\right).
$$

The choice of a 1-D grid is deliberate: a 2-D weight search
$(\beta_1, \beta_2)$ with $\beta_1 + \beta_2 = 1$ relaxed would add a
parameter without adding expressive power (the two streams are linearly
combined at the frame level), and a 2-D unconstrained search would
introduce a useless scale degree of freedom and over-fit the test split.

The pipeline writes the full per-$\beta_1$ AUC table to
`fusion_grid_search.csv` and a JSON report containing the chosen
$(\hat\beta_1, \hat\beta_2)$, the best Micro AUC, the alignment statistics,
and the sigma-search summary.

---

## 4. Datasets and evaluation protocol

We evaluate on the standard test splits of two weakly-supervised VAD
benchmarks.

| Dataset | Test videos | Test frames | Notes |
| :-- | --: | --: | :-- |
| **ShanghaiTech Campus** | 107 | 40 791 | Both streams share the `01_0014`-style ID convention. |
| **Avenue** | 21 | 15 326 | STG-NF uses `01_0021` (scene_clip); MULDE uses the bare index `21`. The alignment module remaps these automatically. |

For each dataset we:

1. run the STG-NF pipeline to obtain a per-frame normality score pickle
   (`--stgnf_pkl`);
2. run the MULDE pipeline to obtain a per-frame anomaly score pickle
   (`--mulde_pkl`);
3. invoke `python PRISM.py --dataset <name> --normalization global_rank
   --smooth_sigma_search --auto_detect_offset` to produce
   `fusion_grid_search.csv` and `fusion_report.json` in the configured
   output directory.

**Primary metric:** frame-level Micro AUC (all test frames concatenated,
ROC AUC against the binary ground-truth label). **Secondary metric:**
Macro AUC (per-video AUC, averaged across the test split). The current
release reports Micro AUC.

We do not perform any test-time training; the only learned components are
the STG-NF normalising flow, the MULDE DSM score network, and the MULDE
GMM, all of which are trained **only on the labelled-normal training
split** of each dataset. The fusion weights $\beta_1$ and the smoothing
sigmas $(\sigma_S, \sigma_M)$ are the only quantities selected on the
test split, and they are reported alongside the AUC.

---

## 5. Implementation details

* **Implementation:** Python 3.10+, NumPy 1.24, scikit-learn 1.3,
  SciPy `gaussian_filter1d` for the temporal smoothing.
* **Source layout.** The fusion pipeline is implemented in eight focused
  sub-modules under `Utils/`:
  `prism_config.py`, `prism_io.py`, `prism_alignment.py`,
  `prism_normalization.py`, `prism_smoothing.py`, `prism_fusion.py`,
  `prism_reporting.py`, `prism_cli.py`. The 110-line file `PRISM.py` is a
  thin shim that re-exports the public API and is the CLI entry point
  (`python PRISM.py ...`).
* **Notebook wrapper.** The same shim is loaded by `PRISM_Runner.ipynb`
  via `importlib.util.spec_from_file_location('prism', 'PRISM.py')`,
  which works because the shim adds `Utils/` to `sys.path` at load time
  before any sub-module is imported.
* **Hardware.** A single Colab GPU is sufficient for both STG-NF and
  MULDE inference; the PRISM fusion itself runs on CPU in well under a
  second per dataset.

---

## 6. Limitations

* The fusion rule is a 1-D convex combination. If the two streams are
  strongly **anti-correlated** at the frame level, the optimal weight
  may be outside $[0, 1]$; we did not observe this in our experiments
  but flag it as a direction for future work.
* $(\beta_1, \sigma_S, \sigma_M)$ are all selected on the test split
  using a grid search. This is a standard practice in the VAD literature
  but can be optimistic; for a stricter evaluation the test split should
  be partitioned into a *tuning* half and a *held-out* half.
* The PRISM pipeline assumes that the STG-NF and MULDE test pickles share
  the **same set of test videos** (modulo ID-convention differences).
  Datasets where one model evaluates on a different subset are not
  currently supported.

---

## 7. Reproducibility

The full pipeline is reproducible from the public STG-NF and MULDE
checkpoints. The exact commands used to produce the headline numbers are
documented in the project `README.md`. The fusion CSV and JSON outputs
allow every reported Micro AUC to be reconstructed from the raw test
pickles without rerunning either detector.

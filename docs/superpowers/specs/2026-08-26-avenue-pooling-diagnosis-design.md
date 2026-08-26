# Avenue Pooling Diagnosis Design

**Date:** 2026-08-26
**Status:** Approved design; implementation plan pending
**Scope:** Diagnostic only — no re-extraction, no retraining, no tracker change, no PRISM changes

## 1. Context and Evidence

Two independent Avenue pose-extraction stacks land on the same Micro AUC plateau:

| Extraction stack | Avenue Micro AUC |
|---|---|
| AlphaPose (YOLOX-X + FastPose + PoseFlow/OSNet) | ~64.2% |
| YOLO26 + ByteTrack + ViTPose++ base | ~64.14% |

The second stack produces measurably better keypoints, especially on small and occluded people. If pose *quality* were the binding constraint, the swap should have moved the score. It did not. The working hypothesis is therefore that the constraint lives **downstream of extraction**, in how per-person segment scores are turned into frame-level anomaly scores.

Review of the scoring path (`utils/scoring_utils.py`, `utils/data_utils.py`, `utils/pose_utils.py`, `models/training.py`) identifies four concrete mechanisms that create a crowd-noise floor:

1. **Global min-pooling over people.** `get_clip_score` ends with
   `clip_score = np.amin(clip_ppl_score_arr, axis=0)`: the most anomalous
   person drives every frame. In a crowd, a single jittery background person
   can score a frame anomalous.
2. **Size blindness.** `normalize_pose` removes the mean position and divides
   by the y-coordinate std, so a 40-px far-away crowd person and a 300-px
   anomaly person are normalised to the same scale. The flow cannot
   distinguish "small person jitter" from "close person running". A size
   prior can only be applied at pooling time, not inside the model.
3. **Unused confidence.** `train_seg_conf_th` defaults to `0.0` (the segment
   filter is implemented but never applied) and `model_confidence` defaults
   to `False` (per-keypoint confidence is not fed to the model, although the
   `nll * score` weighting is implemented).
4. **No-detection frames become max-anomaly.** Frames with no tracked person
   are filled with `inf` in the per-person arrays and replaced by the global
   maximum score after concatenation — the opposite of what a missing-person
   frame should mean. The export additionally applies six cumulative Gaussian
   passes (σ = 1..6), which smears brief anomalies.

Observation from the dataset itself: Avenue test anomalies are consistently
close to the camera, large, and sharp, while the background crowd in both
train and test is far away, small, and jittery. The diagnostic below tests
whether that structure is exploitable at the scoring level.

## 2. Goal

Determine whether the frame-level aggregation (global min over people,
size-blind normalisation, unused confidence) is what caps Avenue at ~64%,
using only existing artifacts:

- the 16 Avenue training + 21 Avenue test ViTPose/ByteTrack tracked-person
  JSONs from the `Avenue_dataset_vitpose` run (Drive root
  `/content/drive/MyDrive/STG-NF/Avenue_dataset_vitpose/`);
- the locked STG-NF checkpoint and complete argument set that produced
  ~64.14% (`logs/Avenue/64_2/checkpoint_64_2.pth.tar`, epoch 3, Micro AUC
  64.1421%, Macro AUC 65.93%);
- the existing STG-NF loader, model, and scoring code.

The primary deliverable is a decision: confirmed (scoring is the constraint),
or refuted (scoring is not the constraint), with quantified evidence.

## 3. Non-Goals

This diagnostic will not:

- re-extract pose/track data from videos;
- retrain or fine-tune STG-NF;
- change the tracker or detector;
- change the STG-NF architecture, training hyperparameters, or the
  evaluation convention (labels, polarity, frame indexing, smoothing) of the
  *baseline* path;
- touch MULDE or the PRISM fusion pipeline;
- claim a new headline number — alternative-pooling AUCs use a new scoring
  convention and establish a ceiling, not a result comparable to the
  reported 64.2%.

## 4. Part A — JSON-only statistics (no GPU)

Compute from the tracked-person JSONs and the existing dataset object:

- **Person size per segment:** p90 of the keypoint-bbox height (keypoints
  with confidence ≥ 0.25), averaged over the segment's frames. The
  percentile, not the max, keeps mid-stride poses from distorting the
  estimate.
- **Mean detection confidence per segment:** already available as
  `segs_score_np` from `PoseSegDataset`; no recomputation needed.
- **Distributions:** person size and confidence in train vs test; fraction
  of test segments that are "crowd-sized" (keypoint height < 80 px) vs
  "foreground-sized" (keypoint height ≥ 160 px), using the same θ
  thresholds as Part B.

Questions answered:

- Does the flow train predominantly on crowd-sized people while test
  anomalies live at foreground sizes?
- How much of the test segment mass is low-confidence or crowd-sized?

## 5. Part B — One model pass + alternative pooling

Recompute per-segment normality scores with the locked checkpoint through the
exact code path of the control run (`get_dataset_and_loader` + `trainer.test`),
then rebuild per-person per-frame score arrays exactly as `get_clip_score`
does and replace the global `amin` with each variant:

1. **Baseline global min** — must reproduce ~64.14% (pipeline-integrity
   sanity check).
2. **Size gate** — min over people whose keypoint height ≥ θ, θ ∈ {80, 120,
   160} px.
3. **Size-weighted min** — weight each person's score by size^α, α ∈ {0.5, 1}.
4. **Robust k-th smallest** — k ∈ {2, 3, 5} smallest over people per frame.
5. **Confidence gate** — drop segments with mean detection confidence < τ,
   τ ∈ {0.3, 0.5, 0.7}.
6. **Confidence-weighted** — segment score × mean confidence before pooling
   (what `model_confidence=True` approximates without retraining).

For every variant, report:

- Micro AUC and Macro AUC (per-video mean, one-class videos skipped);
- **Attribution:** for the top 1% (minimum 100) false-positive frames and
  the true-positive frames, the size and confidence of the argmin person —
  the direct test of the background-crowd hypothesis.

Output: a CSV/JSON table of all variants, saved next to the run artifacts.

## 6. Decision Rule

- The baseline reproduces ~64.14% → the experiment is trustworthy.
- Any variant gains ≥ ~1–2 AUC points over the baseline → hypothesis
  confirmed; the next spec designs the full scoring experiment (confidence
  gating + size-weighted pooling + `seg_len`/smoothing changes) on the
  existing extraction.
- All variants flat → scoring is ruled out cheaply; the constraint is
  identity continuity (BoT-SORT/ReID + track stitching path) or model
  capacity (STG-NF tuning), and effort moves there.

## 7. Caveats and Risks

| Risk | Mitigation |
|---|---|
| Variant AUCs use a new scoring convention | Report them as a ceiling, not as comparable headline numbers; keep the baseline path byte-identical to the control. |
| Keypoint-bbox height is a size proxy | Use the p90 y-extent, and report the size distributions so the proxy's behaviour is visible. |
| One training/eval run is not statistically significant | The diagnostic is a go/no-go gate; any promising variant is confirmed by the follow-up experiment before being claimed. |
| Pooling changes interact with PRISM fusion weights | Explicitly out of scope; fusion re-tuning happens only if a scoring change is adopted. |
| Reusing the wrong checkpoint/args breaks the comparison | Lock the exact control checkpoint path, args, and repo commit in the run manifest before Part B. |

## 8. Artifacts

1. Per-segment size/confidence statistics for train and test (CSV).
2. Alternative-pooling AUC table (CSV/JSON) with per-video rows.
3. Argmin-person attribution for false-positive and true-positive frames.
4. A run manifest recording the exact checkpoint, args, and input JSON paths.

## 9. Next Transition

This design is complete and intentionally stops before implementation. The
next step is a detailed implementation plan identifying the notebook edits,
the pooling-variant implementation, validation checks, and execution order.
No implementation begins until that plan is reviewed.

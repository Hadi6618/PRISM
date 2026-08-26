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

A clarifying note on VAD frame semantics: a frame is labelled abnormal if
*any* person is anomalous — five people walking normally while one lies on
the ground is an abnormal frame, and the lying person must drive the frame
score. The global min-pool is therefore the semantically correct aggregation
for these labels; it is **not** the bug. The real failure mode is that a
*garbage* person (false detection, jittery small track, occlusion artifact)
can also win the min on a frame where no real anomaly exists, raising the
false-positive floor. The diagnostic separates these two cases with the
argmin-person attribution (Section 5): do false positives come from small
crowd-sized people or from real foreground-sized people?

Observation from the dataset itself: Avenue test anomalies are consistently
close to the camera, large, and sharp, while the background crowd in both
train and test is far away, small, and jittery. The diagnostic below tests
whether that structure is exploitable at the scoring level.

## 2. Goal

Determine whether the Avenue ~64% cap is caused by the frame-level
aggregation (global min over people, size-blind normalisation, unused
confidence) or by unfiltered garbage detections reaching it — or neither —
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

Semantic note on the variants: under VAD frame labels ("abnormal if any
person is anomalous"), the true aggregation is a min over *valid* persons —
the lying person among five walkers must drive the frame. Variant 4
(k-th smallest) therefore violates this semantics: with a single anomalous
person, the 2nd-worst person is a normal walker and the anomaly is missed.
It is retained **as a diagnostic probe only**, to quantify how much a single
noise person dominates the global min; it is not a candidate final design.
The viable robustness direction is *filter-then-min*: exclude unreliable
persons (low confidence, tiny size — variants 2 and 5) or pull their scores
toward normal (variant 6) *before* taking the min, so the worst *real* person
still drives the frame. One residual limitation applies to every variant: if
the anomalous person is missed by detection entirely, no pooling change can
recover the frame — that is a detection/tracking responsibility.

For every variant, report:

- Micro AUC and Macro AUC (per-video mean, one-class videos skipped);
- **Attribution:** for the top 1% (minimum 100) false-positive frames and
  the true-positive frames, the size and confidence of the argmin person —
  the direct test of the background-crowd hypothesis.

Output: a CSV/JSON table of all variants, saved next to the run artifacts.

## 5b. Part C — Simulated stricter detection/tracking (JSON filtering, no re-extraction)

The alternative to scoring-side fixes is to prevent garbage at the source:
raise the detection confidence bar, require keypoint quality, ignore tiny
people, and drop unstable tracks — "do not detect them". This is testable at
zero cost because the tracked-person JSONs already carry per-frame detection
confidence (`scores`) and per-keypoint confidence, so stricter extraction can
be simulated by deleting records from the existing JSONs and re-running the
Part B baseline.

Filters (applied to both train and test JSONs):

1. **Detection-confidence floor** — drop person-frames with `scores < tau_det`,
   tau_det in {0.4, 0.5, 0.7}.
2. **Keypoint-quality floor** — drop a person-frame with fewer than K
   confident keypoints (conf >= 0.3), K in {6, 10}; rejects partial and
   heavily occluded detections.
3. **Minimum person size** — drop person-frames with keypoint height < theta,
   theta in {60, 80} px.
4. **Minimum track length** — drop tracks with fewer than L frames,
   L in {10, 24}.
5. **Combined** — the conjunction that keeps the most train coverage while
   removing the most test-time noise; reported as a table with per-filter
   train/test segment counts.

For every filter setting: re-run the exact Part B baseline (global min) and
report Micro/Macro AUC, then combine the best filter with the best scoring
variant from Part B.

Interpretation:

- Filtering alone breaks the plateau → extraction-side confirmation; the
  follow-up is a real re-extraction with those exact thresholds (already
  known from the simulation).
- Filtering is flat but a Part B scoring variant helps → the noise is real
  small-person keypoint jitter, not garbage detections; the scoring-side fix
  is confirmed.
- Both flat → identity continuity or model capacity; effort moves to the
  tracking path.

Caveats: (a) filtering also removes crowd people from *training*, changing
the learned "normal" distribution — intended, but report train segment
counts before/after; (b) simulated filtering is a ceiling, like the pooling
variants: a real higher-threshold extraction may also lose some valid
detections, so a positive result is confirmed only by the real run.

## 5c. Part D — Failure attribution and evidence (why a frame/clip scored badly)

The diagnostic also explains *why* individual frames and clips score badly,
so results are traceable to concrete objects rather than reported as bare
AUC numbers.

**End-to-end traceability.** Every flagged frame score is decomposed as:
`frame score → argmin person → track ID → segment → raw JSON pose + size +
confidence → reason code`. Every number in the report points at a concrete
row in the extracted JSONs. Descriptive attribution (who drove the score —
certain, from the min-pool argmin) is reported separately from causal claims
(what caused the AUC loss — established only by the counterfactual probes).

**Reason-code taxonomy** (every rule stated with its threshold so the
histogram is reproducible):

- `CROWD_JITTER` — argmin person is small (keypoint height < theta) with
  high frame-to-frame keypoint jitter;
- `GARBAGE_DET` — low detection confidence, fewer than K confident keypoints,
  or a merged box (keypoint span much wider than the person's box);
- `OCCLUSION_GAP` — the argmin track has an internal gap within ±seg_len of
  the flagged frame;
- `ID_SWITCH` — the argmin track is shorter than seg_len (newborn/reborn
  track);
- `NO_DET` — no person tracked at the frame (the inf → max-anomaly
  artifact);
- `REAL_ANOMALY` — true positive (correct behaviour);
- `MISSED_NO_DET` — anomaly-event frames where the anomaly person has no
  track;
- `MISSED_MASKED` — anomaly person tracked but outscored by a
  non-anomalous person;
- `MISSED_NORMALIZED` — anomaly person tracked and scored normal by the flow.

**Per-clip report.** For every test clip: per-video AUC and deficit vs
overall; each ground-truth anomaly *event* checked for detection (is the
score peak inside the event window?); reason-code histogram per clip. Events
are anchored to ground-truth windows, not bare label=1 frames.

**Aggregate evidence.** Reason-code histogram over the top-1% (minimum 100)
false-positive frames and over all missed anomaly events. The attribution
runs on BOTH extraction stacks (AlphaPose and ViTPose/ByteTrack) and the
histograms are compared: if both are dominated by `CROWD_JITTER`, that
explains the shared plateau directly — the same mechanism under different
extractors.

**Counterfactual probes (causation).**

- *Delete-person:* remove the argmin person's track from the JSON and
  re-aggregate. Free — other persons' segment scores are unchanged, so this
  is pure re-pooling with no model pass. A false positive that disappears
  was caused by that object.
- *Gap-interpolation:* simulate keypoint interpolation across the occlusion
  gap and score only the newly created segments (one small extra model
  pass). A missed event that recovers was caused by track fragmentation.

**Guardrails.** Reason-code thresholds are recorded in the run manifest;
descriptive vs causal claims are labelled in the report; events use
ground-truth windows.

## 6. Decision Rule

- The baseline reproduces ~64.14% → the experiment is trustworthy.
- A Part C filter setting (simulated stricter detection) alone gains
  ≥ ~1–2 AUC points → extraction-side hypothesis confirmed; the follow-up is
  a real re-extraction with those thresholds, which are already known from
  the simulation.
- A Part B pooling variant gains ≥ ~1–2 AUC points (with the best Part C
  filter held fixed) → scoring-side hypothesis confirmed; the next spec
  designs the full scoring experiment (confidence gating + size-weighted
  pooling + `seg_len`/smoothing changes).
- All variants flat → both scoring aggregation and detection filtering are
  ruled out cheaply; the constraint is identity continuity (BoT-SORT/ReID +
  track stitching path) or model capacity (STG-NF tuning), and effort moves
  there.
- The Part D reason histogram is the mechanism-level interpretation of
  whatever the AUCs show: a gain attributed to `GARBAGE_DET` removal
  confirms Part C; a gain driven by `CROWD_JITTER` frames confirms size
  weighting (Part B); `OCCLUSION_GAP`/`ID_SWITCH` dominance redirects to
  the tracking path even if the pooling variants are flat.

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
4. Filtering-AUC table with per-filter train/test segment coverage (CSV/JSON).
5. Per-clip attribution report with reason-code histograms and per-event
   detection checks (CSV/JSON), for both extraction stacks.
6. Counterfactual results: delete-person and gap-interpolation probes on the
   top-K false-positive frames and top-K missed events.
7. A run manifest recording the exact checkpoint, args, input JSON paths,
   and every reason-code threshold.

## 9. Next Transition

This design is complete and intentionally stops before implementation. The
next step is a detailed implementation plan identifying the notebook edits,
the pooling-variant implementation, validation checks, and execution order.
No implementation begins until that plan is reviewed.

# Avenue ViTPose + BoT-SORT Experiment Design

**Date:** 2026-08-25
**Status:** Approved design; implementation plan pending
**Scope:** Avenue pose extraction and STG-NF evaluation

## 1. Context and Evidence

The two-stage Avenue pipeline in `STG-NF_YOLO.ipynb` uses YOLO26 detection, ByteTrack, and ViTPose++ base. The current configuration is approximately:

- detector: `yolo26l.pt`;
- detector confidence: `0.3`;
- detector image size: `960`;
- tracker: `bytetrack.yaml`;
- pose model: `usyd-community/vitpose-plus-base`;
- pose output: COCO-17 keypoints converted to STG-NF's tracked-person JSON schema.

The notebook successfully generated all 16 Avenue training clips and all 21 test clips. The exported ViTPose run reports approximately `64.14%` Avenue Micro AUC and `65.93%` Macro AUC, which is effectively the same as the existing AlphaPose result reported as approximately `64.2%` Micro AUC. Lowering the detector threshold did not produce a meaningful improvement.

Multiple STG-NF configurations have plateaued at approximately `64.2%` Avenue Micro AUC, so the working hypothesis is now pose extraction rather than STG-NF optimization. The current ViTPose/ByteTrack extraction is therefore the primary control. The next run must isolate temporal tracking quality by changing the tracker while reusing the strongest existing STG-NF configuration unchanged.

## 2. Goal

Determine whether improving person identity continuity with appearance-assisted tracking raises Avenue STG-NF performance above the matched ViTPose/ByteTrack control. The AlphaPose result remains a secondary historical reference.

The primary success criterion is a reproducible improvement over the matched ByteTrack control, not a one-off increase caused by rounding or a changed evaluation protocol. Because the STG-NF configuration is held fixed, an improvement is attributed to the extracted pose/track data rather than to model tuning.

## 3. Non-Goals

This experiment will not:

- sweep detector confidence values again;
- change the ViTPose model size;
- change crop geometry, frame aspect handling, or keypoint order;
- change the STG-NF architecture, training hyperparameters, or score-processing settings;
- tune fusion weights or modify the MULDE stream;
- overwrite existing AlphaPose or ByteTrack pose artifacts;
- claim statistical significance from one training run.

If the result is inconclusive, later work may evaluate pose-stream complementarity in PRISM, but that is outside this experiment.

## 4. Alternatives Considered

### 4.1 BoT-SORT with explicit ReID: selected

Keep the detector and ViTPose model fixed, replace ByteTrack with a pinned custom BoT-SORT configuration, and explicitly enable appearance association. This directly tests whether identity fragmentation and short occlusions are limiting STG-NF. It has additional runtime and can create incorrect merges between visually similar people, so track diagnostics are required. The existing ViTPose/ByteTrack JSONs and scores are the matched control; AlphaPose is only a secondary reference because it uses a different pose estimator and detector/tracker stack.

### 4.2 BoT-SORT without ReID

This would test the association and motion model with lower runtime, but it is a weaker test of the identity-continuity hypothesis. A static Avenue camera also reduces the likely value of camera-motion compensation alone.

### 4.3 Post-hoc repair of ByteTrack tracks

This could be cheaper than re-extraction, but repair rules would modify the pose distribution after inference and be harder to defend as a clean model comparison. It is deferred.

## 5. Selected Extraction Design

The run is named `vitpose_botsort_reid_avenue`.

The fixed extraction settings are:

| Setting | Value |
|---|---|
| Dataset | CUHK Avenue |
| Detector | `yolo26l.pt` |
| Detection confidence | `0.3` |
| Image size | `960` |
| Pose model | `usyd-community/vitpose-plus-base` |
| Pose dataset index | `0` (COCO-17) |
| Aspect handling | Native frames; `SQUASH_TO_4_3=False` |
| Tracker | Custom, repository-pinned BoT-SORT YAML |
| ReID | Enabled with `with_reid=True` |
| ReID model | `auto`; abort if unsupported rather than silently substituting another model |
| Camera motion compensation | `none` for the static Avenue camera |

The custom tracker file must preserve `tracker_type: botsort`. All association, track-buffer, proximity, and appearance thresholds remain at the installed version's official BoT-SORT defaults; the only intentional overrides are `with_reid: True`, `model: auto`, and `gmc_method: none`. The exact file content or a cryptographic hash must be stored in the run manifest. The installed Ultralytics version must also be recorded because tracker defaults and supported fields can change between versions. If `model: auto` or any required field is unsupported, the run stops before extraction rather than silently substituting a different tracker configuration.

The extraction loop keeps the current two-stage order:

```text
Avenue frame -> YOLO26 person boxes -> BoT-SORT/ReID track IDs
             -> ViTPose person crops -> COCO-17 tracked-person JSON
```

The existing conversion contract remains unchanged:

```text
{person_id: {zero_padded_frame: {
    "keypoints": [x, y, confidence] * 17,
    "scores": scalar_detection_confidence
}}}
```

Keypoint confidence remains in each `[x, y, confidence]` triple. The scalar detection confidence remains in `scores`, matching STG-NF's current loader and segment-score handling.

## 6. Extraction Diagnostics

The extraction manifest records one machine-readable entry per clip and a run-level configuration record. Each clip entry includes:

- decoded frame count;
- frames with person detections;
- frames with assigned track IDs;
- total person instances;
- number of unique track IDs;
- per-track first frame, last frame, length, and gap count;
- track fragmentation indicators, including the number of short tracks and tracks with internal gaps;
- mean and selected percentiles for detector confidence;
- mean and selected percentiles for keypoint confidence;
- malformed-result, mismatch, and failure counts;
- output path and completion status.

These measurements are diagnostic, not labels or training inputs. They allow a flat AUC result to be interpreted as either a tracking failure, a pose-quality limitation, or an STG-NF limitation.

Extraction fails loudly for the affected clip when it encounters:

- missing expected clips;
- malformed JSON;
- a pose-result and detection-box count mismatch;
- duplicate `(track_id, frame)` records;
- invalid COCO-17 shape or non-finite coordinates/confidences;
- a clip with no usable tracked person.

The batch summary must make all failures visible. A run is not accepted as complete if any expected clip is missing or malformed.

## 7. STG-NF Control and Export

The current strongest ViTPose/ByteTrack run is the locked STG-NF control. Before extraction, record the exact arguments and checkpoint metadata from the run that produced approximately `64.2%`, including:

- attention type and all attention parameters;
- `seg_len`, training stride, and test stride;
- `model_confidence`, normalization, and confidence handling;
- epochs, batch size, optimizer, learning rate, scheduler, weight decay, and seed;
- STG-NF repository commit;
- checkpoint path and checkpoint arguments;
- score polarity, frame offset, and temporal smoothing.

The BoT-SORT/ReID pose run must be trained and exported with those same STG-NF settings. The only intended experimental change is the tracker-produced pose/track JSON. If the exact control configuration or checkpoint metadata cannot be recovered, the run is labeled a new comparison rather than a clean tracker ablation, and the missing control is resolved before drawing conclusions.

The BoT-SORT model must use separate new train and test pose directories, a fresh output directory, and the same video, ground-truth, and evaluation paths, segment length, attention arguments, checkpoint-selection procedure, and score-export logic as the matched ByteTrack control. The exported pickle must include all 21 Avenue test clips, and its Micro and Macro AUC metadata must be recomputed from the saved per-frame arrays.

The three-epoch trainable-triplet run remains historical context only; it must not be used as the matched control unless it is confirmed as the strongest configuration and its complete metadata is available.

## 8. Evaluation Protocol

The BoT-SORT/ReID run is compared with the existing references under identical evaluation conventions:

- Avenue ground-truth labels, with `1 = anomaly`;
- the same frame indexing and clip set;
- the same STG-NF score polarity;
- the same STG-NF temporal smoothing behavior;
- the same Micro AUC definition over concatenated test frames;
- the same Macro AUC definition over per-video AUCs, skipping one-class videos.

The evaluation report contains:

- Micro AUC;
- Macro AUC and the number of videos contributing to it;
- total evaluated frames and videos;
- per-video AUC rows;
- exact delta against the matched current ViTPose/ByteTrack control;
- secondary delta against the AlphaPose reference;
- the extraction diagnostics summary;
- the complete configuration and artifact paths.

A per-video comparison is required so an aggregate result cannot hide a severe regression in a subset of Avenue clips.

## 9. Artifacts

The run produces:

1. `*_vitpose_tracked_person.json` for every train and test clip;
2. a pose extraction manifest with configuration and per-clip diagnostics;
3. the STG-NF checkpoint and serialized training arguments;
4. `stgnf_scores.pkl` containing per-video frame indices, scores, labels, Micro AUC, and Macro AUC metadata;
5. a CSV of per-video AUC and tracking diagnostics;
6. a JSON summary containing the final decision inputs and exact paths.

All new artifacts live under a new run-specific Drive root. Existing AlphaPose, YOLO26-pose, and ViTPose/ByteTrack outputs remain untouched.

## 10. Validation and Acceptance Criteria

The run is technically valid only if:

- all 16 Avenue training clips and all 21 test clips have validated tracked JSONs;
- every JSON record contains valid COCO-17 data, a scalar detection score, and a unique track/frame key;
- train and test sample shapes match STG-NF's expected `(3, 24, 18)` sample convention after loading;
- the manifest records detector, pose, tracker, library versions, tracker configuration/hash, seed, the complete locked STG-NF control arguments, checkpoint metadata, and output paths;
- the BoT-SORT run uses the same STG-NF arguments and score-processing settings as the matched ByteTrack control;
- the score pickle contains all 21 test clips;
- stored Micro and Macro AUC values match recomputation from the exported arrays;
- frame counts, labels, polarity, smoothing, and indexing match the reference protocol;
- the final report includes per-video AUC and tracking diagnostics.

The numerical result is classified as follows:

### Keep as the preferred pose pipeline

The BoT-SORT/ReID run is above the matched ViTPose/ByteTrack control and the improvement is supported by valid diagnostics. The gain is provisional after one run and must be confirmed by a later repeat before being described as reproducible in a report or paper. The AlphaPose comparison is reported separately as historical context.

### Keep only for fusion evaluation

Pose-only AUC is flat or slightly lower, but the ViTPose stream improves PRISM fusion or shows meaningful complementary per-video wins against the AlphaPose stream. This decision requires a separate PRISM comparison and is not claimed by this experiment alone.

### Stop pose tuning

The run is flat or worse than the reference, has no clear tracking-quality advantage, and does not show a later fusion rationale. No additional detector-confidence sweep is planned. Effort moves to PRISM fusion, appearance-stream improvements, or a different pose representation.

A gain of only a few hundredths of an AUC point is labeled inconclusive until a repeat confirms it. A single run cannot establish statistical significance.

## 11. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| ReID incorrectly merges similar people | Record track lengths/gaps and inspect per-video regressions; keep the exact tracker YAML for rollback. |
| ReID adds excessive extraction time or memory pressure | Use `model=auto`, process one clip sequentially, and record runtime per clip. |
| Ultralytics tracker defaults drift | Pin or record the installed version and store the exact custom YAML content/hash. |
| Training change is confused with tracker change | Recover and lock the complete strongest ByteTrack STG-NF configuration, then reuse it unchanged for BoT-SORT. |
| Missing detections are silently converted into normal frames | Fail the affected clip and expose missing/zero-track counts in the manifest. |
| AUC changes because of evaluation mismatch | Recompute metrics from the exported arrays under the same labels, polarity, frame indexing, and smoothing convention. |

## 12. Next Transition

This design is complete and intentionally stops before implementation. The next step is to create a detailed implementation plan that identifies the notebook edits, tracker configuration handling, diagnostics, validation checks, and the Colab execution order. No implementation should begin until that plan is reviewed.

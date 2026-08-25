# Pose Testing: ByteTrack vs. BoT-SORT/ReID Visualization Design

**Date:** 2026-08-25
**Status:** Approved design; implementation plan pending
**Scope:** `STG-NF/Pose_Testing.ipynb` visualization only

## 1. Goal

Allow a visual decision about whether BoT-SORT with ReID produces better temporal person tracks than ByteTrack for the same selected video or frame folder.

This is a short diagnostic experiment. It does not train STG-NF, calculate AUC, or perform the full Avenue extraction.

## 2. Input and Configuration

The notebook keeps its existing `CLIP_SOURCE` and `SOURCE_MODE` configuration. The user changes that path when they want to inspect another video or image folder; no clip IDs are hardcoded.

Both trackers receive exactly the same input frames and detector configuration:

- YOLO26 detection-only model;
- existing `DET_CONF`;
- existing `IMGSZ`;
- person class only;
- ViTPose++ base;
- existing COCO-17 expert index;
- existing frame ordering and native aspect handling.

The existing ByteTrack behavior remains available as the control.

## 3. Tracker Comparison

Add a pinned custom BoT-SORT configuration alongside ByteTrack:

- `tracker_type: botsort`;
- `with_reid: True`;
- `model: auto`;
- `gmc_method: none` for the static-camera use case;
- all other tracker thresholds remain the installed Ultralytics defaults.

The notebook records the installed Ultralytics version and the exact tracker configuration. If the installed version does not support the requested ReID configuration, the comparison stops with an explicit error rather than silently falling back to another tracker.

Each tracker is reset before processing the selected source. Tracker state is not shared between runs.

## 4. Extraction and Visualization Flow

For each tracker:

```text
same CLIP_SOURCE -> YOLO26 detection/tracking -> ViTPose++ keypoints
                 -> tracked-person records -> overlay rendering
```

The existing ViTPose conversion contract remains unchanged: COCO-17 keypoints are stored as `[x, y, confidence] * 17`, and detector confidence remains a scalar record score.

The notebook renders one synchronized side-by-side MP4:

```text
+---------------------------+-----------------------------+
| ByteTrack                 | BoT-SORT + ReID            |
| tracker label             | tracker label              |
| track IDs                 | track IDs                  |
| ViTPose skeletons         | ViTPose skeletons          |
+---------------------------+-----------------------------+
```

Both panels correspond to the same source frame. Track IDs are drawn in tracker-specific colors, and the overlay includes the source frame number, tracker name, number of active detections, and number of unique tracks observed so far.

The output is written to a new comparison directory derived from `OUT_DIR`. Existing JSONs and rendered videos are never overwritten.

## 5. Diagnostics

For each tracker, write a JSON summary containing:

- source path and source mode;
- decoded frame count;
- frames with detections;
- frames with assigned track IDs;
- total person-frame instances;
- number of unique track IDs;
- per-track first frame, last frame, length, and internal gaps;
- count of short tracks and fragmented tracks;
- mean and selected percentile detector confidence;
- mean and selected percentile keypoint confidence;
- elapsed runtime;
- model, library, and tracker configuration metadata.

The notebook also prints a compact side-by-side summary after rendering. Diagnostics are used to interpret the video, not as a substitute for AUC.

## 6. Validation and Failure Handling

Before rendering, validate that:

- the source exists and contains readable frames;
- both tracker runs process the same number of source frames;
- every pose result has a matching track ID and detection box;
- keypoints have COCO-17 shape and finite coordinates/confidences;
- there are no duplicate `(track_id, frame)` records;
- the output videos open and contain frames.

The notebook raises an explicit error for an unreadable source, unsupported tracker/ReID settings, empty outputs, mismatched frame counts, or malformed pose results. It must not silently produce a pose-free comparison video.

## 7. Decision Rule

After inspecting the synchronized video and summaries:

- **Test BoT-SORT on the full dataset** if it visibly preserves identities through crossings/occlusions and reduces fragmentation without obvious incorrect merges.
- **Keep ByteTrack** if tracks are already continuous or BoT-SORT introduces swaps/merges without a clear continuity benefit.
- **Try a different extraction hypothesis** if both trackers look similar; the next candidate is improved person-crop handling rather than another detector-confidence sweep.

This diagnostic does not claim that one tracker is universally better. It only determines whether BoT-SORT/ReID is promising for the selected source before the expensive full extraction.

## 8. Next Transition

This design intentionally stops before implementation. After the user reviews this spec, the next step is to create an implementation plan for the notebook changes and Colab execution order.

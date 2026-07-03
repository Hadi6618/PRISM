# Fusion

Fusion is a research workspace for video anomaly detection. It contains the code, notebooks, and helpers used to extract video features, train and run MULDE-based scoring, and generate visual reports for anomaly analysis.

## Project Overview

The repository is organized around a simple pipeline:

1. Read a video or dataset sample.
2. Extract features with a pretrained visual backbone.
3. Score the features with the anomaly detection model.
4. Smooth and threshold the scores to find suspicious frame ranges.
5. Save plots, tables, and summaries for review.

The notebooks in the root folder are the main experiment entry points, while the Python files provide reusable model and inference logic.

## Source Tree

```text
Fusion/
├── Avenue_Hiera_L_Feature_Extraction.ipynb
├── fusion.py
├── models.py
├── MULDE_Training_GMM.ipynb
├── mulde_visualization.py
├── Pose Extraction and Testing.ipynb
├── README.md
├── run_custom_anomaly_detection.ipynb
├── run_mulde_on_custom_video.py
├── ShanghaiTech_Ensemble_Fusion.ipynb
└── ShanghaiTech_Hiera_L_Feature_Extraction.ipynb
```

## Root Files

### Core Python Files

#### [run_mulde_on_custom_video.py](run_mulde_on_custom_video.py)

Main command-line runner for custom video inference. It loads a video, extracts frame features, applies MULDE scoring, detects anomaly intervals, and writes the final outputs.

#### [mulde_visualization.py](mulde_visualization.py)

Shared reporting utilities for turning raw scores into readable results. This file handles thresholding, segment detection, result tables, and dashboard creation.

#### [models.py](models.py)

Defines the neural network layers used by MULDE, including the MLP backbone and the wrapper that supports score and log-density behavior.

#### [fusion.py](fusion.py)

Ensemble helper for combining multiple anomaly score streams. It aligns video/frame outputs, normalizes scores, detects label-convention mismatches between STG-NF and MULDE, and searches for the best fusion weights.

### Experiment Notebooks

#### [run_custom_anomaly_detection.ipynb](run_custom_anomaly_detection.ipynb)

Notebook version of the custom-video anomaly detection workflow.

#### [MULDE_Training_GMM.ipynb](MULDE_Training_GMM.ipynb)

Notebook for training the GMM component used by MULDE.

#### [Avenue_Hiera_L_Feature_Extraction.ipynb](Avenue_Hiera_L_Feature_Extraction.ipynb)

Notebook for feature extraction experiments on the Avenue dataset.

#### [ShanghaiTech_Hiera_L_Feature_Extraction.ipynb](ShanghaiTech_Hiera_L_Feature_Extraction.ipynb)

Notebook for feature extraction experiments on the ShanghaiTech dataset.

#### [ShanghaiTech_Ensemble_Fusion.ipynb](ShanghaiTech_Ensemble_Fusion.ipynb)

Notebook for testing the fusion pipeline on ShanghaiTech outputs.

#### [Pose Extraction and Testing.ipynb](Pose%20Extraction%20and%20Testing.ipynb)

Notebook for STG-NF pose extraction, training, and score export. It exposes Dual/Triplet Attention configuration (`ATTENTION_TYPE`, `N_HEADS`, `N_MECATT`, `N_MECATT_INSIDE`) and threads those flags into the STG-NF training and `stgnf_export_scores.py` invocations. The exported `stgnf_scores.pkl` feeds the ensemble fusion step.

### Documentation

#### [README.md](README.md)

Project documentation for the root-level files and overall purpose of the repo.

## What You Get From The Pipeline

The main outputs produced by this project are:

- Frame-level results from MULDE and object-level methods from STG-NF.
- Detected anomaly segments.
- A dashboard image for quick visual review.
- CSV and JSON files with detailed results.

## Results Summary

The project has been evaluated on two datasets: the Avenue Dataset and the ShanghaiTech Campus (STC) Dataset.

| Method | Level | Avenue Micro AUC | STC Micro AUC |
| --- | --- | ---: | ---: |
| STG-NF | Object level | 55.0% | 83.6% |
| MULDE | Frame level | 81.8% | 79.8% |

These numbers show how the two approaches behave on different benchmark datasets. STG-NF performs better on STC, while MULDE is stronger on Avenue, which is why both methods are kept in the repository.

## How To Think About The Repo

Use the notebooks when you want to run or reproduce an experiment. Use the Python files when you want to understand or reuse the underlying scoring and visualization code. The notebooks show the workflow, and the Python modules contain the reusable logic behind it.

## Notes

- Only top-level files are documented here.
- The tree above is the source tree dump for the repository root.
- If you add or rename root files, update this document so it stays accurate.
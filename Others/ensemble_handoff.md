# Video Anomaly Detection Ensemble: STG-NF + MULDE

## Project Overview
This project aims to achieve state-of-the-art Video Anomaly Detection (VAD) on the **ShanghaiTech Campus Dataset** by ensembling two complementary models. 

1. **STG-NF** operates at the **object/pose-level**, tracking human skeletons to detect behavioral anomalies (e.g., fighting, falling). 
2. **MULDE** operates at the **frame-level**, analyzing full-frame features to detect contextual/appearance anomalies (e.g., vehicles on sidewalks).

By ensembling these two approaches, the system can catch anomalies that either model would miss independently.

---

## Model 1: STG-NF (Object-Level Stream)
*   **Paper:** ["Normalizing Flows for Human Pose Anomaly Detection" (ICCV 2023)](https://arxiv.org/abs/2211.10946)
*   **Repository:** [https://github.com/orhir/STG-NF](https://github.com/orhir/STG-NF)
*   **Architecture:** Spatio-Temporal Graph Normalizing Flows. Uses AlphaPose to extract pose sequences and maps them to a latent Gaussian distribution using bijective mapping.
*   **Current Performance:** ~83.9% Micro AUC on ShanghaiTech using custom AlphaPose+YOLOX extractions.
*   **Output:** Negative Log-Likelihood of pose sequences.

## Model 2: MULDE (Frame-Level Stream)
*   **Paper:** ["Multiscale Log-Density Estimation via Denoising Score Matching for Video Anomaly Detection" (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Micorek_MULDE_Multiscale_Log-Density_Estimation_via_Denoising_Score_Matching_for_Video_CVPR_2024_paper.pdf)
*   **Repository:** [https://github.com/jmicorek/mulde](https://github.com/jmicorek/mulde)
*   **Architecture:** Denoising Score Matching (DSM) using a 2-layer MLP (4096 units). Features are extracted using the `Hiera-L` backbone.
*   **Current Performance:** ~79.8% Micro AUC on ShanghaiTech at Epoch 200.
*   **Output:** Negative Log-Likelihood evaluated via a Gaussian Mixture Model (GMM).

---

## Agent Task Description
Your objective is to write the Google Colab Python scripts necessary to extract the raw anomaly scores from both models, normalize them, and fuse them to calculate a maximized final AUC. 

Please follow these specific implementation steps:

### Step 1: Score Extraction (STG-NF)
Modify the `train_eval.py` (or the internal evaluation loop) of the STG-NF repository to export the raw frame-level anomaly scores. 
*   **Requirement:** The exported data must clearly map `video_id -> frame_index -> anomaly_score`.
*   **Format:** Save this dictionary/array as `stgnf_scores.pkl` or `stgnf_scores.npy`.

### Step 2: Score Extraction (MULDE)
Modify the MULDE evaluation script (specifically where it evaluates the GMM and calculates the ROC AUC) to export its raw frame-level scores.
*   **Requirement:** Similar to STG-NF, ensure the scores are perfectly aligned to the exact `video_id` and `frame_index`.
*   **Format:** Save as `mulde_scores.pkl` or `mulde_scores.npy`.

### Step 3: Min-Max Normalization (CRITICAL)
Because STG-NF and MULDE output negative log-likelihoods on completely different mathematical scales, their raw scores cannot be directly added together.
*   **Requirement:** Write a fusion script that loads both score files. For **every individual video**, apply Min-Max scaling to the STG-NF scores to bind them to `[0.0, 1.0]`. 
*   **Requirement:** Do the exact same Min-Max scaling for the MULDE scores for that video.

### Step 4: Weighted Fusion and Grid Search
Compute the final anomaly score for each frame using the formula:
`Final_Score = (Beta_1 * STG_NF_Normalized) + (Beta_2 * MULDE_Normalized)`

*   **Requirement:** Because STG-NF (83.9%) outperforms MULDE (79.8%), `Beta_1` should generally be higher.
*   **Requirement:** Write a grid search loop that iterates through `Beta_1` from `0.0` to `1.0` (where `Beta_2 = 1.0 - Beta_1`). 
*   **Requirement:** For each combination, calculate the overall Micro AUC across the entire ShanghaiTech test set. Output the optimal `Beta_1` and `Beta_2` weights and the resulting maximum AUC.

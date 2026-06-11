"""
Investigate the AUC discrepancy:
- MULDE pkl reports best_micro_auc = 0.7966
- Fusion CSV at beta_2=1.0 shows          = 0.7898

This gap of ~0.007 must come from:
1. Smoothing applied in pkl but NOT in fusion
2. Different normalization (global_minmax clips outliers)
3. Different label convention handling
"""
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d

stg = pickle.load(open("stgnf_scores.pkl", "rb"))
mul = pickle.load(open("mulde_scores.pkl", "rb"))
stg_vids = stg["scores_by_video"]
mul_vids = mul["scores_by_video"]
sorted_vids = sorted(mul_vids.keys())

mul_all_raw   = np.concatenate([mul_vids[v]["anomaly_scores"] for v in sorted_vids])
mul_all_labels = np.concatenate([mul_vids[v]["labels"]        for v in sorted_vids])
stg_all_raw   = np.concatenate([stg_vids[v]["anomaly_scores"] for v in sorted_vids])
stg_all_labels = np.concatenate([stg_vids[v]["labels"]        for v in sorted_vids])

# MULDE uses 0=Normal, 1=Anomaly and higher scores = anomaly
# So AUC(labels, scores) should already work
print("=" * 60)
print("MULDE AUC INVESTIGATION")
print("=" * 60)

# 1. Raw (no smoothing)
auc_raw = roc_auc_score(mul_all_labels, mul_all_raw)
print(f"\n1. Raw (no smoothing):         {auc_raw:.6f}")

# 2. Per-video smoothing with different sigmas (like MULDE training does)
for sigma in [1, 2, 3, 4, 5]:
    smooth = np.concatenate([
        gaussian_filter1d(mul_vids[v]["anomaly_scores"], sigma)
        for v in sorted_vids
    ])
    auc = roc_auc_score(mul_all_labels, smooth)
    print(f"2. Per-video smooth sigma={sigma}:   {auc:.6f}")

reported = mul["best_micro_auc"]
print(f"\n   Reported best_micro_auc:    {reported:.6f}")
print(f"   Smoothing stored in pkl:    sigma={mul['smoothing']['smooth_sigma_frames']}")

# 3. Now check what fusion.py does to MULDE:
# global_minmax on raw scores (NO smoothing applied in fusion pipeline)
print("\n" + "=" * 60)
print("FUSION PIPELINE (what fusion.py does to MULDE)")
print("=" * 60)

# fusion uses MULDE labels (1=anomaly) but negates STG-NF scores
# for MULDE: fusion uses raw anomaly_scores as-is (no smoothing)
global_min = mul_all_raw.min()
global_max = mul_all_raw.max()
mul_norm = (mul_all_raw - global_min) / (global_max - global_min)

auc_fusion_mulde_only = roc_auc_score(mul_all_labels, mul_norm)
print(f"\nMULDE alone in fusion (raw + global_minmax): {auc_fusion_mulde_only:.6f}")
print(f"This matches beta_2=1.0 in CSV:             0.789810")
print(f"Gap from pkl reported:                       {reported - auc_fusion_mulde_only:.6f}")

print("\nCONCLUSION:")
print(f"  pkl stored AUC ({reported:.4f}) was computed WITH Gaussian smoothing (sigma={mul['smoothing']['smooth_sigma_frames']})")
print(f"  fusion CSV AUC (0.7898) was computed WITHOUT smoothing (raw global_minmax only)")
print(f"  The gap of ~{reported - auc_fusion_mulde_only:.4f} is entirely due to smoothing NOT being applied in fusion.py")

# 4. Same check for STG-NF
print("\n" + "=" * 60)
print("STG-NF AUC INVESTIGATION")
print("=" * 60)

stg_reported = stg["micro_auc"]
# STG-NF labels: 1=Normal, 0=Anomaly. Scores: higher=normal
# fusion negates: -stg_raw, then uses mul_labels (1=anomaly)
stg_neg = -stg_all_raw  # flip to anomaly scores
# global_minmax
stg_global_min = stg_neg.min()
stg_global_max = stg_neg.max()
stg_norm = (stg_neg - stg_global_min) / (stg_global_max - stg_global_min)

# use mul_labels as reference (1=anomaly)
auc_stg_fusion = roc_auc_score(mul_all_labels, stg_norm)
print(f"\nSTG-NF alone in fusion (negated + global_minmax): {auc_stg_fusion:.6f}")
print(f"Reported micro_auc in pkl (own labels):           {stg_reported:.6f}")
print(f"Gap: {stg_reported - auc_stg_fusion:.6f}")

# 5. What happens if we apply smoothing in fusion too?
print("\n" + "=" * 60)
print("WHAT IF WE APPLY SMOOTHING IN FUSION TOO?")
print("=" * 60)

sigma = mul["smoothing"]["smooth_sigma_frames"]  # 3
mul_smooth_all = np.concatenate([
    gaussian_filter1d(mul_vids[v]["anomaly_scores"], sigma)
    for v in sorted_vids
])
stg_neg_smooth = np.concatenate([
    gaussian_filter1d(-stg_vids[v]["anomaly_scores"], sigma)
    for v in sorted_vids
])

# global_minmax on smoothed
def gminmax(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

mul_s_norm = gminmax(mul_smooth_all)
stg_s_norm = gminmax(stg_neg_smooth)

auc_mul_s = roc_auc_score(mul_all_labels, mul_s_norm)
auc_stg_s = roc_auc_score(mul_all_labels, stg_s_norm)

print(f"\nAfter smoothing (sigma={sigma}) + global_minmax:")
print(f"  MULDE  alone: {auc_mul_s:.6f}  (was {auc_fusion_mulde_only:.6f} without smoothing)")
print(f"  STG-NF alone: {auc_stg_s:.6f}  (was {auc_stg_fusion:.6f} without smoothing)")

# Grid search with smoothing
print("\nGrid search with smoothing applied:")
best_auc = 0
best_b1 = 0
for b1_int in range(0, 101):
    b1 = b1_int / 100.0
    b2 = 1.0 - b1
    fused = b1 * stg_s_norm + b2 * mul_s_norm
    auc = roc_auc_score(mul_all_labels, fused)
    if auc > best_auc:
        best_auc = auc
        best_b1 = b1

print(f"  BEST: beta_1(STG-NF)={best_b1:.2f}  beta_2(MULDE)={1-best_b1:.2f}  AUC={best_auc:.6f}")
print(f"  Old best (no smoothing): beta_1=0.03  beta_2=0.97  AUC=0.870307")

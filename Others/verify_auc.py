"""Verify AUC computation logic across STG-NF, MULDE, and the fusion pipeline."""
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score

stg = pickle.load(open("stgnf_scores.pkl", "rb"))
mul = pickle.load(open("mulde_scores.pkl", "rb"))
stg_vids = stg["scores_by_video"]
mul_vids = mul["scores_by_video"]

# Concatenate ALL scores and labels for each model separately
sorted_vids = sorted(stg_vids.keys())
stg_all_scores = np.concatenate([np.array(stg_vids[v]["anomaly_scores"]) for v in sorted_vids])
stg_all_labels = np.concatenate([np.array(stg_vids[v]["labels"]) for v in sorted_vids])
mul_all_scores = np.concatenate([np.array(mul_vids[v]["anomaly_scores"]) for v in sorted_vids])
mul_all_labels = np.concatenate([np.array(mul_vids[v]["labels"]) for v in sorted_vids])

print("=" * 60)
print("STANDALONE AUC VERIFICATION")
print("=" * 60)

print("\n--- STG-NF (using its OWN labels) ---")
print("  Label distribution: 0={}, 1={}".format(np.sum(stg_all_labels == 0), np.sum(stg_all_labels == 1)))
auc_stg_pos = roc_auc_score(stg_all_labels, stg_all_scores)
auc_stg_neg = roc_auc_score(stg_all_labels, -stg_all_scores)
print("  AUC(labels, +scores): {:.6f}".format(auc_stg_pos))
print("  AUC(labels, -scores): {:.6f}".format(auc_stg_neg))
print("  Reported micro_auc:   {:.6f}".format(stg["micro_auc"]))

print("\n--- MULDE (using its OWN labels) ---")
print("  Label distribution: 0={}, 1={}".format(np.sum(mul_all_labels == 0), np.sum(mul_all_labels == 1)))
auc_mul_pos = roc_auc_score(mul_all_labels, mul_all_scores)
auc_mul_neg = roc_auc_score(mul_all_labels, -mul_all_scores)
print("  AUC(labels, +scores): {:.6f}".format(auc_mul_pos))
print("  AUC(labels, -scores): {:.6f}".format(auc_mul_neg))
print("  Reported best_micro_auc: {:.6f}".format(mul["best_micro_auc"]))

print("\n--- LABEL INVERSION CHECK ---")
print("  Are labels perfectly inverted? {}".format(np.array_equal(stg_all_labels, 1 - mul_all_labels)))

# Use MULDE convention (0=normal, 1=anomaly) as reference
normal_mask = mul_all_labels == 0
anomaly_mask = mul_all_labels == 1

print("\n--- SCORE DISTRIBUTIONS (MULDE convention: 0=normal, 1=anomaly) ---")
print("  MULDE  normal  mean: {:.4f}".format(mul_all_scores[normal_mask].mean()))
print("  MULDE  anomaly mean: {:.4f}".format(mul_all_scores[anomaly_mask].mean()))
print("  -> Anomaly frames have HIGHER MULDE scores? {}".format(
    mul_all_scores[anomaly_mask].mean() > mul_all_scores[normal_mask].mean()))

print("  STG-NF normal  mean: {:.4f}".format(stg_all_scores[normal_mask].mean()))
print("  STG-NF anomaly mean: {:.4f}".format(stg_all_scores[anomaly_mask].mean()))
print("  -> Anomaly frames have HIGHER STG-NF scores? {}".format(
    stg_all_scores[anomaly_mask].mean() > stg_all_scores[normal_mask].mean()))

print("\n--- CROSS-CHECK: STG-NF scores with MULDE labels ---")
auc_cross_pos = roc_auc_score(mul_all_labels, stg_all_scores)
auc_cross_neg = roc_auc_score(mul_all_labels, -stg_all_scores)
print("  AUC(mulde_labels, +stgnf_scores): {:.6f}".format(auc_cross_pos))
print("  AUC(mulde_labels, -stgnf_scores): {:.6f}".format(auc_cross_neg))

print("\n" + "=" * 60)
print("FUSION PIPELINE VERIFICATION")
print("=" * 60)

# Now simulate what fusion.py actually does
# Step 1: fusion uses MULDE labels (from _resolve_labels)
fusion_labels = mul_all_labels.copy()

# Step 2: fusion detects stgnf_score_mode='normality' and negates STG-NF scores
fusion_stgnf = -stg_all_scores  # _apply_stgnf_polarity with mode='normality'
fusion_mulde = mul_all_scores.copy()

# Step 3: global_minmax normalization
def safe_minmax(v):
    vmin, vmax = v.min(), v.max()
    if vmax <= vmin:
        return np.zeros_like(v)
    return (v - vmin) / (vmax - vmin)

fusion_stgnf_norm = safe_minmax(fusion_stgnf)
fusion_mulde_norm = safe_minmax(fusion_mulde)

print("\nAfter polarity flip + global_minmax:")
print("  STG-NF normalized: mean={:.6f} std={:.6f} min={:.4f} max={:.4f}".format(
    fusion_stgnf_norm.mean(), fusion_stgnf_norm.std(),
    fusion_stgnf_norm.min(), fusion_stgnf_norm.max()))
print("  MULDE  normalized: mean={:.6f} std={:.6f} min={:.4f} max={:.4f}".format(
    fusion_mulde_norm.mean(), fusion_mulde_norm.std(),
    fusion_mulde_norm.min(), fusion_mulde_norm.max()))

# Step 4: individual AUC after normalization
print("\n  STG-NF alone AUC (after flip+norm): {:.6f}".format(
    roc_auc_score(fusion_labels, fusion_stgnf_norm)))
print("  MULDE  alone AUC (after norm):      {:.6f}".format(
    roc_auc_score(fusion_labels, fusion_mulde_norm)))

# Step 5: grid search fusion
print("\n--- GRID SEARCH ---")
best_auc = 0
best_b1 = 0
for b1_int in range(0, 101):
    b1 = b1_int / 100.0
    b2 = 1.0 - b1
    fused = b1 * fusion_stgnf_norm + b2 * fusion_mulde_norm
    auc = roc_auc_score(fusion_labels, fused)
    if auc > best_auc:
        best_auc = auc
        best_b1 = b1
    if b1_int <= 10 or b1_int % 10 == 0:
        print("  beta_1={:.2f} beta_2={:.2f} -> AUC={:.6f}".format(b1, b2, auc))

print("\n  BEST: beta_1={:.2f} beta_2={:.2f} -> AUC={:.6f}".format(
    best_b1, 1.0 - best_b1, best_auc))

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("STG-NF label convention: 1=NORMAL, 0=ANOMALY")  
print("MULDE  label convention: 0=NORMAL, 1=ANOMALY")
print("Labels are INVERTED between the two PKL files.")
print()
print("The fusion code:")
print("  1. Uses MULDE labels as ground truth (1=anomaly)")
print("  2. Negates STG-NF scores (mode='normality')")
print("  3. Applies global_minmax to both")
print("  4. Runs grid search")

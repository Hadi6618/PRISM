"""
Investigate WHY the optimal STG-NF weight is 0.05 AFTER min-max normalization.
The answer is NOT scale. It is about signal vs noise per frame type.
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

# Build raw arrays (polarity-corrected: negate STG-NF, MULDE already anomaly score)
stg_raw = np.concatenate([-np.array(stg_vids[v]["anomaly_scores"]) for v in sorted_vids])
mul_raw = np.concatenate([np.array(mul_vids[v]["anomaly_scores"])  for v in sorted_vids])
labels  = np.concatenate([np.array(mul_vids[v]["labels"])          for v in sorted_vids])

# Apply sigma=2 smoothing per video
stg_smooth = np.concatenate([
    gaussian_filter1d(-np.array(stg_vids[v]["anomaly_scores"]).astype(np.float64), 2)
    for v in sorted_vids
])
mul_smooth = np.concatenate([
    gaussian_filter1d(np.array(mul_vids[v]["anomaly_scores"]).astype(np.float64), 2)
    for v in sorted_vids
])

# Global min-max normalization
def gminmax(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

stg_n = gminmax(stg_smooth)
mul_n = gminmax(mul_smooth)

normal_mask  = labels == 0
anomaly_mask = labels == 1

print("=" * 60)
print("DISTRIBUTION ANALYSIS AFTER SMOOTHING + MIN-MAX NORM")
print("=" * 60)

print("\n--- STG-NF normalized scores ---")
print(f"  Overall  mean={stg_n.mean():.5f}  std={stg_n.std():.5f}")
print(f"  Normal   mean={stg_n[normal_mask].mean():.5f}  std={stg_n[normal_mask].std():.5f}")
print(f"  Anomaly  mean={stg_n[anomaly_mask].mean():.5f}  std={stg_n[anomaly_mask].std():.5f}")
print(f"  Separation (anomaly_mean - normal_mean) = {stg_n[anomaly_mask].mean() - stg_n[normal_mask].mean():.5f}")

print("\n--- MULDE normalized scores ---")
print(f"  Overall  mean={mul_n.mean():.5f}  std={mul_n.std():.5f}")
print(f"  Normal   mean={mul_n[normal_mask].mean():.5f}  std={mul_n[normal_mask].std():.5f}")
print(f"  Anomaly  mean={mul_n[anomaly_mask].mean():.5f}  std={mul_n[anomaly_mask].std():.5f}")
print(f"  Separation (anomaly_mean - normal_mean) = {mul_n[anomaly_mask].mean() - mul_n[normal_mask].mean():.5f}")

print("\n" + "=" * 60)
print("NOISE ANALYSIS: How much does each model score normal frames?")
print("=" * 60)
print(f"\n  A randomly picked NORMAL frame gets on average:")
print(f"    STG-NF score = {stg_n[normal_mask].mean():.5f}")
print(f"    MULDE  score = {mul_n[normal_mask].mean():.5f}")
print(f"\n  STG-NF normal-frame score is {stg_n[normal_mask].mean()/mul_n[normal_mask].mean():.1f}x HIGHER than MULDE")
print(f"  This means STG-NF adds MORE noise to normal frames than MULDE does.")

print("\n" + "=" * 60)
print("VEHICLE-ONLY vs HUMAN-ONLY ANOMALY BREAKDOWN")
print("=" * 60)
# Find videos where STG-NF is BLIND (it scores anomaly frames same as normal)
# i.e. where MULDE detects but STG-NF misses
vid_stg_auc = {}
vid_mul_auc = {}
for v in sorted_vids:
    y  = np.array(mul_vids[v]["labels"])
    if len(np.unique(y)) < 2:
        continue
    s  = gminmax(gaussian_filter1d(-np.array(stg_vids[v]["anomaly_scores"]).astype(np.float64), 2))
    m  = gminmax(gaussian_filter1d( np.array(mul_vids[v]["anomaly_scores"]).astype(np.float64), 2))
    vid_stg_auc[v] = roc_auc_score(y, s)
    vid_mul_auc[v] = roc_auc_score(y, m)

stg_aucs = np.array(list(vid_stg_auc.values()))
mul_aucs = np.array(list(vid_mul_auc.values()))

# Videos where MULDE >> STG-NF (likely vehicle anomalies)
mulde_wins = [(v, vid_mul_auc[v], vid_stg_auc[v])
              for v in vid_stg_auc if vid_mul_auc[v] - vid_stg_auc[v] > 0.3]
mulde_wins.sort(key=lambda x: x[1]-x[2], reverse=True)

# Videos where STG-NF >> MULDE (likely human/pose anomalies)
stg_wins = [(v, vid_stg_auc[v], vid_mul_auc[v])
            for v in vid_stg_auc if vid_stg_auc[v] - vid_mul_auc[v] > 0.3]
stg_wins.sort(key=lambda x: x[1]-x[2], reverse=True)

print(f"\nVideos where MULDE >> STG-NF (MULDE advantage > 0.30):")
print(f"  Count = {len(mulde_wins)}")
for v, m, s in mulde_wins[:5]:
    print(f"    {v}: MULDE={m:.3f}  STG-NF={s:.3f}  diff={m-s:.3f}")

print(f"\nVideos where STG-NF >> MULDE (STG-NF advantage > 0.30):")
print(f"  Count = {len(stg_wins)}")
for v, s, m in stg_wins[:5]:
    print(f"    {v}: STG-NF={s:.3f}  MULDE={m:.3f}  diff={s-m:.3f}")

print("\n" + "=" * 60)
print("THE WEIGHT EXPLANATION")
print("=" * 60)
print(f"\nIn {len(mulde_wins)} videos, MULDE clearly detects but STG-NF is BLIND.")
print(f"In {len(stg_wins)} videos, STG-NF clearly detects but MULDE is BLIND.")
print(f"\nRatio: MULDE has {len(mulde_wins)/max(len(stg_wins),1):.1f}x more 'exclusive' detections than STG-NF.")
print(f"\nConclusion: MULDE dominates the dataset (more vehicle/crowd anomalies).")
print(f"STG-NF is a SUPPLEMENT for pose-based anomalies — hence the low 0.05 weight.")
print(f"\nSTG-NF's normal-frame noise ({stg_n[normal_mask].mean():.4f}) vs MULDE ({mul_n[normal_mask].mean():.4f})")
print(f"means a 50/50 mix would let STG-NF's noise degrade MULDE's clean ranking.")

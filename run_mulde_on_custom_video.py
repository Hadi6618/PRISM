#!/usr/bin/env python
"""
Run MULDE Anomaly Detection on a Custom MP4 Video
-------------------------------------------------
This script runs the entire MULDE inference pipeline on a single custom .mp4 video file:
1. Loads the pretrained Hiera-L model from PyTorch Hub (head set to Identity).
2. Decodes and preprocesses the video frames.
3. Extracts spatiotemporal features (1152-dim) in batches.
4. Standardizes features using training stats (mean/std).
5. Computes the 16-dimensional multiscale log-density signature using the trained MLP.
6. Scores the signatures using the GMM to compute raw log-likelihood scores.
7. Applies temporal Gaussian smoothing (sigma=15.0).
8. Generates a professional line chart of the log-likelihood scores (with optional shaded anomaly zones)
   and saves a CSV of the frame-level scores.

Usage:
  python run_mulde_on_custom_video.py \
    --video path/to/video.mp4 \
    --checkpoint path/to/mulde_final.pt \
    --stats path/to/train_feature_stats.npz \
    --gmm path/to/gmm_components_5.joblib \
    --output_dir output_results \
    --shading "50-200,340-440"
"""

import os
import sys
import gc
import argparse
import json
import time
import math
import numpy as np
import pandas as pd
import torch
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
import subprocess
from decord import VideoReader, cpu
from scipy.ndimage import gaussian_filter1d

# Inject official MULDE repo path if available to import architectures
OFFICIAL_REPO_URL = "https://github.com/jakubmicorek/MULDE-Multiscale-Log-Density-Estimation-via-Denoising-Score-Matching-for-Video-Anomaly-Detection.git"
OFFICIAL_REPO_DIR = Path("/content/MULDE_official")
# Optional reproducibility pin. Leave as None to use the repository default branch.
OFFICIAL_REPO_COMMIT = None

if not OFFICIAL_REPO_DIR.exists():
    subprocess.run(["git", "clone", OFFICIAL_REPO_URL, str(OFFICIAL_REPO_DIR)], check=True)
else:
    print(f"Repository already exists: {OFFICIAL_REPO_DIR}")

if OFFICIAL_REPO_COMMIT is not None:
    subprocess.run(["git", "-C", str(OFFICIAL_REPO_DIR), "checkout", OFFICIAL_REPO_COMMIT], check=True)

repo_sha = subprocess.check_output(
    ["git", "-C", str(OFFICIAL_REPO_DIR), "rev-parse", "HEAD"],
    text=True,
).strip()

sys.path.insert(0, str(OFFICIAL_REPO_DIR))
from models import MLPs, ScoreOrLogDensityNetwork

print(f"Imported official MULDE models from: {OFFICIAL_REPO_DIR}")
print(f"Official repo commit: {repo_sha}")




def load_hiera_extractor(device):
    print("Loading Hiera-L model (hiera_large_16x224, checkpoint=mae_k400_ft_k400) from PyTorch Hub...")
    model = torch.hub.load(
        "facebookresearch/hiera",
        model="hiera_large_16x224",
        pretrained=True,
        checkpoint="mae_k400_ft_k400"
    )
    model.head = torch.nn.Identity()  # Replace classifier head with Identity
    model = model.to(device)
    model.eval()
    return model


def preprocess_all_frames(vr, num_frames, target_size=(224, 224), chunk_size=128):
    """Decode and normalize all video frames exactly once."""
    mean = np.array([0.45, 0.45, 0.45], dtype=np.float32).reshape(1, 3, 1, 1)
    std  = np.array([0.225, 0.225, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    all_frames = np.empty((num_frames, 3, target_size[0], target_size[1]), dtype=np.float32)

    for start in range(0, num_frames, chunk_size):
        end = min(start + chunk_size, num_frames)
        indices = list(range(start, end))
        frames_np = vr.get_batch(indices).asnumpy()  # [chunk, H, W, C] RGB uint8

        for j, img in enumerate(frames_np):
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            img_float = img_resized.astype(np.float32) / 255.0
            all_frames[start + j] = img_float.transpose(2, 0, 1)  # HWC -> CHW

    all_frames = (all_frames - mean) / std
    return all_frames


def generate_clip_indices(i, num_frames):
    """Sample 16 frames with stride 4, centered around target frame i."""
    indices = []
    for k in range(16):
        idx = i - 30 + 4 * k
        idx = max(0, min(idx, num_frames - 1))
        indices.append(idx)
    return np.array(indices, dtype=np.int64)


def extract_hiera_features(video_path, model, device, batch_size=8):
    """Run Hiera-L batched inference to extract 1152-dim features."""
    print(f"Decoding video: {video_path}...")
    vr = VideoReader(video_path, ctx=cpu(0))
    num_frames = len(vr)
    fps = vr.get_avg_fps()

    print(f"Pre-caching {num_frames} frames...")
    cached_frames = preprocess_all_frames(vr, num_frames)
    del vr
    gc.collect()

    print("Running Hiera-L batched feature extraction...")
    frame_indices = np.arange(num_frames, dtype=np.int64)
    clip_indices = np.zeros((num_frames, 16), dtype=np.int64)
    for i in range(num_frames):
        clip_indices[i] = generate_clip_indices(i, num_frames)

    features_list = []
    current_batch_size = batch_size
    success = False

    while not success and current_batch_size >= 1:
        try:
            features_list = []
            for batch_start in range(0, num_frames, current_batch_size):
                batch_end = min(batch_start + current_batch_size, num_frames)
                batch_clips = []
                for i in range(batch_start, batch_end):
                    clip_frames = cached_frames[clip_indices[i]]  # [16, 3, 224, 224]
                    clip_tensor = torch.from_numpy(clip_frames.copy()).permute(1, 0, 2, 3)
                    batch_clips.append(clip_tensor)

                stacked = torch.stack(batch_clips, dim=0).to(device)
                with torch.no_grad():
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        feats = model(stacked)
                    features_list.append(feats.float().cpu().numpy())
                del stacked
            features = np.concatenate(features_list, axis=0)
            success = True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[WARNING] OOM with batch={current_batch_size}. Halving and retrying...")
                current_batch_size //= 2
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if current_batch_size < 1:
                    raise e
            else:
                raise e

    del cached_frames
    gc.collect()
    return features, num_frames, fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input .mp4 video file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained MULDE neural network (.pt)")
    parser.add_argument("--stats", type=str, required=True, help="Path to training train_feature_stats.npz")
    parser.add_argument("--gmm", type=str, required=True, help="Path to trained GMM model (.joblib)")
    parser.add_argument("--output_dir", type=str, default="output_results", help="Directory to save output files")
    parser.add_argument("--smooth_sigma", type=float, default=15.0, help="Sigma for temporal Gaussian smoothing")
    parser.add_argument("--shading", type=str, default=None, help="Shaded frame ranges for plotting (e.g. '50-200,340-440')")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Hiera Feature Extraction
    hiera_model = load_hiera_extractor(device)
    features, num_frames, fps = extract_hiera_features(args.video, hiera_model, device)
    print(f"Features extracted: {features.shape} at {fps:.2f} FPS")

    # 2. Standardization
    print(f"Loading feature standardization stats from: {args.stats}")
    stats = np.load(args.stats)
    train_mean = stats["mean"].astype(np.float32)
    train_std = stats["std"].astype(np.float32)
    train_std = np.where(train_std < 1e-8, 1.0, train_std)  # Avoid div-by-zero
    features_std = (features - train_mean) / train_std

    # 3. Load MULDE Network & Compute Signatures
    print(f"Loading MULDE network from checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Recreate the MLP/Log-Density model matching training configuration
    mulde_net = ScoreOrLogDensityNetwork(
        MLPs(
            input_dim=1152 + 1,
            output_dim=1,
            units=[4096, 4096]
        ),
        score_network=False
    ).to(device)
    
    # Load state dict
    if "model_state_dict" in checkpoint:
        mulde_net.load_state_dict(checkpoint["model_state_dict"])
    else:
        mulde_net.load_state_dict(checkpoint)
    mulde_net.eval()

    # Generate multiscale log-density signatures (L=16)
    sigma_levels = np.linspace(1e-3, 1.0, 16, dtype=np.float32)
    signatures = np.empty((num_frames, 16), dtype=np.float32)

    print("Computing multiscale log-density signatures...")
    with torch.no_grad():
        # Process in batches to limit memory
        batch_size = 512
        for start in range(0, num_frames, batch_size):
            end = min(start + batch_size, num_frames)
            x_batch = torch.from_numpy(features_std[start:end]).to(device)
            cols = []
            for sigma_val in sigma_levels:
                sigma_col = torch.full((x_batch.shape[0], 1), float(sigma_val), device=device)
                log_density = mulde_net(torch.cat([x_batch, sigma_col], dim=1)).reshape(-1)
                cols.append(log_density.cpu().numpy())
            signatures[start:end] = np.stack(cols, axis=1)

    # 4. Fit/Score GMM
    print(f"Loading GMM model from: {args.gmm}")
    gmm = joblib.load(args.gmm)
    
    # Compute raw log-likelihood under GMM (scores matching the user chart shape)
    raw_log_likelihood = gmm.score_samples(signatures)
    
    # Apply 1D Gaussian smoothing
    smoothed_log_likelihood = gaussian_filter1d(raw_log_likelihood, sigma=args.smooth_sigma)

    # 5. Export scores to CSV
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    csv_path = os.path.join(args.output_dir, f"{video_name}_mulde_scores.csv")
    df_out = pd.DataFrame({
        "frame_index": np.arange(num_frames),
        "raw_log_likelihood": raw_log_likelihood,
        "smoothed_log_likelihood": smoothed_log_likelihood
    })
    df_out.to_csv(csv_path, index=False)
    print(f"✓ Saved frame scores to CSV: {csv_path}")

    # 6. Plot log-likelihood chart (matching user reference)
    plt.figure(figsize=(12, 4))
    
    # Plot raw scores (light purple/blue)
    plt.plot(raw_log_likelihood, color="#b3b3f2", alpha=0.7, linewidth=1.0, label="Raw Scores")
    # Plot smoothed scores (green)
    plt.plot(smoothed_log_likelihood, color="#008000", linewidth=1.8, label="Smoothed Scores")

    # Handle shading/anomaly boundaries
    if args.shading:
        try:
            ranges = args.shading.split(",")
            for r in ranges:
                start_r, end_r = map(int, r.split("-"))
                plt.axvspan(start_r, end_r, color="#ffb3b3", alpha=0.6, label="Anomaly Window" if "Anomaly Window" not in plt.gca().get_legend_handles_labels()[1] else "")
        except Exception as e:
            print(f"[WARNING] Failed to parse shading ranges: {e}. Format should be: '50-200,340-440'")

    plt.xlabel("Frame", fontsize=11, labelpad=8)
    plt.ylabel("Log-Likelihood Score", fontsize=11, labelpad=8)
    plt.title(f"MULDE Anomaly Score Profile — {video_name}", fontsize=12, fontweight="bold", pad=12)
    plt.xlim(0, num_frames)
    
    # Invert y-axis limits dynamically based on values, but keep standard log-density limits
    margin = (raw_log_likelihood.max() - raw_log_likelihood.min()) * 0.05
    plt.ylim(raw_log_likelihood.min() - margin, raw_log_likelihood.max() + margin)

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()

    chart_path = os.path.join(args.output_dir, f"{video_name}_anomaly_profile.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"✓ Saved anomaly visualization chart: {chart_path}")
    print("Inference completed successfully!")


if __name__ == "__main__":
    main()

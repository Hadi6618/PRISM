from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import zipfile
import shutil
from pathlib import Path

# Add the script parent directory to system path and walk up to search for paper_architecture
script_parent = Path(__file__).parent.absolute()
sys.path.append(str(script_parent))
sys.path.append(str(script_parent.resolve()))
sys.path.append("/content/drive/MyDrive/Project/Mohammed models")
sys.path.append(os.getcwd())

# Traverse up to find parent of architecture
curr = script_parent
for _ in range(4):
    if (curr / "architecture").exists():
        sys.path.append(str(curr))
        break
    curr = curr.parent

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# Import datasets helper, model architectures, and post-processing tools
try:
    from CAE.Inference.datasets import adjust_path, load_gray_tensor, load_flow_tensor
    from CAE.Inference.models import AppearanceCAE, MotionCAE, BinaryClassifier
except ImportError as err:
    print("\n--- DIAGNOSTICS FOR IMPORT ERROR ---")
    print(f"Python Executable: {sys.executable}")
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Script File: {__file__}")
    print(f"Resolved Script Parent: {script_parent}")
    print(f"Parent contents: {os.listdir(script_parent) if script_parent.exists() else 'Does not exist'}")
    print(f"sys.path: {sys.path}")
    print("------------------------------------\n")
    raise err


try:
    import scipy.ndimage
except ImportError:
    print("scipy is required for post-processing filters. Installing scipy...")
    # SciPy is standard in Colab, but fallback import is safe

from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference and evaluate the model on ShanghaiTech testing data."
    )
    parser.add_argument(
        "--test-inputs-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/Experiments/custom_video_cache/CAE/1"),
        help="Path to the synced test zip files in Google Drive."
    )
    parser.add_argument(
        "--tracking-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/Project/Mohammed models/tracking_outputs_test"),
        help="Folder containing *_trajectories.txt files for the test set."
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/Experiments/CAE_weights"),
        help="Folder containing the trained CAE and classifier weights (.pt checkpoints)."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/Experiments/1/CAE"),
        help="Output directory to save frame-level anomaly scores and metrics."
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("/content/test_inputs/labels_output"),
        help="ShanghaiTech test set frame masks for AUC calculation."
    )
    parser.add_argument(
        "--frames-root-dir",
        type=Path,
        default=Path("/content/avenue/testing/frames"),
        help="Directory containing original test frames (used to query width/height)."
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--spatial-filter-size",
        type=int,
        default=15,
        help="Kernel size for spatial smoothing of the anomaly map."
    )
    parser.add_argument(
        "--temporal-filter-size",
        type=int,
        default=3,
        help="Temporal kernel size for the 3D mean filter."
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=2.0,
        help="Standard deviation for 1D Gaussian temporal smoothing of frame scores."
    )
    return parser.parse_args()


def auto_extract_test_data(drive_test_inputs_dir: Path, local_test_inputs_dir: Path) -> Path:
    """Copy/extract test dataset from Drive to local SSD on Google Colab.

    Handles three cases:
      a) Zip files exist inside drive_test_inputs_dir  -> extract them.
      b) Subfolders (appearance/, flow_forward/, etc.) exist inside drive_test_inputs_dir -> copy them.
      c) The data lives in a sibling folder on Drive called 'test_inputs' (old name) -> copy from there.
    """
    if not Path("/content").exists():
        print("Local environment detected. Skipping zip extraction...")
        return drive_test_inputs_dir

    print("Google Colab environment detected. Checking test dataset...")
    local_test_inputs_dir.mkdir(parents=True, exist_ok=True)

    # --- Locate the actual Drive source folder ---
    # Priority: drive_test_inputs_dir itself, then sibling 'test_inputs' folder
    drive_source = None
    candidates = [
        drive_test_inputs_dir,
        #drive_test_inputs_dir.parent / "test_inputs",
        drive_test_inputs_dir.parent / "test_input_data",
    ]
    for c in candidates:
        if c.exists():
            drive_source = c
            print(f"Found Drive source folder: {drive_source}")
            break

    if drive_source is None:
        print("WARNING: Could not find test data folder on Drive. Will attempt inference from manifest paths directly.")
        return local_test_inputs_dir

    # --- Copy/extract manifest CSV ---
    csv_files = list(drive_source.glob("*.csv"))
    if csv_files:
        local_manifest = Path("/content/test_inputs/test_object_inputs_manifest.csv")
        if not local_manifest.exists():
            print(f"Copying manifest {csv_files[0].name} to local storage...")
            shutil.copy(csv_files[0], local_manifest)
            print("Manifest CSV copied.")

    # --- Case (a): Extract zip files ---
    zip_files = list(drive_source.glob("*.zip"))
    for zip_path in zip_files:
        zip_name = zip_path.name
        marker = local_test_inputs_dir / f".{zip_name}.extracted"
        if not marker.exists():
            print(f"Extracting {zip_name} to local storage...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(local_test_inputs_dir)
            marker.touch()
            print(f"Extracted {zip_name}.")
        else:
            print(f"{zip_name} already extracted.")

    # --- Case (b/c): Copy unzipped subfolders (appearance, flow_forward, flow_backward) ---
    folders_to_copy = ["appearance", "flow_forward", "flow_backward"]
    for folder_name in folders_to_copy:
        src_folder = drive_source / folder_name
        dst_folder = local_test_inputs_dir / folder_name
        if src_folder.exists() and not dst_folder.exists():
            print(f"Copying {folder_name}/ from Drive to local SSD...")
            shutil.copytree(str(src_folder), str(dst_folder))
            print(f"Copied {folder_name}/.")
        elif dst_folder.exists():
            print(f"{folder_name}/ already on local SSD.")

    # --- Check for manifest inside extracted content ---
    local_manifest = Path("/content/test_inputs/test_object_inputs_manifest.csv")
    if not local_manifest.exists():
        extracted_csvs = list(local_test_inputs_dir.glob("**/*.csv"))
        if extracted_csvs:
            print(f"Found manifest inside extracted content: {extracted_csvs[0].name}")
            shutil.copy(extracted_csvs[0], local_manifest)

    print("Test data preparation completed successfully.")
    return local_test_inputs_dir


def generate_fallback_manifest(local_test_inputs_dir: Path, tracking_dir: Path) -> Path:
    """Fallback generator to construct manifest CSV by scanning crop filenames and matching bboxes."""
    print("Manifest CSV not found. Generating fallback manifest from filenames and tracking files...")
    
    appearance_dir = local_test_inputs_dir / "appearance"
    if not appearance_dir.exists():
        # Maybe it's nested or zip extracted with parent directories
        subdirs = [d for d in local_test_inputs_dir.glob("**/appearance") if d.is_dir()]
        if subdirs:
            appearance_dir = subdirs[0]
            print(f"Redirecting appearance directory to nested folder: {appearance_dir}")
        else:
            raise FileNotFoundError(f"Appearance directory not found at {appearance_dir}. Cannot generate manifest.")

    # Look for files like clip_id__frame_id__object_id.png
    pattern = re.compile(r"(.+)__frame_(\d+)__object_(\d+)\.(png|jpg|jpeg|bmp)", re.IGNORECASE)
    crop_files = sorted(appearance_dir.rglob("*.*"))
    rows = []
    
    tracking_cache = {}
    print(f"Scanning {len(crop_files)} crop files in {appearance_dir}...")
    for crop_path in tqdm(crop_files, desc="Parsing crop filenames"):
        match = pattern.match(crop_path.name)
        if not match:
            continue
        clip_id = match.group(1)
        frame_id = int(match.group(2))
        object_id = int(match.group(3))
        
        # Load trajectories for this clip
        if clip_id not in tracking_cache:
            txt_path = tracking_dir / f"{clip_id}_trajectories.txt"
            if not txt_path.exists():
                # Try alternatives
                alt_id = clip_id.replace("clip_", "")
                txt_path = tracking_dir / f"{alt_id}_trajectories.txt"
                
            if txt_path.exists():
                try:
                    df = pd.read_csv(txt_path, header=None)
                    df.columns = ["frame_id", "object_id", "x_center", "y_center", "width", "height", "label"][:df.shape[1]]
                    df["frame_id"] = df["frame_id"].astype(int)
                    df["object_id"] = df["object_id"].astype(int)
                    tracking_cache[clip_id] = df
                except Exception as e:
                    print(f"Error loading tracking file {txt_path}: {e}")
                    tracking_cache[clip_id] = None
            else:
                tracking_cache[clip_id] = None
                
        # Lookup bbox coordinates
        x1, y1, x2, y2 = 0, 0, 64, 64  # fallback default
        df = tracking_cache.get(clip_id)
        if df is not None:
            match_row = df[(df["frame_id"] == frame_id) & (df["object_id"] == object_id)]
            if not match_row.empty:
                row = match_row.iloc[0]
                box_w = float(row["width"])
                box_h = float(row["height"])
                cx = float(row["x_center"])
                cy = float(row["y_center"])
                
                # Standard cxcywh transformation (matching prepare_paper_inputs.py)
                x1_val = cx - box_w / 2.0
                y1_val = cy - box_h / 2.0
                x2_val = x1_val + box_w
                y2_val = y1_val + box_h
                
                x1 = int(round(x1_val))
                y1 = int(round(y1_val))
                x2 = int(round(x2_val))
                y2 = int(round(y2_val))
        
        # Reconstruct path variables
        rel_app_path = crop_path.relative_to(local_test_inputs_dir).as_posix()
        
        # Check corresponding flow files
        stem = crop_path.stem
        # Try local paths matching the relative paths
        rel_flow_forward = f"flow_forward/{clip_id}/{stem}.npy"
        rel_flow_backward = f"flow_backward/{clip_id}/{stem}.npy"
        
        flow_forward_path = local_test_inputs_dir / rel_flow_forward
        flow_backward_path = local_test_inputs_dir / rel_flow_backward
        
        has_forward = int(flow_forward_path.exists())
        has_backward = int(flow_backward_path.exists())
        
        rows.append({
            "clip_id": clip_id,
            "frame_id": frame_id,
            "object_id": object_id,
            "label": 0,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "appearance_crop": rel_app_path,
            "flow_forward": rel_flow_forward if has_forward else "",
            "flow_backward": rel_flow_backward if has_backward else "",
            "has_forward_flow": has_forward,
            "has_backward_flow": has_backward,
        })
        
    df_manifest = pd.DataFrame(rows)
    manifest_path = Path("/content/test_inputs/test_object_inputs_manifest.csv")
    df_manifest.to_csv(manifest_path, index=False)
    print(f"Generated fallback manifest CSV at {manifest_path} with {len(df_manifest)} rows.")
    return manifest_path


class TestDataset(Dataset):
    """Dataset class to load crops dynamically for batch inference."""

    def __init__(self, manifest_csv: Path, inputs_dir: Path):
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df.dropna(subset=["appearance_crop"])
        self.inputs_dir_str = str(inputs_dir.resolve().as_posix())

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.df.iloc[index]
        clip_id = str(row["clip_id"])
        frame_id = int(row["frame_id"])
        object_id = int(row["object_id"])
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])

        # Load appearance crop
        app_path = adjust_path(row["appearance_crop"], "appearance", self.inputs_dir_str)
        app_tensor = load_gray_tensor(app_path)

        # Load forward optical flow crop
        has_forward = int(row["has_forward_flow"])
        if has_forward and not pd.isna(row["flow_forward"]) and str(row["flow_forward"]).strip():
            forward_path = adjust_path(row["flow_forward"], "flow_forward", self.inputs_dir_str)
            try:
                forward_tensor = load_flow_tensor(forward_path)
            except Exception:
                forward_tensor = torch.zeros(2, 64, 64, dtype=torch.float32)
                has_forward = 0
        else:
            forward_tensor = torch.zeros(2, 64, 64, dtype=torch.float32)
            has_forward = 0

        # Load backward optical flow crop
        has_backward = int(row["has_backward_flow"])
        if has_backward and not pd.isna(row["flow_backward"]) and str(row["flow_backward"]).strip():
            backward_path = adjust_path(row["flow_backward"], "flow_backward", self.inputs_dir_str)
            try:
                backward_tensor = load_flow_tensor(backward_path)
            except Exception:
                backward_tensor = torch.zeros(2, 64, 64, dtype=torch.float32)
                has_backward = 0
        else:
            backward_tensor = torch.zeros(2, 64, 64, dtype=torch.float32)
            has_backward = 0

        return {
            "appearance": app_tensor,
            "flow_forward": forward_tensor,
            "flow_backward": backward_tensor,
            "has_forward": has_forward,
            "has_backward": has_backward,
            "clip_id": clip_id,
            "frame_id": frame_id,
            "object_id": object_id,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }


def load_model_weights(model: torch.nn.Module, path: Path, device: torch.device) -> torch.nn.Module:
    """Safely load state dict into model from saved checkpoint."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint weight file not found at: {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up local directories
    local_test_inputs = Path("/content/test_inputs")
    local_test_inputs = auto_extract_test_data(args.test_inputs_dir, local_test_inputs)

    manifest_path = Path("/content/test_inputs/test_object_inputs_manifest.csv")
    if not manifest_path.exists():
        manifest_path = generate_fallback_manifest(local_test_inputs, args.tracking_dir)

    print("Initializing models...")
    appearance_cae = AppearanceCAE()
    backward_cae = MotionCAE()
    forward_cae = MotionCAE()

    app_classifier = BinaryClassifier(in_channels=1)
    backward_classifier = BinaryClassifier(in_channels=2)
    forward_classifier = BinaryClassifier(in_channels=2)

    # Load weights
    print(f"Loading weights from {args.weights_dir}...")
    try:
        appearance_cae = load_model_weights(appearance_cae, args.weights_dir / "appearance_cae.pt", device)
        backward_cae = load_model_weights(backward_cae, args.weights_dir / "backward_motion_cae.pt", device)
        forward_cae = load_model_weights(forward_cae, args.weights_dir / "forward_motion_cae.pt", device)

        app_classifier = load_model_weights(app_classifier, args.weights_dir / "appearance_classifier.pt", device)
        backward_classifier = load_model_weights(backward_classifier, args.weights_dir / "backward_classifier.pt", device)
        forward_classifier = load_model_weights(forward_classifier, args.weights_dir / "forward_classifier.pt", device)
        print("All model weights loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR loading checkpoints: {e}")
        sys.exit(1)

    # Load dataset & loader
    dataset = TestDataset(manifest_path, local_test_inputs)
    print(f"Loaded test dataset: {len(dataset)} objects found.")
    print(f"Local test inputs dir: {local_test_inputs}")
    print(f"Local test inputs exists: {local_test_inputs.exists()}")
    if local_test_inputs.exists():
        top_level = [str(p.name) for p in local_test_inputs.iterdir()]
        print(f"Contents of local test inputs: {top_level}")
    # Show what paths the manifest contains (first 2 rows)
    raw_df = pd.read_csv(manifest_path)
    print(f"Manifest sample appearance_crop: {list(raw_df['appearance_crop'].head(2))}")
    if 'flow_forward' in raw_df.columns:
        print(f"Manifest sample flow_forward: {list(raw_df['flow_forward'].head(2))}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # 1. Run batch inference to obtain anomaly scores for all objects
    results = []
    print("\nRunning batch inference on objects...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Batch Inference"):
            app_batch = batch["appearance"].to(device)
            back_batch = batch["flow_backward"].to(device)
            forward_batch = batch["flow_forward"].to(device)
            
            has_forward = batch["has_forward"].numpy()
            has_backward = batch["has_backward"].numpy()
            
            clip_ids = batch["clip_id"]
            frame_ids = batch["frame_id"].numpy()
            object_ids = batch["object_id"].numpy()
            x1s = batch["x1"].numpy()
            y1s = batch["y1"].numpy()
            x2s = batch["x2"].numpy()
            y2s = batch["y2"].numpy()

            # Appearance reconstruction and classification
            app_recon, app_encoded = appearance_cae.reconstruct(app_batch)
            app_diff = torch.abs(app_batch - app_recon)
            app_logits = app_classifier(app_diff, app_encoded.latent)
            app_scores = torch.softmax(app_logits, dim=1)[:, 1].cpu().numpy()

            # Backward flow reconstruction and classification
            back_recon, back_encoded = backward_cae.reconstruct(back_batch)
            back_diff = torch.abs(back_batch - back_recon)
            back_logits = backward_classifier(back_diff, back_encoded.latent)
            back_scores = torch.softmax(back_logits, dim=1)[:, 1].cpu().numpy()

            # Forward flow reconstruction and classification
            forward_recon, forward_encoded = forward_cae.reconstruct(forward_batch)
            forward_diff = torch.abs(forward_batch - forward_recon)
            forward_logits = forward_classifier(forward_diff, forward_encoded.latent)
            forward_scores = torch.softmax(forward_logits, dim=1)[:, 1].cpu().numpy()

            # Compute object anomaly score: s(x) = 1 - mean(ŷ_i)
            for i in range(len(app_batch)):
                norm_scores = [app_scores[i]]
                if has_backward[i]:
                    norm_scores.append(back_scores[i])
                if has_forward[i]:
                    norm_scores.append(forward_scores[i])
                # Anomaly score between 0 and 1
                anomaly_score = 1.0 - np.mean(norm_scores)

                results.append({
                    "clip_id": clip_ids[i],
                    "frame_id": int(frame_ids[i]),
                    "object_id": int(object_ids[i]),
                    "x1": int(x1s[i]),
                    "y1": int(y1s[i]),
                    "x2": int(x2s[i]),
                    "y2": int(y2s[i]),
                    "anomaly_score": float(anomaly_score)
                })

    df_results = pd.DataFrame(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(args.output_dir / "object_anomaly_scores.csv", index=False)
    print(f"Saved object-level scores to {args.output_dir}/object_anomaly_scores.csv")

    # 2. Spatio-temporal reassembly and post-processing per clip
    clips = df_results["clip_id"].unique()
    print(f"\nReassembling and post-processing anomaly maps for {len(clips)} clips...")

    all_gts = []
    all_scores = []
    clip_aucs = {}
    
    frame_scores_list = []

    for clip_id in sorted(clips):
        clip_df = df_results[df_results["clip_id"] == clip_id]
        max_frame_id = clip_df["frame_id"].max()
        T = max_frame_id + 1

        # Query frame size from actual frames if available, else use fallback
        W, H = 640, 360
        clip_frames_dir = args.frames_root_dir / clip_id
        if clip_frames_dir.exists():
            images = list(clip_frames_dir.glob("*.jpg")) + list(clip_frames_dir.glob("*.png"))
            if images:
                try:
                    with Image.open(images[0]) as img:
                        W, H = img.size
                except Exception:
                    pass

        # Reconstruct pixel-level anomaly maps (keep max score for overlaps)
        anomaly_maps = np.zeros((T, H, W), dtype=np.float32)
        for _, row in clip_df.iterrows():
            f_id = int(row["frame_id"])
            x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
            score = float(row["anomaly_score"])

            # Boundary safety check
            x1 = max(0, min(x1, W - 1))
            x2 = max(0, min(x2, W))
            y1 = max(0, min(y1, H - 1))
            y2 = max(0, min(y2, H))

            if x2 > x1 and y2 > y1:
                anomaly_maps[f_id, y1:y2, x1:x2] = np.maximum(anomaly_maps[f_id, y1:y2, x1:x2], score)

        # Apply 3D Mean Filter (spatial & temporal smoothing)
        # kernel dimensions: temporal x spatial_y x spatial_x
        temporal_size = args.temporal_filter_size
        spatial_size = args.spatial_filter_size
        
        smoothed_maps = scipy.ndimage.uniform_filter(
            anomaly_maps, 
            size=(temporal_size, spatial_size, spatial_size), 
            mode='constant', 
            cval=0.0
        )

        # Compute raw frame-level anomaly scores (maximum of smoothed map)
        frame_scores = np.max(smoothed_maps, axis=(1, 2))

        # Apply Gaussian Filter (temporal smoothing of frame-level scores)
        smoothed_frame_scores = scipy.ndimage.gaussian_filter1d(frame_scores, sigma=args.gaussian_sigma)

        # Append to CSV results
        for f_id in range(T):
            frame_scores_list.append({
                "clip_id": clip_id,
                "frame_id": f_id,
                "raw_score": float(frame_scores[f_id]),
                "smoothed_score": float(smoothed_frame_scores[f_id])
            })

        # Load Ground Truth (if available) for evaluation
        gt_path = args.gt_dir / f"{clip_id}.npy"
        if gt_path.exists():
            try:
                clip_gt = np.load(gt_path)
                min_len = min(len(clip_gt), T)
                gt_aligned = clip_gt[:min_len]
                scores_aligned = smoothed_frame_scores[:min_len]

                all_gts.extend(gt_aligned)
                all_scores.extend(scores_aligned)

                if len(np.unique(gt_aligned)) > 1:
                    clip_auc = roc_auc_score(gt_aligned, scores_aligned)
                    clip_aucs[clip_id] = clip_auc
                    print(f"Clip {clip_id}: ROC AUC = {clip_auc:.4f}")
                else:
                    # Clip only contains normal frames
                    print(f"Clip {clip_id}: ROC AUC = N/A (single ground truth class)")
            except Exception as e:
                print(f"Warning: Failed to load/evaluate ground truth for {clip_id}: {e}")
        else:
            # Try alternative ground truth format if it is a txt file
            txt_gt_path = args.gt_dir / f"{clip_id}.txt"
            if txt_gt_path.exists():
                try:
                    clip_gt = np.loadtxt(txt_gt_path, dtype=int)
                    min_len = min(len(clip_gt), T)
                    gt_aligned = clip_gt[:min_len]
                    scores_aligned = smoothed_frame_scores[:min_len]

                    all_gts.extend(gt_aligned)
                    all_scores.extend(scores_aligned)

                    if len(np.unique(gt_aligned)) > 1:
                        clip_auc = roc_auc_score(gt_aligned, scores_aligned)
                        clip_aucs[clip_id] = clip_auc
                        print(f"Clip {clip_id}: ROC AUC = {clip_auc:.4f}")
                    else:
                        print(f"Clip {clip_id}: ROC AUC = N/A (single ground truth class)")
                except Exception as e:
                    print(f"Warning: Failed to load/evaluate txt ground truth for {clip_id}: {e}")

    df_frames = pd.DataFrame(frame_scores_list)
    df_frames.to_csv(args.output_dir / "frame_anomaly_scores.csv", index=False)
    print(f"\nSaved frame-level smoothed scores to {args.output_dir}/frame_anomaly_scores.csv")

    # 3. Compute and Print Micro/Macro ROC AUC
    if all_gts and all_scores:
        try:
            # Micro AUC: Concatenated frames across all clips
            micro_auc = roc_auc_score(all_gts, all_scores)
            
            # Macro AUC: Average of clip-level AUCs
            if clip_aucs:
                macro_auc = np.mean(list(clip_aucs.values()))
            else:
                macro_auc = float('nan')

            print("\n================ EVALUATION METRICS ================")
            print(f"Micro Frame-Level ROC AUC (All concatenated): {micro_auc:.4f}")
            print(f"Macro Frame-Level ROC AUC (Average of clip AUCs): {macro_auc:.4f}")
            print("====================================================")

            # Save metrics to JSON file
            metrics = {
                "micro_auc": float(micro_auc),
                "macro_auc": float(macro_auc),
                "clip_aucs": {k: float(v) for k, v in clip_aucs.items()}
            }
            with open(args.output_dir / "evaluation_metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)
            print(f"Saved evaluation metrics to {args.output_dir}/evaluation_metrics.json")
        except Exception as e:
            print(f"Error computing AUC metrics: {e}")
    else:
        print("\nNote: Ground truth files were not found or loaded, skipping AUC metric evaluation.")
        print(f"Place ground truth .npy files in {args.gt_dir} if you wish to compute AUC scores.")


if __name__ == "__main__":
    main()

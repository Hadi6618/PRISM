import os
import numpy as np

def verify_avenue_npy_labels(npy_dir_path):
    print(f"{'File Name':<12} | {'Total Frames':<14} | {'Anomalous':<12} | {'Normal':<10} | {'Status':<8}")
    print("-" * 65)
    
    missing_files = 0
    invalid_files = 0
    
    for i in range(1, 22):
        filename = f"{i}.npy"
        file_path = os.path.join(npy_dir_path, filename)
        
        if not os.path.exists(file_path):
            print(f"{filename:<12} | {'MISSING':<14} | {'-':<12} | {'-':<10} | ❌ Failed")
            missing_files += 1
            continue
            
        try:
            # Load the numpy file
            labels = np.load(file_path)
            
            total_frames = len(labels)
            anomalous_frames = np.sum(labels == 1)
            normal_frames = np.sum(labels == 0)
            
            # Validation checks: Must be 1D, greater than 0 frames, and strictly binary (0 or 1)
            is_binary = np.all((labels == 0) | (labels == 1))
            is_1d = labels.ndim == 1
            
            if is_1d and is_binary and total_frames > 0:
                status = "✅ Pass"
            else:
                status = "❌ Corrupt"
                invalid_files += 1
                
            print(f"{filename:<12} | {total_frames:<14} | {anomalous_frames:<12} | {normal_frames:<10} | {status}")
            
        except Exception as e:
            print(f"{filename:<12} | {'READ ERROR':<14} | {'-':<12} | {'-':<10} | ❌ Error")
            invalid_files += 1

    print("-" * 65)
    print(f"Summary: {21 - missing_files - invalid_files}/21 files passed perfectly.")

# --- Run Verification ---
OUTPUT_DIR = "C:/Users/Windows.10/Downloads/ground_truth_avenue"
verify_avenue_npy_labels(OUTPUT_DIR)
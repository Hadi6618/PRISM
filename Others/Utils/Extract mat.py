import os
import numpy as np
import scipy.io as sio

def batch_convert_avenue_gt(mat_folder_path, output_dir_path):
    """
    Loops through all 21 Avenue test .mat files, extracts 1D frame labels,
    and saves each one as an individual .npy file (1.npy, 2.npy, etc.).
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir_path, exist_ok=True)
    
    # Avenue dataset has exactly 21 test videos
    for i in range(1, 22):
        mat_filename = f"{i}_label.mat"
        mat_path = os.path.join(mat_folder_path, mat_filename)
        
        if not os.path.exists(mat_path):
            print(f"⚠️ Warning: Could not find {mat_filename} in {mat_folder_path}. Skipping.")
            continue
            
        # Load the MATLAB file
        mat_contents = sio.loadmat(mat_path)
        
        # Using 'volLabel' from your snippet
        masks = mat_contents['volLabel'][0]
        
        # Convert 2D pixel masks to 1D binary frame labels
        frame_labels = np.array([1 if np.any(mask > 0) else 0 for mask in masks])
        
        # Construct the individual output path (e.g., .../ground_truth_avenue/1.npy)
        npy_filename = f"{i}.npy"
        npy_path = os.path.join(output_dir_path, npy_filename)
        
        # Save the individual numpy file
        np.save(npy_path, frame_labels)
        
        print(f"✅ Processed and saved {npy_filename}: {len(frame_labels)} frames.")

# --- Execute the Conversion ---
# Replace these paths with your actual local paths
MAT_FOLDER = "C:/Users/Windows.10/Downloads/ground_truth_demo/ground_truth_demo/testing_label_mask"
OUTPUT_DIR = "C:/Users/Windows.10/Downloads/ground_truth_avenue"

batch_convert_avenue_gt(MAT_FOLDER, OUTPUT_DIR)
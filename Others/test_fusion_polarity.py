import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(r"c:\Projects\Graduate Project\Fusion")
spec = importlib.util.spec_from_file_location("fusion", ROOT / "fusion.py")
fusion = importlib.util.module_from_spec(spec)
sys.modules["fusion"] = fusion
spec.loader.exec_module(fusion)


class FusionPolarityTests(unittest.TestCase):
    def test_auto_detect_fixes_stgnf_normality_scores(self):
        # STG-NF values are normality scores: high for normal, low for abnormal.
        # MULDE labels follow the anomaly convention: 1=abnormal, 0=normal.
        stgnf = {
            "01_0001": {
                "frame_indices": np.arange(8, dtype=np.int64),
                "anomaly_scores": np.array(
                    [0.95, 0.90, 0.15, 0.10, 0.85, 0.80, 0.20, 0.25],
                    dtype=np.float32,
                ),
            }
        }
        mulde = {
            "01_0001": {
                "frame_indices": np.arange(8, dtype=np.int64),
                "anomaly_scores": np.array(
                    [0.10, 0.15, 0.80, 0.85, 0.20, 0.25, 0.75, 0.70],
                    dtype=np.float32,
                ),
                "labels": np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.uint8),
            }
        }

        aligned, stats = fusion.align_per_video(
            stgnf,
            mulde,
            auto_detect_offset=True,
        )
        aligned = fusion.apply_normalization(aligned, strategy="global_minmax")
        y = np.concatenate([v.labels for v in aligned])
        s = np.concatenate([v.stgnf_scores for v in aligned])

        # The auto-detected polarity should turn the STG-NF stream into a valid
        # anomaly score instead of leaving it near 0 AUC.
        self.assertEqual(stats["stgnf_frame_offset"], 0)
        self.assertGreater(roc_auc_score(y, s), 0.95)


if __name__ == "__main__":
    unittest.main()

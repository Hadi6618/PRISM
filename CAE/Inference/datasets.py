from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Iterator
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
import random
import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def load_gray_tensor(path: str | Path, size: int = 64) -> Tensor:
    image = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def load_mask_tensor(path: str | Path, size: int = 64) -> Tensor:
    image = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
 
    array = (np.asarray(image, dtype=np.uint8) > 127).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0)


def load_flow_tensor(path: str | Path) -> Tensor:
    array = np.load(path).astype(np.float32)
    if array.ndim != 3 or array.shape[-1] != 2:
        raise ValueError(f"Expected optical flow array shaped HxWx2, got {array.shape} at {path}")
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _existing_path(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value)
    return text if text and Path(text).exists() else ""


def adjust_path(p: object, category: str, inputs_dir_str: str) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    p_str = str(p).replace("\\", "/").strip()
    if not p_str:
        return ""
        
    old_base = "/content/drive/MyDrive/Project/Mohammed models/paper_train_inputs"
    
    # 1. Standard path redirection
    if p_str.startswith(old_base):
        p_local = p_str.replace(old_base, inputs_dir_str)
        if Path(p_local).exists():
            return p_local
            
        # 2. Check if category folder was omitted during zipping
        category_prefix = f"/{category}/"
        if category_prefix in p_local:
            p_alt = p_local.replace(category_prefix, "/", 1)
            if Path(p_alt).exists():
                return p_alt
                
        return p_local
        
    if Path(p_str).exists():
        return p_str
        
    # 3. Fallback path reconstruction
    try:
        p_path = Path(p_str)
        if len(p_path.parts) >= 2:
            clip_id = p_path.parts[-2]
            filename = p_path.name
            
            p_local1 = str(Path(inputs_dir_str) / category / clip_id / filename).replace("\\", "/")
            if Path(p_local1).exists():
                return p_local1
                
            p_local2 = str(Path(inputs_dir_str) / clip_id / filename).replace("\\", "/")
            if Path(p_local2).exists():
                return p_local2
    except Exception:
        pass
        
    return p_str


class AppearanceNormalDataset(Dataset):
    """Normal object crops and generated Mask R-CNN masks."""

    def __init__(self, manifest_csv: str | Path, require_masks: bool = True, cache_in_ram: bool = True):
        manifest = pd.read_csv(manifest_csv)
        '''allowed_clips =  [
          '01_002', '01_003', '01_005', '01_007', '01_009',
          '01_029', '01_031','01_033', '01_035', '01_036',
          '01_037', '01_049','01_051', '01_054', '01_060',
          '01_062','01_066','01_067','01_068',
          '01_069','01_070','01_073','01_083','02_002','02_004',
          '02_005']'''

        #manifest=manifest[manifest['clip_id']]
        manifest = manifest.dropna(subset=['segmentation_mask','prev_frame', 'next_frame'])
        manifest = manifest[manifest['has_forward_flow'] != 0]
        
        if require_masks and "segmentation_mask" not in manifest.columns:
            raise ValueError(
                "segmentation_mask column is missing. Run prepare_masks_and_raft.py first."
            )

        inputs_dir_str = str(Path(manifest_csv).parent.resolve().as_posix())
        rows: list[dict[str, str]] = []
        for row in manifest.to_dict("records"):
            image_path = adjust_path(row.get("appearance_crop"), "appearance", inputs_dir_str)
            mask_path =  adjust_path(row.get("segmentation_mask"), "segmentation_masks", inputs_dir_str)
            if not image_path:
                continue
            if require_masks and not mask_path:
                continue
            rows.append({"image": image_path, "mask": mask_path})

        if not rows:
            raise ValueError(f"No usable appearance rows found in {manifest_csv}")
        self.rows = rows
        self.cache_in_ram = cache_in_ram

        if self.cache_in_ram:
            print(f"Pre-loading {len(rows)} AppearanceNormal samples into RAM in parallel...")
            t0 = time.time()
            def load_single(row):
                x = load_gray_tensor(row["image"])
                mask = load_mask_tensor(row["mask"]) if row["mask"] else torch.ones_like(x)
                return x, mask
            with ThreadPoolExecutor(max_workers=16) as executor:
                self.samples = list(executor.map(load_single, rows))
            print(f"Loaded {len(rows)} samples in {time.time() - t0:.2f}s")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if self.cache_in_ram:
            return self.samples[index]
        row = self.rows[index]
        x = load_gray_tensor(row["image"])
        mask = load_mask_tensor(row["mask"]) if row["mask"] else torch.ones_like(x)
        return x, mask


class AppearancePseudoDataset(Dataset):
    """Pseudo-abnormal appearance images from the allowed folders only."""

    def __init__(self, roots: Iterable[str | Path], cache_in_ram: bool = False):
        paths: list[Path] = []
        for root in roots:
            paths.extend(collect_images(Path(root)))
        if not paths:
            raise ValueError(f"No pseudo-abnormal images found in {list(roots)}")
        self.paths = sorted(paths)
        self.cache_in_ram = cache_in_ram

        if self.cache_in_ram:
            print(f"Pre-loading {len(self.paths)} AppearancePseudo samples into RAM in parallel...")
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=16) as executor:
                self.samples = list(executor.map(load_gray_tensor, self.paths))
            print(f"Loaded {len(self.paths)} samples in {time.time() - t0:.2f}s")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        if self.cache_in_ram:
            return self.samples[index]
        return load_gray_tensor(self.paths[index])


class MotionNormalDataset(Dataset):
    """Normal RAFT motion tensors for one stream: backward or forward."""

    def __init__(self, manifest_csv: str | Path, stream: str, cache_in_ram: bool = False):
        if stream not in {"backward", "forward"}:
            raise ValueError("stream must be 'backward' or 'forward'")
        column = "flow_backward" if stream == "backward" else "flow_forward"
        manifest = pd.read_csv(manifest_csv)
        '''allowed_clips =  [
          '01_002', '01_003', '01_005', '01_007', '01_009',
          '01_029', '01_031','01_033', '01_035', '01_036',
          '01_037', '01_049','01_051', '01_054', '01_060',
          '01_062','01_066','01_067','01_068',
          '01_069','01_070','01_073','01_083','02_002','02_004',
          '02_005']'''
        #manifest=manifest[manifest['clip_id']]
        manifest = manifest.dropna(subset=['segmentation_mask','prev_frame', 'next_frame'])
        manifest = manifest[manifest['has_forward_flow'] != 0]
        inputs_dir_str = str(Path(manifest_csv).parent.resolve().as_posix())
        raw_paths = [str(value) for value in manifest[column].tolist()]
        category = "flow_backward" if stream == "backward" else "flow_forward"
        self.paths = [adjust_path(p, category, inputs_dir_str) for p in raw_paths]
        self.paths = [p for p in self.paths if p]
        if not self.paths:
            raise ValueError(f"No usable {column} files found in {manifest_csv}")
        self.cache_in_ram = cache_in_ram

        if self.cache_in_ram:
            print(f"Pre-loading {len(self.paths)} MotionNormal ({stream}) samples into RAM in parallel...")
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=16) as executor:
                self.samples = list(executor.map(load_flow_tensor, self.paths))
            print(f"Loaded {len(self.paths)} samples in {time.time() - t0:.2f}s")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        if self.cache_in_ram:
            return self.samples[index]
        return load_flow_tensor(self.paths[index])


class MotionPseudoDataset(Dataset):
    """Pseudo-abnormal RAFT motion generated from t-k,t,t+k."""

    def __init__(self,inputs_dir: str | Path,stream: str,cache_in_ram: bool = False,):
        if stream not in {"backward", "forward"}:
            raise ValueError("stream must be 'backward' or 'forward'")

        root_name = (
            "pseudo_flow_backward"
            if stream == "backward"
            else "pseudo_flow_forward"
        )
        self.root = Path(inputs_dir) / root_name

        self.paths = sorted(self.root.rglob("*.npy"))

        if not self.paths:
            raise ValueError(f"No pseudo motion files found in {self.root}")

        self.cache_in_ram = cache_in_ram

        if self.cache_in_ram:
            print(
                f"Pre-loading {len(self.paths)} MotionPseudo ({stream}) samples into RAM in parallel..."
            )
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=16) as executor:
                self.samples = list(executor.map(load_flow_tensor, self.paths))
            print(
                f"Loaded {len(self.paths)} samples in {time.time() - t0:.2f}s"
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        if self.cache_in_ram:
            return self.samples[index]
        return load_flow_tensor(self.paths[index])

def infinite(loader: Iterable) -> Iterator:
    return cycle(loader)

def infinite_pro(loader: Iterable) -> Iterator:
    while True:
        for batch in loader:
            yield batch

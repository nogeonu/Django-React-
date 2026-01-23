"""
MAMA-MIA Dataset Classes
- Stage 1: Tumor Segmentation (DCE-MRI multi-seq + expert mask)
- Stage 2: pCR Classification (DCE-MRI + predicted mask)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from monai.data import PersistentDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    NormalizeIntensityd,
    CropForegroundd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandScaleIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    Lambdad,
    ToTensord,
    CenterSpatialCropd,
)

import config


# -------------------------
# Utilities
# -------------------------

def pad_channels(x: Any) -> Any:
    """
    Ensure channel dimension == config.NUM_SEQUENCES
    - truncate if too many
    - pad with zeros if too few
    """
    target = config.NUM_SEQUENCES
    c = x.shape[0]

    # 1) exactly match
    if c == target:
        return x
        
    # 2) too many channels -> truncate
    if c > target:
        return x[:target]

    # 3) too few -> pad
    pad_c = target - c
    pad_shape = (pad_c, *x.shape[1:])

    if isinstance(x, torch.Tensor):
        padding = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        return torch.cat([x, padding], dim=0)

    padding = np.zeros(pad_shape, dtype=x.dtype)
    return np.concatenate([x, padding], axis=0)


def select_dce_sequences(paths: List[str], num_sequences: int) -> List[str]:
    """
    Select sequences consistently. (Smart Sequence Selection)
    - If enough sequences: pick [0,1,2,last] (washout captured)
    - Else: return all
    """
    if len(paths) >= num_sequences:
        if num_sequences >= 4:
            return [paths[0], paths[1], paths[2], paths[-1]]
        return paths[:num_sequences]
    return paths


def safe_read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


# -------------------------
# Stage 1: Segmentation
# -------------------------

@dataclass(frozen=True)
class SegSample:
    image_paths: List[str]
    label_path: str
    patient_id: str
    pcr: Any


class MAMAMIASegmentationDataset(Dataset):
    """
    Tumor segmentation dataset (Stage 1)
    - Optimized with PersistentDataset (Disk Caching)
    - Split transforms: Deterministic (Cache) vs Stochastic (Runtime)
    """

    def __init__(
        self,
        patient_ids: List[str],
        mode: str = "train",
        use_augmentation: bool = True,
        num_samples_per_case: int = 2,
        cache_dir: Optional[Path] = None
    ):
        assert mode in {"train", "val", "test"}
        self.patient_ids = patient_ids
        self.mode = mode
        self.use_augmentation = use_augmentation and (mode == "train")
        self.num_samples_per_case = int(num_samples_per_case) if mode == "train" else 1

        self.samples: List[SegSample] = self._build_index()
        
        # Build data list for MONAI
        self.data_list = []
        for s in self.samples:
            # Apply Smart Sequence Selection HERE so PersistentDataset caches exactly the right files
            img_paths = select_dce_sequences(s.image_paths, config.NUM_SEQUENCES)
            
            self.data_list.append({
                "image": img_paths,
                "label": s.label_path,
                "patient_id": s.patient_id,
                "pcr": -1 if s.pcr is None else s.pcr,
            })

        self.cache_transforms, self.runtime_transforms = self._get_split_transforms()

        # --------- PersistentDataset Setup ----------
        if cache_dir is not None:
            final_cache_dir = Path(cache_dir)
            final_cache_dir.mkdir(parents=True, exist_ok=True)
            self.base_ds = PersistentDataset(
                data=self.data_list,
                transform=self.cache_transforms,
                cache_dir=str(final_cache_dir),
            )
        else:
            self.base_ds = self.data_list

        print(f"{self.mode} dataset: {len(self.samples)} samples")
        short_seqs = sum(1 for s in self.samples if len(s.image_paths) < config.NUM_SEQUENCES)
        if short_seqs > 0:
            print(f"  - {short_seqs} samples have < {config.NUM_SEQUENCES} sequences (will be padded)")

    def _build_index(self) -> List[SegSample]:
        out: List[SegSample] = []
        missing_seg = []
        missing_meta = []
        for pid in self.patient_ids:
            image_dir = config.IMAGES_DIR / pid
            if not image_dir.exists():
                continue

            paths = sorted([str(p) for p in image_dir.glob(f"{pid.lower()}_*.nii.gz")])
            if not paths:
                continue

            # Try multiple possible segmentation paths
            seg_path = None
            possible_seg_paths = [
                config.SEGMENTATIONS_DIR / f"{pid.lower()}.nii.gz",
                config.DATA_ROOT / "segmentations" / f"{pid.lower()}.nii.gz",
                config.IMAGES_DIR / pid / f"{pid.lower()}_seg.nii.gz",
            ]
            for path in possible_seg_paths:
                if path.exists():
                    seg_path = path
                    break
            
            if seg_path is None:
                missing_seg.append(pid)
                continue

            # Try to load patient info from JSON or CSV
            meta = None
            meta_path = config.PATIENT_INFO_DIR / f"{pid.lower()}.json"
            if meta_path.exists():
                meta = safe_read_json(meta_path)
            
            # If JSON doesn't exist, try to load from CSV
            if meta is None:
                try:
                    import pandas as pd
                    csv_path = config.DATA_ROOT / "clinical.csv"
                    if csv_path.exists():
                        df = pd.read_csv(csv_path)
                        patient_row = df[df["patient_id"] == pid]
                        if not patient_row.empty:
                            meta = {
                                "primary_lesion": {
                                    "pcr": float(patient_row.iloc[0]["pcr"]) if "pcr" in patient_row.columns else None,
                                    "tumor_subtype": patient_row.iloc[0].get("subtype", "luminal") if "subtype" in patient_row.columns else "luminal"
                                },
                                "clinical_data": {
                                    "age": float(patient_row.iloc[0]["age"]) if "age" in patient_row.columns else 48
                                }
                            }
                except Exception as e:
                    print(f"Warning: Could not load patient info for {pid}: {e}")
                    meta = None

            if meta is None:
                missing_meta.append(pid)
                continue

            pcr = meta.get("primary_lesion", {}).get("pcr", None)

            out.append(
                SegSample(
                    image_paths=paths,
                    label_path=str(seg_path),
                    patient_id=pid,
                    pcr=pcr,
                )
            )
        
        if missing_seg:
            print(f"Warning: {len(missing_seg)} patients missing segmentation files (first 5: {missing_seg[:5]})")
        if missing_meta:
            print(f"Warning: {len(missing_meta)} patients missing metadata (first 5: {missing_meta[:5]})")
        
        return out

    def _get_split_transforms(self):
        # -----------------------
        # 1. Deterministic stage (Heavy + Cached on Disk)
        # -----------------------
        cache_t = Compose([
            LoadImaged(keys=["image", "label"], reader="NibabelReader"),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=config.SPACING, mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=False),
            # pad_channels(x) assumes config.NUM_SEQUENCES
            Lambdad(keys=["image"], func=pad_channels),
        ])

        # -----------------------
        # 2. Runtime stage (Crop / Patch / Augment)
        # -----------------------
        runtime = [
            CropForegroundd(keys=["image", "label"], source_key="image"),
            SpatialPadd(keys=["image", "label"], spatial_size=config.PATCH_SIZE),
        ]

        if self.mode == "train":
            runtime.append(
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=config.PATCH_SIZE,
                    pos=4,
                    neg=1,
                    num_samples=self.num_samples_per_case,
                    image_key="image",
                    image_threshold=0,
                )
            )

            if self.use_augmentation:
                runtime += [
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
                    RandAffined(
                        keys=["image", "label"],
                        prob=0.3,
                        rotate_range=(np.radians(config.ROTATION_RANGE),) * 3,
                        scale_range=(0.1, 0.1, 0.1),
                        mode=("bilinear", "nearest"),
                    ),
                    RandScaleIntensityd(keys=["image"], factors=config.INTENSITY_SHIFT, prob=0.5),
                    RandGaussianNoised(keys=["image"], prob=0.2, std=0.01),
                    RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.0)),
                ]

        runtime.append(ToTensord(keys=["image", "label"]))

        return cache_t, Compose(runtime)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        try:
            # 1) Deterministic stage (Heavy - Cached or Fresh)
            if isinstance(self.base_ds, PersistentDataset):
                data = self.base_ds[idx]
            else:
                data = self.cache_transforms(self.data_list[idx])

            # 2) Runtime stage (Stochastic/Crop)
            data = self.runtime_transforms(data)
            return data

        except Exception as e:
            print(f"Error in __getitem__ (Idx {idx}): {e}")
            return None


# -------------------------
# Stage 2: Classification
# -------------------------

class MAMAMIAClassificationDataset(Dataset):
    """
    pCR classification dataset (Stage 2 - Legacy/Deprecated)
    Kept for reference or full-volume experiments.
    """

    def __init__(
        self,
        patient_ids: List[str],
        seg_predictions_dir: Path,
        mode: str = "train",
        use_augmentation: bool = True,
    ):
        assert mode in {"train", "val", "test"}
        self.patient_ids = patient_ids
        self.seg_predictions_dir = Path(seg_predictions_dir)
        self.mode = mode
        self.use_augmentation = use_augmentation and (mode == "train")

        self.data_dicts = self._build_index()
        self.transforms = self._build_transforms()

    def _build_index(self) -> List[Dict[str, Any]]:
        out = []
        for pid in self.patient_ids:
            image_dir = config.IMAGES_DIR / pid
            if not image_dir.exists():
                continue

            img_paths = sorted([str(p) for p in image_dir.glob(f"{pid.lower()}_*.nii.gz")])
            if len(img_paths) < config.NUM_SEQUENCES:
                continue

            img_paths = select_dce_sequences(img_paths, config.NUM_SEQUENCES)

            pred_seg = self.seg_predictions_dir / f"{pid}_seg.nii.gz"
            if not pred_seg.exists():
                continue

            meta = safe_read_json(config.PATIENT_INFO_DIR / f"{pid.lower()}.json")
            if meta is None:
                continue
            pcr = meta.get("primary_lesion", {}).get("pcr", None)
            if pcr is None:
                continue

            out.append({"image": img_paths, "mask": str(pred_seg), "patient_id": pid, "pcr": float(pcr)})
        return out

    def _build_transforms(self):
        base = [
            LoadImaged(keys=["image", "mask"], reader="NibabelReader"),
            EnsureChannelFirstd(keys=["image", "mask"]),
            Spacingd(keys=["image", "mask"], pixdim=config.SPACING, mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=False),
            Lambdad(keys=["image"], func=pad_channels),
        ]

        spatial = [
            CropForegroundd(keys=["image", "mask"], source_key="mask", margin=10, allow_smaller=False), # Crop to mask
            SpatialPadd(keys=["image", "mask"], spatial_size=config.PATCH_SIZE), # Pad if smaller
            CenterSpatialCropd(keys=["image", "mask"], roi_size=config.PATCH_SIZE), # Crop if larger
        ]

        final = [ToTensord(keys=["image", "mask"])]

        return Compose(base + spatial + final)

    def __len__(self) -> int:
        return len(self.data_dicts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        d = self.data_dicts[idx]
        data = self.transforms(d)
        x = torch.cat([data["image"], data["mask"]], dim=0)

        return {
            "image": x,
            "label": torch.tensor(d["pcr"], dtype=torch.float32),
            "patient_id": d["patient_id"],
        }


# -------------------------
# Stage 9 (New): Phase 3 Classification (Focused ROI)
# -------------------------

class MAMAMIAFocusedDataset(Dataset):
    """
    Phase 3: Characterization Dataset
    Loads pre-cached 'Focused ROIs' (.pt files) generated in Phase 2.
    """
    def __init__(self, patient_ids: List[str], cache_dir: Path, mode: str = "train", use_augmentation: bool = True):
        self.patient_ids = patient_ids
        self.cache_dir = Path(cache_dir)
        self.mode = mode
        self.use_augmentation = use_augmentation and (mode == "train")
        
        self.data_files = self._build_index()
        self.transforms = self._get_transforms()
        
    def _build_index(self):
        files = []
        for pid in self.patient_ids:
            p = self.cache_dir / f"{pid}.pt"
            if p.exists():
                files.append(p)
        return files
        
    def _get_transforms(self):
        # Data is already Spaced, Normalized, Cropped, and Tensor-ized.
        # We only need Augmentation.
        tr = []
        if self.use_augmentation:
            # Keys in .pt: 'image' (C,H,W,D), 'label' (int), 'patient_id' (str)
            tr.extend([
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandRotate90d(keys=["image"], prob=0.5, max_k=3),
                RandAffined(
                    keys=["image"],
                    prob=0.3, 
                    rotate_range=(np.radians(15),)*3,
                    scale_range=(0.1, 0.1, 0.1),
                    mode=("bilinear")
                ),
                RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
                RandGaussianNoised(keys=["image"], prob=0.1, std=0.01),
            ])
        return Compose(tr)

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        try:
            # Load .pt file directly
            # It contains {"image": Tensor, "label": int, "patient_id": str}
            data_dict = torch.load(self.data_files[idx], map_location="cpu", weights_only=False)
            pid = data_dict["patient_id"]
            
            # --- Phase 4: Clinical Feature Extraction ---
            # 1. Load JSON
            meta_path = config.PATIENT_INFO_DIR / f"{pid.lower()}.json"
            meta = safe_read_json(meta_path)
            
            clinical_tensor = torch.zeros(config.CLINICAL_CONFIG["feature_dim"], dtype=torch.float32)
            
            if meta:
                # 2. Age Normalization (Min-Max)
                clinical_data = meta.get("clinical_data", {})
                if clinical_data is None: clinical_data = {}
                age = clinical_data.get("age", 48)
                if age is None: age = 48
                age_norm = (float(age) - 20) / 60.0 # Range 20-80 approx
                clinical_tensor[0] = age_norm
                
                # 3. Subtype One-hot Encoding
                primary_lesion = meta.get("primary_lesion", {})
                if primary_lesion is None: primary_lesion = {}
                subtype = primary_lesion.get("tumor_subtype", "luminal")
                if subtype is None: subtype = "luminal"
                subtype = str(subtype).lower()
                subtype_idx = config.SUBTYPE_MAPPING.get(subtype, 0) # Default luminal (0)
                clinical_tensor[1 + subtype_idx] = 1.0
            
            # Apply Augmentation
            if self.use_augmentation:
                 # Monai transforms expect dictionary
                 aug_input = {"image": data_dict["image"]}
                 aug_output = self.transforms(aug_input)
                 image = aug_output["image"]
            else:
                 image = data_dict["image"]
            
            return {
                "image": image,              # (4, 64, 64, 64)
                "clinical": clinical_tensor,   # (7,) vector
                "label": torch.tensor(data_dict["label"], dtype=torch.float32),
                "patient_id": pid
            }
        except Exception as e:
            print(f"Error loading {self.data_files[idx]}: {e}")
            return None


# -------------------------
# Splits
# -------------------------

def get_train_val_split(val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(config.TRAIN_TEST_SPLIT)
    train_ids = df["train_split"].dropna().tolist()
    rng = np.random.RandomState(seed)
    rng.shuffle(train_ids)
    split_idx = int(len(train_ids) * (1 - val_ratio))
    return train_ids[:split_idx], train_ids[split_idx:]


def get_test_ids() -> List[str]:
    df = pd.read_csv(config.TRAIN_TEST_SPLIT)
    return df["test_split"].dropna().tolist()

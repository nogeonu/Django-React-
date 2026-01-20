"""
MAMA-MIA Medical Transformer Pipeline Configuration
Optimized for RTX 2060 6GB VRAM
"""
import os
from pathlib import Path

# ============================================================================
# Paths
# ============================================================================
# Get project root directory (where config.py is located)
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_ROOT / "images"
SEGMENTATIONS_DIR = DATA_ROOT / "segmentations" / "expert"
PATIENT_INFO_DIR = DATA_ROOT / "patient_info_files"
TRAIN_TEST_SPLIT = DATA_ROOT / "train_test_splits.csv"

# Output directories (use absolute path for outputs to avoid issues)
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
LOG_DIR = OUTPUT_ROOT / "logs"
RESULTS_DIR = OUTPUT_ROOT / "results"
CACHE_DIR = PROJECT_ROOT / "cache"

# Create directories
for dir_path in [OUTPUT_ROOT, CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
    # Create sub-cache dirs for v2 safety
    (CACHE_DIR / "seg_train_v2").mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "seg_val_v2").mkdir(parents=True, exist_ok=True)

# ============================================================================
# Hardware Optimization (RTX 3060 12GB VRAM)
# ============================================================================
USE_MIXED_PRECISION = True  # FP16 training
USE_GRADIENT_CHECKPOINTING = True  # Save VRAM at cost of compute
NUM_WORKERS = 12  # Ryzen 9 3900X (12-Core): Full power as requested
BATCH_SIZE = 1 # Optimized for 12GB VRAM (Safe for 128^3)
PIN_MEMORY = True

# ============================================================================
# Data Configuration
# ============================================================================
# DCE-MRI sequences (0000-0004, some patients have 4, some have 5)
NUM_SEQUENCES = 4  # Use first 4 sequences for consistency

# Image preprocessing
SPACING = (1.5, 1.5, 1.5)  # Target spacing in mm (isotropic)
PATCH_SIZE = (128, 128, 128)  # Training patch size (High Res maintained)
ROI_SIZE = (96, 96, 96)       # For Classification/Multimodal Cropping
OVERLAP = 0.25  # Overlap for sliding window inference

# Intensity normalization
INTENSITY_NORM = "zscore"  # Options: "zscore", "minmax", "percentile"

# Data augmentation
USE_AUGMENTATION = True
AUG_PROB = 0.5
ROTATION_RANGE = 15  # degrees
INTENSITY_SHIFT = 0.1  # ±10%

# ============================================================================
# Segmentation Model (SwinUNETR + LoRA)
# ============================================================================
SEG_MODEL_CONFIG = {
    "img_size": PATCH_SIZE,
    "in_channels": NUM_SEQUENCES,  # Multi-phase DCE-MRI
    "out_channels": 1,  # Binary tumor mask
    "feature_size": 24,  # Reduced back to 24 for Speed (Compensates for 128 patch)
    "depths": [2, 2, 2, 2],
    "num_heads": [3, 6, 12, 24],
    "use_checkpoint": USE_GRADIENT_CHECKPOINTING,
}

# LoRA configuration for segmentation
SEG_LORA_CONFIG = {
    "r": 8,  # Rank
    "lora_alpha": 16,  # Scaling factor
    "lora_dropout": 0.1,
    "target_modules": ["qkv"],  # Apply to attention layers
}

# Segmentation training
SEG_TRAIN_CONFIG = {
    "batch_size": 1,  # 128^3 input fits with batch=1
    "gradient_accumulation_steps": 2,  # Effective batch size = 2
    "num_epochs": 200,
    "learning_rate": 5e-4, 
    "weight_decay": 1e-5,
    "lr_scheduler": "cosine",
    "warmup_epochs": 10,
    "early_stopping_patience": 30, # Increased patience
    "val_interval": 1, 
}

# Loss function
SEG_LOSS_CONFIG = {
    "dice_weight": 0.5,
    "focal_weight": 0.5,
    "focal_gamma": 2.0,
    "focal_alpha": 0.75,  # Weight for positive samples
    "squared_pred": True, # For better boundary stability
}

# EMA and Inference Optimization
SEG_INF_CONFIG = {
    "ema_decay": 0.999,  # Decay for EMA
    "val_thresholds": [0.35, 0.4, 0.45, 0.5, 0.55, 0.6],  # Thresholds to sweep during validation
}

# ============================================================================
# Phase 4: Multimodal Fusion Configuration
# ============================================================================
PHASE4_CACHE_DIR = DATA_ROOT / "phase2_cache_v2_crop_only"

CLINICAL_CONFIG = {
    "subtype_classes": 6,  # luminal, luminal_a, luminal_b, her2_enriched, her2_pure, triple_negative
    "feature_dim": 7,      # Age (1) + Subtype (6 one-hot)
    "head_hidden_dim": 32,
    "head_dropout": 0.2
}

SUBTYPE_MAPPING = {
    "luminal": 0,
    "luminal_a": 1,
    "luminal_b": 2,
    "her2_enriched": 3,
    "her2_pure": 4,
    "triple_negative": 5
}

# ============================================================================
# Classification Model (Vision Transformer + LoRA)
# ============================================================================
CLASS_MODEL_CONFIG = {
    "img_size": ROI_SIZE,     # Use ROI_SIZE (96^3) for Classification
    "in_channels": NUM_SEQUENCES + 1,  # DCE-MRI + predicted mask
    "num_classes": 1,  # Binary pCR prediction
    "patch_size": 16,
    "embed_dim": 384,  # ViT-Small
    "depth": 12,
    "num_heads": 6,
    "mlp_ratio": 4.0,
    "dropout": 0.1,
}

# LoRA configuration for classification
CLASS_LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "target_modules": ["qkv", "proj"],
}

# Classification training
CLASS_TRAIN_CONFIG = {
    "batch_size": 4,  # Can be larger than segmentation
    "gradient_accumulation_steps": 2,  # Effective batch size = 8
    "num_epochs": 100,
    "learning_rate": 5e-5,
    "weight_decay": 1e-5,
    "lr_scheduler": "cosine",
    "warmup_epochs": 5,
    "early_stopping_patience": 15,
    "val_interval": 1,
}

# Class weighting (handle imbalance)
CLASS_LOSS_CONFIG = {
    "pos_weight": None,  # Will be computed from data
    "label_smoothing": 0.1,
}

# ============================================================================
# Evaluation Metrics
# ============================================================================
SEG_METRICS = ["dice", "iou", "hausdorff95"]
CLASS_METRICS = ["auc", "accuracy", "precision", "recall", "f1"]

# ============================================================================
# Inference Configuration
# ============================================================================
INFERENCE_CONFIG = {
    "use_sliding_window": True,
    "overlap": OVERLAP,
    "blend_mode": "gaussian",  # Smooth blending at patch boundaries
    "tta": False,  # Test-time augmentation (flip/rotate)
}

# ============================================================================
# Random Seed
# ============================================================================
# Device Configuration
# ============================================================================
import torch
DEVICE = "cuda" # or "cpu"
RANDOM_SEED = 42  # Seed for reproducibility
print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

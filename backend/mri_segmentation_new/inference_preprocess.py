"""
Preprocessing Module for Phase 1 Segmentation Inference
Matches the exact preprocessing pipeline used during training.
"""
import torch
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, NormalizeIntensityd, EnsureTyped
)
import config


def get_inference_transforms():
    """
    Returns the preprocessing transform pipeline for inference.
    Must match the training preprocessing exactly.
    """
    return Compose([
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=config.SPACING,
            mode="bilinear"
        ),
        # Select first 4 sequences (matching training)
        # This assumes input has shape [C, H, W, D] where C >= 4
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"], dtype=torch.float32)
    ])


def preprocess_single_case(image_path, num_sequences=4):
    """
    Preprocess a single DCE-MRI case for inference.
    
    Args:
        image_path: Path to either:
            - A single multi-sequence NIfTI file (e.g., patient_001.nii.gz)
            - A folder containing individual sequence files (e.g., DUKE_001/)
        num_sequences: Number of sequences to use (default: 4, matching training)
    
    Returns:
        dict: Preprocessed data dictionary with keys:
            - 'image': Tensor of shape [1, 4, H, W, D] (batch dimension added)
            - 'image_meta_dict': Metadata for coordinate restoration
    """
    from pathlib import Path
    import glob
    
    image_path = Path(image_path)
    
    # Check if input is a directory (multi-file case)
    if image_path.is_dir():
        # Find all sequence files in the directory
        sequence_files = sorted(glob.glob(str(image_path / "*.nii.gz")))
        # Filter out metadata files
        sequence_files = [f for f in sequence_files if "metadata" not in f.lower()]
        
        if len(sequence_files) == 0:
            raise FileNotFoundError(f"No .nii.gz files found in {image_path}")
        
        # Select first N sequences (matching training)
        sequence_files = sequence_files[:num_sequences]
        
        if len(sequence_files) < num_sequences:
            raise ValueError(
                f"Found {len(sequence_files)} sequences in {image_path}, "
                f"but {num_sequences} required."
            )
        
        print(f"  Loading {len(sequence_files)} sequence files from folder...")
        image_input = sequence_files
    else:
        # Single file case
        if not image_path.exists():
            raise FileNotFoundError(f"File not found: {image_path}")
        image_input = str(image_path)
    
    transforms = get_inference_transforms()
    
    # Load and preprocess
    data = {"image": image_input}
    preprocessed = transforms(data)
    
    # Ensure we have the right number of sequences
    image = preprocessed["image"]
    if image.shape[0] > num_sequences:
        image = image[:num_sequences]
    elif image.shape[0] < num_sequences:
        raise ValueError(
            f"Input has {image.shape[0]} sequences, but {num_sequences} required. "
            f"Please check your DCE-MRI data."
        )
    
    # Add batch dimension
    preprocessed["image"] = image.unsqueeze(0)  # [1, 4, H, W, D]
    
    return preprocessed


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference_preprocess.py <path_to_nifti>")
        sys.exit(1)
    
    test_path = sys.argv[1]
    print(f"Preprocessing: {test_path}")
    
    result = preprocess_single_case(test_path)
    print(f"Output shape: {result['image'].shape}")
    print(f"Value range: [{result['image'].min():.3f}, {result['image'].max():.3f}]")
    print("Preprocessing successful!")

# Phase 1 Segmentation Inference Guide

## 🚀 Quick Start (For Deployment Team)

### Prerequisites
```bash
pip install torch monai nibabel scipy
```

### Basic Usage (One Command)
```bash
python inference_pipeline.py \
    --model C:\datasets\MAMA_MIA\outputs\checkpoints\segmentation\best_model.pth \
    --input /path/to/patient_001.nii.gz \
    --output /path/to/output_segmentation.nii.gz
```

### Advanced Options
```bash
python inference_pipeline.py \
    --model best_model.pth \
    --input patient_data.nii.gz \
    --output result.nii.gz \
    --threshold 0.6 \          # Adjust sensitivity (default: 0.5)
    --device cuda \             # or 'cpu'
    --use-ema                   # Use EMA weights for better performance
```

---

## 📦 Module Overview

### 1. `inference_preprocess.py`
Handles data preprocessing (matching training pipeline):
- Loads NIfTI files
- Resamples to 1.5mm isotropic spacing
- Normalizes intensity (Z-score, channel-wise)
- Selects first 4 DCE sequences

### 2. `inference_postprocess.py`
Converts model output to final segmentation:
- Applies threshold (default: 0.5)
- Removes small noise components
- Fills holes in segmentation
- Optional: Restores original patient spacing

### 3. `inference_pipeline.py` ⭐
**Main entry point** - Complete end-to-end pipeline:
- Loads trained model
- Runs preprocessing → inference → postprocessing
- Saves results
- Provides statistics (tumor volume, detection status)

---

## 🔧 Integration Example (Python API)

```python
from inference_pipeline import SegmentationInferencePipeline

# Initialize once
pipeline = SegmentationInferencePipeline(
    model_path="best_model.pth",
    device="cuda",
    threshold=0.5
)

# Process multiple cases
for patient_file in patient_list:
    results = pipeline.predict(
        image_path=patient_file,
        output_path=f"output_{patient_file.stem}.nii.gz"
    )
    
    print(f"Patient: {patient_file.name}")
    print(f"  Tumor detected: {results['tumor_detected']}")
    print(f"  Volume: {results['tumor_volume_voxels']} voxels")
```

---

## 📊 Input/Output Specifications

### Input Requirements
- **Format**: NIfTI (`.nii` or `.nii.gz`)
- **Sequences**: At least 4 DCE-MRI sequences (channels)
- **Orientation**: Any (automatically reoriented to RAS)
- **Spacing**: Any (automatically resampled to 1.5mm)

### Output Format
- **Format**: NIfTI (`.nii.gz`)
- **Values**: Binary mask (0 = background, 1 = tumor)
- **Spacing**: 1.5mm isotropic (matching model training)
- **Coordinate System**: Aligned with input

---

## ⚙️ Configuration

Key parameters in `config.py`:
- `PATCH_SIZE = (128, 128, 128)`: Sliding window size
- `SPACING = (1.5, 1.5, 1.5)`: Target voxel spacing
- `OVERLAP = 0.25`: Sliding window overlap ratio

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Use CPU mode
```bash
python inference_pipeline.py --device cpu ...
```

### Issue: "Input has X sequences, but 4 required"
**Solution**: Check your DCE-MRI data. Model expects 4 temporal sequences.

### Issue: "No tumor detected" (but you see one visually)
**Solution**: Lower threshold
```bash
python inference_pipeline.py --threshold 0.3 ...
```

---

## 📞 Support

For deployment issues, contact the ML team with:
1. Input file path
2. Error message
3. Command used
4. GPU/CPU info (`nvidia-smi` output)

---

## 🎯 Performance Notes

- **Speed**: ~30-60 seconds per case (RTX 3060, CUDA)
- **Memory**: ~12GB VRAM (GPU) or ~16GB RAM (CPU)
- **Accuracy**: Dice Score ~0.76 on validation set

**Model Version**: Phase 1 Segmentation (Epoch 89, Dice 0.7625)
**Last Updated**: 2026-01-16

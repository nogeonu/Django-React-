# MAMA-MIA Phase 1 Deployment Package

This package contains the complete Phase 1 Segmentation pipeline, including the trained model, inference code, and sample data.

## Folder Structure

```
MAMA_MIA_DELIVERY_PKG/
├── checkpoints/       # Contains 'best_model.pth'
├── src/              # Source code (preprocessing, inference, postprocessing)
├── sample_data/      # Sample DCE-MRI case (ISPY2_213913)
├── results/          # Output directory (created after running demo)
└── run_demo.py       # Simple script to test the model
```

## How to Run

1. **Environment Setup**
   Ensure you have the required dependencies installed. You can install them using:
   ```bash
   pip install -r src/requirements.txt
   ```
   *Note: Requires `pydicom==2.4.4` and `pydicom-seg==0.4.1`.*

2. **Run Demo**
   Execute the demo script to verify everything is working:
   ```bash
   python run_demo.py
   ```

3. **Output**
   The script will generate:
   - `results/ISPY2_213913_segmentation.nii.gz`: 3D segmentation mask
   - `results/ISPY2_213913_visualization.png`: Visualization of the segmentation

## Key Features

- **Robust Preprocessing**: Handles DICOM/NIfTI inputs, standardizing to 1.5mm resolution.
- **Accurate Segmentation**: Uses SwinUNETR + LoRA model trained on DUKE/ISPY2 datasets.
- **Standardized Output**: Defaults to NIfTI (.nii.gz).
- **DICOM SEG Support**: capable of generating DICOM Segmentation Objects (.dcm) for PACS overlay if input is a DICOM series folder.
    - Set `output_format='dicom'` in API or pipeline.
- **Improved Post-processing**: 
    - Automatically restores segmentation mask to **original patient geometry** (Spacing & Orientation).
    - Removes small artifacts and fills holes.
- **CPU Optimized**: 
    - Verified to run efficiently on standard CPU nodes (Infrastructure Cost Save!).
    - Benchmark: ~8 seconds per case on Standard CPU (4 vCPU tested).
    - Min Specs: 4 vCPU, 8GB RAM recommended.
- **Deployment Ready**: Modular code structure suitable for API integration.

## Contact

If you encounter any issues, please contact the MAMA-MIA AI Team.

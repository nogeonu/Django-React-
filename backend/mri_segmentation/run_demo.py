"""
MAMA-MIA Phase 1 Segmentation Demo Script
This script demonstrates the breast tumor segmentation model using sample data.
"""
import sys
import os
from pathlib import Path

# Add src to python path to import modules
SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

from inference_pipeline import SegmentationInferencePipeline
from visualize_segmentation import visualize_segmentation

def main():
    # 1. Setup Paths
    BASE_DIR = Path(__file__).parent
    
    MODEL_PATH = BASE_DIR / "checkpoints" / "best_model.pth"
    INPUT_PATH = BASE_DIR / "sample_data" / "ISPY2_213913"
    OUTPUT_DIR = BASE_DIR / "results"
    
    OUTPUT_FILE = OUTPUT_DIR / "ISPY2_213913_segmentation.nii.gz"
    VIZ_FILE = OUTPUT_DIR / "ISPY2_213913_visualization.png"
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("="*60)
    print("MAMA-MIA Phase 1 Segmentation Demo")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_FILE}")
    print("="*60)
    
    if not MODEL_PATH.exists():
        print("Error: Model file not found at", MODEL_PATH)
        return

    # 2. Load Model
    print("\n[1/4] Loading model (CPU Mode)...")
    import time
    start_time = time.time()
    
    pipeline = SegmentationInferencePipeline(str(MODEL_PATH), device="cpu") 
    
    # 3. Run Inference
    print("\n[2/4] Running inference...")
    inf_start = time.time()
    result = pipeline.predict(
        str(INPUT_PATH),
        output_path=str(OUTPUT_FILE)
    )
    inf_end = time.time()
    
    print(f"Inference Time (NIfTI): {inf_end - inf_start:.2f} seconds")
    
    # 3.1 Run Inference (DICOM SEG) - Only if input has DICOMs
    dct_files = list(INPUT_PATH.glob("*.dcm"))
    if dct_files:
        print("\n[2.1/4] Running inference (DICOM SEG Output)...")
        DICOM_OUTPUT = OUTPUT_DIR / "ISPY2_213913_segmentation.dcm"
        inf_start = time.time()
        result_dcm = pipeline.predict(
            str(INPUT_PATH),
            output_path=str(DICOM_OUTPUT),
            output_format="dicom"
        )
        inf_end = time.time()
        print(f"Inference Time (DICOM): {inf_end - inf_start:.2f} seconds")
    else:
        print("\n[2.1/4] Skipping DICOM SEG Output (No .dcm files in sample data)")
        print("         To test DICOM capabilities, please provide a folder with .dcm files.")
    
    # 4. Print Results
    print("\n[3/4] Inference complete!")
    print(f"  Tumor detected: {result['tumor_detected']}")
    print(f"  Tumor volume: {result['tumor_volume_voxels']} voxels")
    if result['tumor_detected']:
        # Calculate volume in cm3 (assuming 1.5mm spacing? No, restored spacing!)
        # Check header
        pass
        
    print(f"  Segmentation saved to: {OUTPUT_FILE.name}")
    
    # 5. Visualize
    print("\n[4/4] Creating visualization...")
    visualize_segmentation(
        image_path=str(INPUT_PATH),
        segmentation_path=str(OUTPUT_FILE),
        output_path=str(VIZ_FILE),
        num_slices=5
    )
    
    print(f"\n[Done] Demo Completed Successfully!")
    print(f"  Check results in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

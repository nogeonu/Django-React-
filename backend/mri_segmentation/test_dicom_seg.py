"""
Test DICOM SEG generation with real DICOM input
"""
import sys
sys.path.insert(0, 'src')

from inference_pipeline import SegmentationInferencePipeline
from pathlib import Path
import time

def main():
    print("="*60)
    print("DICOM SEG Generation Test")
    print("="*60)
    
    # Paths
    MODEL_PATH = Path("checkpoints/best_model.pth")
    DICOM_INPUT = Path("sample_data/ISPY2_213913_DICOM")
    OUTPUT_DIR = Path("results")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Initialize pipeline (CPU mode)
    print("\n[1/3] Loading model (CPU Mode)...")
    pipeline = SegmentationInferencePipeline(
        model_path=str(MODEL_PATH),
        device="cpu",
        threshold=0.5
    )
    
    # Run inference with DICOM SEG output
    print("\n[2/3] Running inference with DICOM input...")
    DICOM_SEG_OUTPUT = OUTPUT_DIR / "ISPY2_213913_seg.dcm"
    
    start_time = time.time()
    result = pipeline.predict(
        str(DICOM_INPUT),
        output_path=str(DICOM_SEG_OUTPUT),
        output_format="dicom"
    )
    end_time = time.time()
    
    print(f"\n[3/3] Results:")
    print(f"  Inference Time: {end_time - start_time:.2f} seconds")
    print(f"  Tumor Detected: {result['tumor_detected']}")
    print(f"  Tumor Volume: {result['tumor_volume_voxels']} voxels")
    print(f"  DICOM SEG saved to: {DICOM_SEG_OUTPUT}")
    
    # Verify DICOM SEG file
    if DICOM_SEG_OUTPUT.exists():
        import pydicom
        ds = pydicom.dcmread(str(DICOM_SEG_OUTPUT))
        print(f"\n  DICOM SEG Verification:")
        print(f"    Modality: {ds.Modality}")
        print(f"    Number of Frames: {ds.NumberOfFrames}")
        print(f"    Rows x Columns: {ds.Rows} x {ds.Columns}")
        print(f"    Patient ID: {ds.PatientID}")
        print("\n[Done] DICOM SEG test completed successfully!")
    else:
        print("\n[Error] DICOM SEG file was not created!")

if __name__ == "__main__":
    main()

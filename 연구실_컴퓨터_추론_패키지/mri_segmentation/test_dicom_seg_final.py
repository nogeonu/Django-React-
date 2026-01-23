"""
Final DICOM SEG test with 4-channel DICOM input
"""
import sys
sys.path.insert(0, 'src')

from inference_pipeline import SegmentationInferencePipeline
from pathlib import Path
import time

def main():
    print("="*60)
    print("DICOM SEG Generation Test (4-Channel)")
    print("="*60)
    
    # Paths
    MODEL_PATH = Path("checkpoints/best_model.pth")
    DICOM_INPUT = Path("sample_data/ISPY2_213913_DICOM_4CH")
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
    print("\n[2/3] Running inference with 4-channel DICOM input...")
    DICOM_SEG_OUTPUT = OUTPUT_DIR / "ISPY2_213913_seg_final.dcm"
    
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
        print(f"    Frame of Reference UID: {ds.FrameOfReferenceUID}")
        
        # Check frame positions
        if hasattr(ds, 'PerFrameFunctionalGroupsSequence'):
            print(f"    Per-Frame Groups: {len(ds.PerFrameFunctionalGroupsSequence)} frames")
            if len(ds.PerFrameFunctionalGroupsSequence) > 0:
                first_frame = ds.PerFrameFunctionalGroupsSequence[0]
                if hasattr(first_frame, 'PlanePositionSequence'):
                    pos = first_frame.PlanePositionSequence[0].ImagePositionPatient
                    print(f"    First Frame Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        
        print("\n[SUCCESS] DICOM SEG test completed successfully!")
    else:
        print("\n[ERROR] DICOM SEG file was not created!")

if __name__ == "__main__":
    main()

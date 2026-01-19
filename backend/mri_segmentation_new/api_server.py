"""
FastAPI Server for Phase 1 Segmentation
GCP Cloud Run compatible
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import torch
import tempfile
import os
from pathlib import Path
from inference_pipeline import SegmentationInferencePipeline

app = FastAPI(title="Phase 1 Tumor Segmentation API")

# Load model at startup
MODEL_PATH = "best_model.pth"
pipeline = None

@app.on_event("startup")
async def load_model():
    global pipeline
    print(f"Loading model from {MODEL_PATH}...")
    pipeline = SegmentationInferencePipeline(MODEL_PATH)
    print("Model loaded successfully!")

@app.get("/")
async def root():
    return {
        "service": "Phase 1 Tumor Segmentation",
        "version": "1.0",
        "status": "ready"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": pipeline is not None}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    output_format: str = "nifti"
):
    """
    Predict tumor segmentation from MRI image.
    
    Args:
        file: NIfTI file or folder (zip)
        output_format: 'nifti' or 'dicom'
    
    Returns:
        Segmentation mask file
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Save uploaded file
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / file.filename
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Run inference
        output_path = Path(tmpdir) / f"segmentation.{'nii.gz' if output_format == 'nifti' else 'dcm'}"
        
        try:
            result = pipeline.predict(
                str(input_path),
                output_path=str(output_path),
                output_format=output_format
            )
            
            # Return file
            return FileResponse(
                path=str(output_path),
                filename=output_path.name,
                media_type="application/octet-stream"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

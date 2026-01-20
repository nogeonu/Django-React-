"""
Create 4-channel DICOM test data (4 sequences in separate folders)
"""
import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid
from pathlib import Path
from datetime import datetime

def create_dicom_series_from_nifti(nifti_path, output_dir, series_number, patient_id="ISPY2_213913"):
    """Convert a single NIfTI file to a DICOM series"""
    # Load NIfTI
    nii = nib.load(nifti_path)
    data = nii.get_fdata()
    affine = nii.affine
    
    # Get spacing from affine
    spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate UIDs
    study_uid = "1.2.840.113619.2.55.3.2831161234.123.1234567890.1"  # Fixed for all series
    series_uid = generate_uid()
    frame_of_reference_uid = "1.2.840.113619.2.55.3.2831161234.123.1234567890.2"  # Fixed
    
    # Get number of slices
    num_slices = data.shape[2]
    
    for slice_idx in range(num_slices):
        # Extract slice
        slice_data = data[:, :, slice_idx].astype(np.float32)
        
        # Normalize to uint16
        slice_min = slice_data.min()
        slice_max = slice_data.max()
        if slice_max > slice_min:
            slice_data = ((slice_data - slice_min) / (slice_max - slice_min) * 4095).astype(np.uint16)
        else:
            slice_data = np.zeros_like(slice_data, dtype=np.uint16)
        
        # Calculate position
        position = affine @ np.array([0, 0, slice_idx, 1])
        
        # Create DICOM dataset
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4'
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
        file_meta.ImplementationClassUID = generate_uid()
        
        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
        
        # Patient Information
        ds.PatientName = patient_id
        ds.PatientID = patient_id
        ds.PatientBirthDate = '19700101'
        ds.PatientSex = 'F'
        
        # Study Information
        ds.StudyInstanceUID = study_uid
        ds.StudyDate = datetime.now().strftime('%Y%m%d')
        ds.StudyTime = datetime.now().strftime('%H%M%S')
        ds.StudyID = '1'
        ds.AccessionNumber = 'TEST001'
        
        # Series Information
        ds.SeriesInstanceUID = series_uid
        ds.SeriesNumber = str(series_number)
        ds.SeriesDescription = f'DCE-MRI Sequence {series_number}'
        ds.Modality = 'MR'
        
        # Image Information
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.InstanceNumber = str(slice_idx + 1)
        
        # Image Orientation and Position
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [float(position[0]), float(position[1]), float(position[2])]
        
        # Pixel Spacing
        ds.PixelSpacing = [float(spacing[0]), float(spacing[1])]
        ds.SliceThickness = float(spacing[2])
        
        # Frame of Reference
        ds.FrameOfReferenceUID = frame_of_reference_uid
        
        # Image Pixel Data
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.Rows = slice_data.shape[0]
        ds.Columns = slice_data.shape[1]
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0
        ds.PixelData = slice_data.tobytes()
        
        # Save
        output_file = output_dir / f"slice_{slice_idx:04d}.dcm"
        ds.save_as(output_file, write_like_original=False)
    
    return output_dir

if __name__ == "__main__":
    # Create 4 sequences
    base_dir = Path(r"C:\Users\shrjs\Desktop\MAMA_MIA_ALL_PHASES_TRAINING\MAMA_MIA_DELIVERY_PKG\sample_data\ISPY2_213913_DICOM_4CH")
    
    for seq_idx in range(4):
        nifti_file = rf"C:\Users\shrjs\Desktop\MAMA_MIA_ALL_PHASES_TRAINING\data\images\ISPY2_213913\ispy2_213913_000{seq_idx}.nii.gz"
        output_dir = base_dir / f"seq_{seq_idx}"
        
        print(f"Creating sequence {seq_idx}...")
        create_dicom_series_from_nifti(nifti_file, output_dir, series_number=seq_idx+1)
        print(f"  Done: {output_dir}")
    
    print(f"\nAll 4 sequences created in: {base_dir}")

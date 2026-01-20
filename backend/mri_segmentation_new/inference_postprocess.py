"""
Post-processing Module for Phase 1 Segmentation Inference
Converts model output (probabilities) to final segmentation mask.
"""
import torch
import numpy as np
import logging
from monai.transforms import (
    Compose, Invertd, SaveImaged, AsDiscreted,
    KeepLargestConnectedComponentd, FillHolesd
)

logger = logging.getLogger(__name__)
from scipy import ndimage
import config


def get_postprocess_transforms(threshold=0.5, apply_morphology=True):
    """
    Returns post-processing transform pipeline.
    
    Args:
        threshold: Probability threshold for binarization (default: 0.5)
        apply_morphology: Whether to apply morphological cleaning
    
    Returns:
        Compose: MONAI transform pipeline
    """
    transforms_list = [
        AsDiscreted(keys=["pred"], threshold=threshold),
    ]
    
    if apply_morphology:
        transforms_list.extend([
            KeepLargestConnectedComponentd(keys=["pred"], applied_labels=[1]),
            FillHolesd(keys=["pred"], applied_labels=[1])
        ])
    
    return Compose(transforms_list)


def postprocess_prediction(
    prediction, 
    original_meta_dict=None,
    preprocessed_data=None, # For Invertd
    threshold=0.5,
    apply_morphology=True,
    restore_original_spacing=True
):
    """
    Post-process model prediction to final segmentation mask.
    
    Args:
        prediction: Model output tensor [1, 1, H, W, D] (probabilities after sigmoid)
        original_meta_dict: Metadata from preprocessing for coordinate restoration
        threshold: Binarization threshold
        apply_morphology: Whether to clean up small components
        restore_original_spacing: Whether to resample back to original patient spacing
    
    Returns:
        np.ndarray: Final segmentation mask in original coordinate space
    """
    # Remove batch dimension
    pred = prediction.squeeze(0).squeeze(0)  # [H, W, D]
    
    # Apply threshold
    binary_mask = (pred > threshold).cpu().numpy().astype(np.uint8)
    
    # Morphological cleaning
    if apply_morphology:
        # Keep largest connected component
        labeled, num_features = ndimage.label(binary_mask)
        if num_features > 0:
            sizes = ndimage.sum(binary_mask, labeled, range(1, num_features + 1))
            largest_component = np.argmax(sizes) + 1
            binary_mask = (labeled == largest_component).astype(np.uint8)
        
        # Fill holes
        binary_mask = ndimage.binary_fill_holes(binary_mask).astype(np.uint8)
    
    # Restore to original spacing/orientation if requested
    if restore_original_spacing and preprocessed_data is not None:
        logger.info("  Restoring to original spacing using Invertd...")
        from monai.transforms import Invertd, EnsureChannelFirstd, Compose
        from monai.data import MetaTensor
        
        # Prepare data for inversion
        # We need the original MetaTensor that holds transform information
        input_image = preprocessed_data.get("image")
        
        # Create a dictionary for Invertd
        # Note: binary_mask is numpy [H, W, D], we need [C, H, W, D] for MONAI
        # And convert to Tensor (Invertd expects Tensor/MetaTensor)
        # IMPORTANT: We must wrap it as MetaTensor with the current affine (1.5mm)
        # so Invertd knows the starting point.
        mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0).float()
        
        # If input_image has batch dim, remove it matching pipeline logic
        if len(input_image.shape) == 5: # [B, C, H, W, D]
             input_image = input_image.squeeze(0)
             
        # Create MetaTensor with affine from input_image
        if isinstance(input_image, MetaTensor):
            mask_tensor = MetaTensor(mask_tensor, affine=input_image.affine)
        
        data = dict(preprocessed_data)
        data["image"] = input_image
        data["pred"] = mask_tensor
        
        # Import transforms pipeline
        from inference_preprocess import get_inference_transforms
        transforms = get_inference_transforms()
        
        # Define inverter
        inverter = Invertd(
            keys=["pred"],
            transform=transforms,
            orig_keys=["image"],
            nearest_interp=True,
            to_tensor=True
        )
        
        # Apply inversion
        # Invertd expects a LIST of dicts (batch) or a single dict?
        # MONAI transforms usually handle single dict.
        try:
            result = inverter(data)
            restored = result["pred"]
            
            # restored is [C, H, W, D]
            binary_mask = restored.squeeze(0).numpy().astype(np.uint8)
            logger.info(f"  Restored mask shape: {binary_mask.shape}")
        except Exception as e:
            logger.warning(f"  Invertd failed ({e}). Keeping inference resolution.")
            # Keep original mask without restoration

    elif restore_original_spacing and original_meta_dict is not None:
        # Legacy fallback
        logger.info("  Restoring to original spacing using manual Spacing transform...")
        from monai.transforms import Spacing
        original_spacing = original_meta_dict.get('pixdim', None)
        if original_spacing is not None:
            original_spacing = original_spacing[1:4]
            spacing_transform = Spacing(pixdim=original_spacing, mode="nearest")
            mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0).float()
            restored = spacing_transform(mask_tensor)
            binary_mask = restored.squeeze(0).numpy().astype(np.uint8)
            logger.info(f"  Restored mask shape: {binary_mask.shape}")
    else:
        logger.info(f"  Keeping inference resolution: {binary_mask.shape}")
    
    return binary_mask


def save_segmentation(mask, output_path, reference_meta_dict=None):
    """
    Save segmentation mask as NIfTI file.
    
    Args:
        mask: Binary segmentation mask (numpy array)
        output_path: Path to save the output file
        reference_meta_dict: Optional metadata to preserve affine/header info
    """
    import nibabel as nib
    
    if reference_meta_dict is not None and 'affine' in reference_meta_dict:
        affine = reference_meta_dict['affine']
    else:
        affine = np.eye(4)
    
    nifti_img = nib.Nifti1Image(mask, affine)
    nib.save(nifti_img, output_path)
    print(f"Segmentation saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Post-processing module loaded successfully.")

def save_as_dicom_seg(mask, output_path, reference_dicom_path, prediction_label="Tumor"):
    """
    Save segmentation as DICOM SEG object using highdicom.
    
    Args:
        mask: 3D numpy array (binary mask) [H, W, D]
        output_path: Path to save .dcm file
        reference_dicom_path: Path to folder containing original DICOM series
        prediction_label: Label name
    """
    import pydicom
    import numpy as np
    from highdicom.seg import (
        Segmentation,
        SegmentDescription,
        SegmentAlgorithmTypeValues
    )
    from highdicom.content import (
        PixelMeasuresSequence,
        PlanePositionSequence,
        PlaneOrientationSequence,
        AlgorithmIdentificationSequence
    )
    from highdicom import UID
    from highdicom.sr import CodedConcept
    from pathlib import Path
    
    # 1. Read original DICOM series to get template
    path_obj = Path(reference_dicom_path)
    dicom_files = sorted(path_obj.glob("*.dcm"))
    
    # If no DICOMs in root, check subfolders (e.g., seq_00) generated by the backend
    if not dicom_files:
        subfolders = sorted([d for d in path_obj.iterdir() if d.is_dir() and d.name.startswith("seq_")])
        if subfolders:
            print(f"  No DICOMs in root, using first subfolder as reference: {subfolders[0].name}")
            dicom_files = sorted(subfolders[0].glob("*.dcm"))

    if not dicom_files:
        raise FileNotFoundError(f"No .dcm files found in {reference_dicom_path} or its sequence subfolders")
        
    source_images = [pydicom.dcmread(str(f)) for f in dicom_files]
    
    # 1.1 Meta-data Safety Check: Ensure mandatory tags for highdicom
    # Generate consistent UIDs once for the whole series if missing
    fallback_for_uid = UID()
    fallback_study_uid = UID()
    fallback_series_uid = UID()
    
    # Check if we need to use fallback UIDs
    first_ds = source_images[0]
    use_fallback_series = 'SeriesInstanceUID' not in first_ds
    
    for idx, ds in enumerate(source_images):
        if 'AccessionNumber' not in ds:
            ds.AccessionNumber = ''
        if 'ReferringPhysicianName' not in ds:
            ds.ReferringPhysicianName = ''
        if 'FrameOfReferenceUID' not in ds:
            ds.FrameOfReferenceUID = fallback_for_uid
        if 'StudyInstanceUID' not in ds:
            ds.StudyInstanceUID = fallback_study_uid
        # CRITICAL: All images must have the SAME SeriesInstanceUID
        if use_fallback_series or 'SeriesInstanceUID' not in ds:
            ds.SeriesInstanceUID = fallback_series_uid
        # PixelSpacing is required for spatial measurements
        if 'PixelSpacing' not in ds:
            ds.PixelSpacing = [1.0, 1.0]  # Default 1mm spacing
        # ImageOrientationPatient: standard axial orientation
        if 'ImageOrientationPatient' not in ds:
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        # ImagePositionPatient: use slice index if missing
        if 'ImagePositionPatient' not in ds:
            slice_thickness = float(getattr(ds, 'SliceThickness', 1.0))
            ds.ImagePositionPatient = [0, 0, idx * slice_thickness]
        # SliceLocation for ordering
        if 'SliceLocation' not in ds:
            ds.SliceLocation = idx * float(getattr(ds, 'SliceThickness', 1.0))
            
    # Validate that all images have the same dimensions (required by highdicom)
    first_rows = source_images[0].Rows
    first_cols = source_images[0].Columns
    for idx, ds in enumerate(source_images):
        if ds.Rows != first_rows or ds.Columns != first_cols:
            print(f"Warning: Image {idx} has different dimensions ({ds.Rows}x{ds.Columns}) vs first image ({first_rows}x{first_cols})")
            
    # Sort by ImagePositionPatient (Z-axis) to ensure correct order matches mask
    # We sort by Instance Number as a robust proxy for Z-ordering in standard series
    source_images.sort(key=lambda x: getattr(x, 'InstanceNumber', 0))
    
    # 2. Prepare Mask Data
    # mask is [H, W, D] (RAS from MONAI usually). 
    # highdicom expects [Frames, Rows, Cols] (Z, Y, X).
    # We need to transpose [H, W, D] -> [D, W, H] (Z, Y, X)
    # Note: If MONAI loaded as RAS, and DICOM is LPS, we rely on MONAI's Invertd 
    # having already restored it to the Patient Coordinate System geometry?
    # Actually, `Invertd` output is in the same grid as the input image.
    # So if we simply match the frame order, we just need to align dimensions.
    
    logger.info(f"  Input mask shape: {mask.shape}")
    logger.info(f"  Number of source images: {len(source_images)}")
    logger.info(f"  Source image dimensions: {first_rows}x{first_cols}")
    
    # Transpose [H, W, D] -> [D, H, W]
    mask_frames = mask.transpose(2, 0, 1)
    logger.info(f"  After transpose: {mask_frames.shape}")
    
    # CRITICAL: Ensure mask matches source images in ALL dimensions
    needs_resize = False
    zoom_factors = [1.0, 1.0, 1.0]
    
    # Check depth (number of frames)
    if mask_frames.shape[0] != len(source_images):
        logger.warning(f"  Mask has {mask_frames.shape[0]} frames but {len(source_images)} source images")
        zoom_factors[0] = len(source_images) / mask_frames.shape[0]
        needs_resize = True
    
    # Check height (rows)
    if mask_frames.shape[1] != first_rows:
        logger.warning(f"  Mask height {mask_frames.shape[1]} != source height {first_rows}")
        zoom_factors[1] = first_rows / mask_frames.shape[1]
        needs_resize = True
    
    # Check width (columns)
    if mask_frames.shape[2] != first_cols:
        logger.warning(f"  Mask width {mask_frames.shape[2]} != source width {first_cols}")
        zoom_factors[2] = first_cols / mask_frames.shape[2]
        needs_resize = True
    
    # Apply resizing if needed
    if needs_resize:
        logger.warning(f"  *** RESIZING MASK ***")
        logger.warning(f"  Original mask shape: {mask_frames.shape}")
        logger.warning(f"  Target size: ({len(source_images)}, {first_rows}, {first_cols})")
        logger.warning(f"  Zoom factors: {zoom_factors}")
        from scipy.ndimage import zoom
        # Use float for zoom, then convert back
        mask_frames_float = mask_frames.astype(np.float32)
        mask_frames_resized = zoom(mask_frames_float, zoom_factors, order=0, mode='nearest')
        # Convert back to original dtype (preserve boolean if it was boolean)
        if mask.dtype == bool:
            mask_frames = (mask_frames_resized > 0.5).astype(bool)
        else:
            mask_frames = mask_frames_resized.astype(mask.dtype)
        logger.warning(f"  *** RESIZED MASK SHAPE: {mask_frames.shape} ***")
        logger.warning(f"  Verification: height={mask_frames.shape[1]}, width={mask_frames.shape[2]}, target={first_rows}x{first_cols}")
    
    # Ensure boolean
    mask_frames = mask_frames > 0
    
    # 2.5 SPARSE ENCODING: Only keep non-empty frames
    # Find frames that have at least one positive voxel
    non_empty_indices = []
    non_empty_frames = []
    
    for frame_idx in range(mask_frames.shape[0]):
        frame = mask_frames[frame_idx]
        if np.any(frame):  # Frame has at least one positive voxel
            non_empty_indices.append(frame_idx)
            non_empty_frames.append(frame)
    
    if len(non_empty_frames) == 0:
        logger.warning("  No tumor detected in any frame. Creating empty segmentation.")
        # Create a single empty frame
        non_empty_frames = [np.zeros((first_rows, first_cols), dtype=bool)]
        non_empty_indices = [0]
    
    logger.info(f"  Sparse encoding: {len(non_empty_frames)} non-empty frames out of {mask_frames.shape[0]} total")
    logger.info(f"  Non-empty frame indices: {non_empty_indices[:10]}{'...' if len(non_empty_indices) > 10 else ''}")
    
    # Stack non-empty frames
    mask_frames_sparse = np.stack(non_empty_frames, axis=0)
    
    # Select corresponding source images for non-empty frames
    source_images_sparse = [source_images[idx] for idx in non_empty_indices]
    
    logger.info(f"  Final sparse mask shape: {mask_frames_sparse.shape}")
    logger.info(f"  Number of reference images: {len(source_images_sparse)}")
    
    # 3. Create Segment Description with CodedConcept
    # SCT (SNOMED CT) 코드 직접 생성
    tissue_category = CodedConcept(
        value="85756007",
        scheme_designator="SCT",
        meaning="Tissue"
    )
    neoplasm_type = CodedConcept(
        value="126906006",
        scheme_designator="SCT",
        meaning="Neoplasm"
    )
    ai_family = CodedConcept(
        value="T-D0050",
        scheme_designator="DCM",
        meaning="Artificial Intelligence"
    )
    
    segment_description = SegmentDescription(
        segment_number=1,
        segment_label=prediction_label,
        segmented_property_category=tissue_category,
        segmented_property_type=neoplasm_type,
        algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=AlgorithmIdentificationSequence(
            name="MAMA-MIA-AI",
            version="1.0",
            family=ai_family
        )
    )
    
    # 4. Create Segmentation Object
    from datetime import datetime
    current_date = datetime.now().strftime("%Y%m%d")
    current_time = datetime.now().strftime("%H%M%S")
    
    seg_dataset = Segmentation(
        source_images=source_images_sparse,  # Only non-empty frames
        pixel_array=mask_frames_sparse,      # Only non-empty frames
        segmentation_type="BINARY",
        segment_descriptions=[segment_description],
        series_instance_uid=UID(),
        series_number=1000,
        sop_instance_uid=UID(),
        instance_number=1,
        manufacturer="MAMA-MIA Team",
        manufacturer_model_name="Phase1-Segmentation",
        software_versions="1.0",
        device_serial_number="123456"
    )
    
    # Add missing DICOM metadata
    seg_dataset.SeriesDate = current_date
    seg_dataset.SeriesTime = current_time
    seg_dataset.SeriesDescription = f"AI Segmentation - {prediction_label}"
    seg_dataset.BodyPartExamined = "BREAST"
    seg_dataset.ProtocolName = "MAMA-MIA AI Segmentation"
    
    # 5. Save
    logger.info(f"  Final sparse mask shape before saving: {mask_frames_sparse.shape}")
    logger.info(f"  NumberOfFrames (sparse): {len(source_images_sparse)}")
    logger.info(f"  Frame mapping: {dict(zip(range(len(non_empty_indices)), non_empty_indices))}")
    seg_dataset.save_as(output_path)
    logger.info(f"DICOM SEG saved to: {output_path}")
    
    # Verify saved file
    import pydicom
    saved_ds = pydicom.dcmread(output_path)
    logger.info(f"  Saved DICOM SEG verification:")
    logger.info(f"    NumberOfFrames: {getattr(saved_ds, 'NumberOfFrames', 'N/A')}")
    logger.info(f"    Rows: {saved_ds.Rows}, Columns: {saved_ds.Columns}")
    logger.info(f"    PixelData size: {len(saved_ds.PixelData) if hasattr(saved_ds, 'PixelData') else 'N/A'} bytes")
    
    # Verify saved file
    import pydicom
    saved_ds = pydicom.dcmread(output_path)
    logger.info(f"  Saved DICOM SEG verification:")
    logger.info(f"    NumberOfFrames: {getattr(saved_ds, 'NumberOfFrames', 'N/A')}")
    logger.info(f"    Rows: {saved_ds.Rows}, Columns: {saved_ds.Columns}")
    logger.info(f"    PixelData size: {len(saved_ds.PixelData) if hasattr(saved_ds, 'PixelData') else 'N/A'} bytes")

"""
Post-processing Module for Phase 1 Segmentation Inference
Converts model output (probabilities) to final segmentation mask.
"""
import torch
import numpy as np
from monai.transforms import (
    Compose, Invertd, SaveImaged, AsDiscreted,
    KeepLargestConnectedComponentd, FillHolesd
)
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
            # print("Restored to original geometry using Invertd")
        except Exception as e:
            print(f"Warning: Invertd failed ({e}). Falling back to manual spacing restoration.")
            # Fallback to manual Spacing logic
            if original_meta_dict is not None:
                original_spacing = original_meta_dict.get('pixdim', None)
                if original_spacing is not None:
                    original_spacing = original_spacing[1:4]
                    from monai.transforms import Spacing
                    spacing_transform = Spacing(pixdim=original_spacing, mode="nearest")
                    mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0).float()
                    restored = spacing_transform(mask_tensor)
                    binary_mask = restored.squeeze(0).numpy().astype(np.uint8)

    elif restore_original_spacing and original_meta_dict is not None:
        # Legacy fallback
        from monai.transforms import Spacing
        original_spacing = original_meta_dict.get('pixdim', None)
        if original_spacing is not None:
            original_spacing = original_spacing[1:4]
            spacing_transform = Spacing(pixdim=original_spacing, mode="nearest")
            mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0).float()
            restored = spacing_transform(mask_tensor)
            binary_mask = restored.squeeze(0).numpy().astype(np.uint8)
    
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
    dicom_files = sorted(Path(reference_dicom_path).glob("*.dcm"))
    if not dicom_files:
        raise FileNotFoundError(f"No .dcm files found in {reference_dicom_path}")
        
    source_images = [pydicom.dcmread(str(f)) for f in dicom_files]
    
    # Sort by ImagePositionPatient (Z-axis) to ensure correct order matches mask
    # We sort by Instance Number as a robust proxy for Z-ordering in standard series
    source_images.sort(key=lambda x: x.InstanceNumber)
    
    # 2. Prepare Mask Data
    # mask is [H, W, D] (RAS from MONAI usually). 
    # highdicom expects [Frames, Rows, Cols] (Z, Y, X).
    # We need to transpose [H, W, D] -> [D, W, H] (Z, Y, X)
    # Note: If MONAI loaded as RAS, and DICOM is LPS, we rely on MONAI's Invertd 
    # having already restored it to the Patient Coordinate System geometry?
    # Actually, `Invertd` output is in the same grid as the input image.
    # So if we simply match the frame order, we just need to align dimensions.
    
    # Transpose [H, W, D] -> [D, H, W]
    mask_frames = mask.transpose(2, 0, 1)
    
    # Ensure boolean
    mask_frames = mask_frames > 0
    
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
    seg_dataset = Segmentation(
        source_images=source_images,
        pixel_array=mask_frames,
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
    
    # 5. Save
    seg_dataset.save_as(output_path)
    print(f"DICOM SEG saved to: {output_path}")

"""
Post-processing Module for Phase 1 Segmentation Inference
Converts model output (probabilities) to final segmentation mask.
"""
import torch
import numpy as np
import logging
import sys
from monai.transforms import (
    Compose, Invertd, SaveImaged, AsDiscreted,
    KeepLargestConnectedComponentd, FillHolesd
)
from scipy import ndimage
import config

# Logger 설정 (Django/Gunicorn에서 journal에 기록되도록)
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


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
        
        # 🔍 디버깅: 조원님 분석 확인
        logger.debug("[DEBUG] Invertd 복원 시작")
        logger.debug(f"  - binary_mask shape (전처리 후): {binary_mask.shape}")
        logger.debug(f"  - input_image type: {type(input_image)}")
        logger.debug(f"  - input_image is MetaTensor: {isinstance(input_image, MetaTensor)}")
        if isinstance(input_image, MetaTensor):
            logger.debug(f"  - input_image.affine exists: {hasattr(input_image, 'affine')}")
            if hasattr(input_image, 'affine'):
                logger.debug(f"  - input_image.affine shape: {input_image.affine.shape if input_image.affine is not None else None}")
            if hasattr(input_image, 'meta'):
                logger.debug(f"  - input_image.meta keys: {list(input_image.meta.keys())[:10] if hasattr(input_image, 'meta') else None}")
        logger.debug(f"  - original_meta_dict: {original_meta_dict is not None}")
        if original_meta_dict:
            logger.debug(f"  - original_meta_dict keys: {list(original_meta_dict.keys())[:10]}")
            logger.debug(f"  - original_meta_dict spacing: {original_meta_dict.get('spacing', 'NOT FOUND')}")
            logger.debug(f"  - original_meta_dict pixdim: {original_meta_dict.get('pixdim', 'NOT FOUND')}")
            logger.debug(f"  - original_meta_dict spatial_shape: {original_meta_dict.get('spatial_shape', 'NOT FOUND')}")
        
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
            logger.debug(f"  - mask_tensor created as MetaTensor with affine")
        else:
            logger.warning(f"  - ⚠️ WARNING: input_image is NOT MetaTensor! Type: {type(input_image)}")
        
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
            binary_mask_after_invertd = restored.squeeze(0).numpy().astype(np.uint8)
            logger.debug(f"  - Invertd 실행 완료 (예외 없음)")
            logger.debug(f"  - binary_mask shape (Invertd 후): {binary_mask_after_invertd.shape}")
            
            # 🔍 검증: Invertd가 제대로 복원했는지 확인
            # 원본 크기와 비교 (original_meta_dict의 spatial_shape 확인)
            if original_meta_dict and 'spatial_shape' in original_meta_dict:
                original_shape = original_meta_dict['spatial_shape']
                restored_shape = binary_mask_after_invertd.shape
                logger.debug(f"  - 원본 크기 (meta_dict): {original_shape}")
                logger.debug(f"  - 복원 후 크기: {restored_shape}")
                
                if restored_shape != tuple(original_shape):
                    logger.warning(f"  - ⚠️ WARNING: Invertd가 제대로 복원하지 못함!")
                    logger.warning(f"  - 차원 불일치 감지 → Fallback 사용")
                    # Fallback 사용
                    binary_mask = binary_mask  # 원래 크기 유지
                    use_fallback = True
                else:
                    logger.debug(f"  - ✅ Invertd 복원 성공!")
                    binary_mask = binary_mask_after_invertd
                    use_fallback = False
            else:
                logger.warning(f"  - ⚠️ WARNING: original_meta_dict에 spatial_shape 없음")
                logger.warning(f"  - Invertd 결과 사용 (검증 불가)")
                binary_mask = binary_mask_after_invertd
                use_fallback = False
            
            # Fallback이 필요한 경우
            if use_fallback:
                logger.info(f"  - Fallback으로 수동 복원 시도...")
                # 조원님 추천: pixdim 우선, 없으면 spacing 사용
                original_spacing = None
                if original_meta_dict:
                    # pixdim 우선 확인 (NIfTI)
                    original_spacing = original_meta_dict.get('pixdim', None)
                    if original_spacing is None:
                        # spacing 사용 (DICOM)
                        original_spacing = original_meta_dict.get('spacing', None)
                        if original_spacing is not None:
                            logger.debug(f"  - spacing 키 사용: {original_spacing}")
                
                if original_spacing is not None:
                    from monai.transforms import Spacing
                    # spacing 값 추출 (리스트/튜플/텐서 등 다양한 형태 처리)
                    if hasattr(original_spacing, 'tolist'):
                        spacing_values = original_spacing.tolist()
                    elif hasattr(original_spacing, '__iter__') and not isinstance(original_spacing, str):
                        spacing_values = list(original_spacing)
                    else:
                        spacing_values = [original_spacing]
                    
                    # 앞의 1 제거 (pixdim 형태인 경우: [1, x, y, z])
                    if len(spacing_values) == 4:
                        spacing_values = spacing_values[1:]
                    
                    logger.debug(f"  - 최종 spacing 값: {spacing_values}")
                    spacing_transform = Spacing(pixdim=spacing_values, mode="nearest")
                    mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0).float()
                    restored = spacing_transform(mask_tensor)
                    binary_mask = restored.squeeze(0).numpy().astype(np.uint8)
                    logger.info(f"  - ✅ Fallback 복원 성공: {binary_mask.shape}")
                else:
                    logger.error(f"  - ❌ spacing/pixdim 모두 없음 - 복원 불가")
        except Exception as e:
            logger.error(f"  - ❌ Invertd 예외 발생: {e}")
            logger.info(f"  - Fallback으로 수동 복원 시도...")
            # Fallback to manual Spacing logic
            # 조원님 추천: pixdim 우선, 없으면 spacing 사용
            original_spacing = None
            if original_meta_dict is not None:
                # pixdim 우선 확인 (NIfTI)
                original_spacing = original_meta_dict.get('pixdim', None)
                if original_spacing is None:
                    # spacing 사용 (DICOM)
                    original_spacing = original_meta_dict.get('spacing', None)
                    if original_spacing is not None:
                        logger.debug(f"  - spacing 키 사용: {original_spacing}")
                
                if original_spacing is not None:
                    from monai.transforms import Spacing
                    # spacing 값 추출 (리스트/튜플/텐서 등 다양한 형태 처리)
                    if hasattr(original_spacing, 'tolist'):
                        spacing_values = original_spacing.tolist()
                    elif hasattr(original_spacing, '__iter__') and not isinstance(original_spacing, str):
                        spacing_values = list(original_spacing)
                    else:
                        spacing_values = [original_spacing]
                    
                    # 앞의 1 제거 (pixdim 형태인 경우: [1, x, y, z])
                    if len(spacing_values) == 4:
                        spacing_values = spacing_values[1:]
                    
                    logger.debug(f"  - 최종 spacing 값: {spacing_values}")
                    spacing_transform = Spacing(pixdim=spacing_values, mode="nearest")
                    mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0).float()
                    restored = spacing_transform(mask_tensor)
                    binary_mask = restored.squeeze(0).numpy().astype(np.uint8)
                    logger.info(f"  - ✅ Fallback 복원 성공: {binary_mask.shape}")
                else:
                    logger.error(f"  - ❌ spacing/pixdim 모두 없음 - 복원 불가")

    elif restore_original_spacing and original_meta_dict is not None:
        # Legacy fallback
        from monai.transforms import Spacing
        # 조원님 추천: pixdim 우선, 없으면 spacing 사용
        original_spacing = None
        original_spacing = original_meta_dict.get('pixdim', None)
        if original_spacing is None:
            original_spacing = original_meta_dict.get('spacing', None)
        
        if original_spacing is not None:
            # spacing 값 추출 (리스트/튜플/텐서 등 다양한 형태 처리)
            if hasattr(original_spacing, 'tolist'):
                spacing_values = original_spacing.tolist()
            elif hasattr(original_spacing, '__iter__') and not isinstance(original_spacing, str):
                spacing_values = list(original_spacing)
            else:
                spacing_values = [original_spacing]
            
            # 앞의 1 제거 (pixdim 형태인 경우: [1, x, y, z])
            if len(spacing_values) == 4:
                spacing_values = spacing_values[1:]
            
            spacing_transform = Spacing(pixdim=spacing_values, mode="nearest")
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
    from pydicom.sr.codedict import codes
    from pathlib import Path
    
    # 1. Read original DICOM series to get template
    dicom_files = sorted(Path(reference_dicom_path).glob("*.dcm"))
    
    # If no DICOM files in root, check subdirectories (e.g., seq_0, seq_1, ...)
    if not dicom_files:
        subdirs = sorted([d for d in Path(reference_dicom_path).iterdir() if d.is_dir()])
        if subdirs:
            # Use first sequence folder as reference (all sequences share same geometry)
            dicom_files = sorted(subdirs[0].glob("*.dcm"))
            if not dicom_files:
                raise FileNotFoundError(f"No .dcm files found in {reference_dicom_path} or subdirectories")
        else:
            raise FileNotFoundError(f"No .dcm files found in {reference_dicom_path}")
        
    source_images = [pydicom.dcmread(str(f)) for f in dicom_files]
    
    # Sort by ImagePositionPatient (spatial Z-axis) to match MONAI's volume order
    # MONAI LoadImage sorts files spatially. InstanceNumber might be inconsistent or reversed.
    def get_z_position(ds):
        # Calculate projection onto slice normal to robustly handle tilted acquisitions
        iop = ds.ImageOrientationPatient
        # Row cosine (x) and Column cosine (y)
        row_cos = np.array(iop[:3])
        col_cos = np.array(iop[3:])
        # Slice normal (z) = cross product
        slice_norm = np.cross(row_cos, col_cos)
        # Position
        pos = np.array(ds.ImagePositionPatient)
        # Projection
        return np.dot(pos, slice_norm)

    source_images.sort(key=get_z_position)
    
    # 2. Prepare Mask Data
    # CRITICAL: highdicom.Segmentation expects:
    #   - pixel_array shape: (Frames, Rows, Columns) where Frames = len(source_images)
    #   - Each frame corresponds 1:1 with source_images[i]
    
    # Input mask is [H, W, D] from MONAI (after Invertd restoration)
    # We need [D, H, W] for highdicom (Frames=D, Rows=H, Cols=W)
    
    print(f"  Input mask shape: {mask.shape}")
    print(f"  Source images count: {len(source_images)}")
    
    # Transpose to [D, H, W]
    mask_frames = mask.transpose(2, 0, 1)
    print(f"  Transposed mask shape: {mask_frames.shape}")
    
    # Verify dimensions match
    # Invertd가 정상 작동했다면 차원이 일치해야 함
    # 불일치 시 명확한 에러 메시지로 문제 알림 (안전장치)
    if mask_frames.shape[0] != len(source_images):
        raise ValueError(
            f"Dimension mismatch: mask has {mask_frames.shape[0]} frames "
            f"but source_images has {len(source_images)} images. "
            f"Original mask shape: {mask.shape}. "
            f"This indicates Invertd failed to restore original spacing. "
            f"Please check restore_original_spacing=True and Invertd transform."
        )
    
    # Ensure boolean type for BINARY segmentation
    mask_frames = mask_frames > 0
    print(f"  Non-empty frames: {np.sum(np.any(mask_frames, axis=(1,2)))}")
    
    # 3. Create Segment Description
    segment_description = SegmentDescription(
        segment_number=1,
        segment_label=prediction_label,
        segmented_property_category=codes.SCT.Tissue,
        segmented_property_type=codes.SCT.Neoplasm,
        algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=AlgorithmIdentificationSequence(
            name="MAMA-MIA-AI",
            version="1.0",
            family=codes.DCM.ArtificialIntelligence
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

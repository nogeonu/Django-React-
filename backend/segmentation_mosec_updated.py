#!/usr/bin/env python3
"""
Mosec 기반 MRI 세그멘테이션 서버 (GCS 통합 버전)
모델 입력: [4, 96, 96, 96] (4 channels, 96 depth, 96 height, 96 width)
"""
import os
import io
import base64
import logging
import json
import numpy as np
import torch
from monai.inferers import sliding_window_inference
import pydicom
from PIL import Image
from scipy import ndimage
from datetime import datetime
from pydicom.uid import generate_uid
from pydicom.dataset import Dataset, FileDataset
import requests
import tempfile

from mosec import Server, Worker, ValidationError
from monai.networks.nets import SwinUNETR

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수
MODEL_PATH = '/home/shrjsdn908/models/mri_models/Phase1_Segmentation_best.pth'
ORTHANC_URL = 'http://localhost:8042'
ORTHANC_USER = 'admin'
ORTHANC_PASSWORD = 'admin123'


def dicom_to_numpy(dicom_bytes):
    """DICOM 바이트를 numpy 배열로 변환"""
    dicom = pydicom.dcmread(io.BytesIO(dicom_bytes))
    pixel_array = dicom.pixel_array.astype(np.float32)
    
    # 정규화
    if pixel_array.max() > pixel_array.min():
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
    
    return pixel_array, dicom


def create_4d_input_from_sequences(sequences_3d):
    """4개 시퀀스의 3D 볼륨을 [4, 96, 96, 96]로 변환
    
    Args:
        sequences_3d: list of 4 numpy arrays, 각각 [D, H, W] 형태 (D=96, H=256, W=256)
    
    Returns:
        volume_4d: [4, 96, 96, 96] numpy array
    """
    from scipy.ndimage import zoom
    target_spatial = 96
    target_depth = 96
    
    resized_sequences = []
    for seq_3d in sequences_3d:
        d, h, w = seq_3d.shape
        zoom_factors = (target_depth / d, target_spatial / h, target_spatial / w)
        resized = zoom(seq_3d, zoom_factors, order=1)
        resized_sequences.append(resized)
    
    # [4, 96, 96, 96]
    volume_4d = np.stack(resized_sequences, axis=0)
    
    logger.info(f"✅ 3D 볼륨 생성 완료: {volume_4d.shape} (4 channels, 96 depth, 96x96)")
    return volume_4d


def create_mock_4d_input(slice_2d):
    """단일 2D 슬라이스를 4D MRI 입력으로 변환 (fallback)"""
    mock_3d = np.stack([slice_2d] * 96, axis=0)  # [96, H, W]
    return create_4d_input_from_sequences([mock_3d] * 4)


def postprocess_mask(mask):
    """세그멘테이션 마스크 후처리"""
    mask_filled = ndimage.binary_fill_holes(mask)
    labeled, num_features = ndimage.label(mask_filled)
    if num_features > 0:
        sizes = ndimage.sum(mask_filled, labeled, range(1, num_features + 1))
        max_label = np.argmax(sizes) + 1
        mask_cleaned = (labeled == max_label).astype(np.uint8)
    else:
        mask_cleaned = mask_filled.astype(np.uint8)
    return mask_cleaned


def create_dicom_seg(original_dicom, mask_array, seg_series_uid, instance_number, original_series_id):
    """세그멘테이션 마스크를 DICOM SEG 파일로 변환"""
    file_meta = Dataset()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7'
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    # 필수 DICOM 태그
    ds.PatientName = getattr(original_dicom, 'PatientName', 'Anonymous')
    ds.PatientID = getattr(original_dicom, 'PatientID', 'Unknown')
    ds.StudyInstanceUID = getattr(original_dicom, 'StudyInstanceUID', generate_uid())
    ds.SeriesInstanceUID = seg_series_uid
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.Modality = 'SEG'
    ds.SeriesDescription = f'AI Segmentation of {original_series_id}'
    ds.InstanceNumber = instance_number
    ds.SeriesNumber = 9999
    ds.StudyDate = datetime.now().strftime('%Y%m%d')
    ds.StudyTime = datetime.now().strftime('%H%M%S')
    ds.ContentDate = ds.StudyDate
    ds.ContentTime = ds.StudyTime
    
    # 이미지 데이터
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.Rows, ds.Columns = mask_array.shape
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = mask_array.tobytes()
    
    return ds


def download_from_gcs(gcs_url):
    """GCS URL에서 데이터 다운로드"""
    try:
        from google.cloud import storage
        
        if not gcs_url.startswith('gs://'):
            raise ValueError(f"Invalid GCS URL: {gcs_url}")
        
        parts = gcs_url[5:].split('/', 1)
        bucket_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else ''
        
        logger.info(f"📥 GCS 다운로드: bucket={bucket_name}, blob={blob_name}")
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        json_data_str = blob.download_as_text()
        json_data = json.loads(json_data_str)
        
        logger.info(f"✅ GCS 다운로드 완료: {len(json_data_str) / (1024**2):.2f} MB")
        return json_data
        
    except Exception as e:
        logger.error(f"❌ GCS 다운로드 실패: {e}", exc_info=True)
        raise


def upload_to_orthanc(dicom_dataset):
    """DICOM 데이터셋을 Orthanc에 업로드"""
    try:
        buffer = io.BytesIO()
        dicom_dataset.save_as(buffer, write_like_original=False)
        buffer.seek(0)
        
        response = requests.post(
            f"{ORTHANC_URL}/instances",
            auth=(ORTHANC_USER, ORTHANC_PASSWORD),
            data=buffer.getvalue(),
            headers={'Content-Type': 'application/dicom'}
        )
        
        response.raise_for_status()
        result = response.json()
        instance_id = result.get('ID')
        
        logger.info(f"✅ Orthanc 업로드 완료: {instance_id}")
        return instance_id
        
    except Exception as e:
        logger.error(f"❌ Orthanc 업로드 실패: {e}", exc_info=True)
        raise


class SegmentationWorker(Worker):
    """Mosec Worker for MRI Segmentation"""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🔧 Worker 초기화: device={self.device}")
    
    def deserialize(self, data: bytes) -> dict:
        """요청 데이터 역직렬화 (GCS URL 지원)"""
        try:
            # JSON 파싱
            json_str = data.decode('utf-8')
            json_data = json.loads(json_str)
            
            # GCS URL 처리
            if "gcs_url" in json_data:
                logger.info(f"📥 GCS URL 감지: {json_data['gcs_url']}")
                gcs_data = download_from_gcs(json_data["gcs_url"])
                # 메타데이터 병합
                gcs_data.update({
                    "seg_series_uid": json_data.get("seg_series_uid"),
                    "original_series_id": json_data.get("original_series_id")
                })
                return gcs_data
            
            # 기존 로직: sequences_3d 또는 dicom_data
            if "sequences_3d" in json_data or "sequences" in json_data:
                logger.info("📥 4-channel JSON 입력 감지")
                return json_data
                
        except json.JSONDecodeError:
            # Raw bytes (단일 DICOM)
            logger.info("📥 Raw DICOM bytes 입력 감지")
            return {"dicom_bytes": data}
        except Exception as e:
            logger.error(f"역직렬화 실패: {e}", exc_info=True)
            return {"error": str(e)}
        
        logger.warning("알 수 없는 데이터 형식")
        return {}

    def forward(self, data: dict) -> dict:
        """세그멘테이션 추론"""
        try:
            # 모델 로드
            if self.model is None:
                logger.info(f"🔄 세그멘테이션 모델 로딩 중: {MODEL_PATH}")
                
                self.model = SwinUNETR(
                    spatial_dims=3,
                    in_channels=4,
                    out_channels=1,
                    feature_size=24,
                    use_checkpoint=False,
                )
                
                checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
                
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # 키 이름 변환
                new_state_dict = {}
                for key, value in state_dict.items():
                    if 'lora_A' in key or 'lora_B' in key:
                        continue
                    new_key = key.replace('model.base_model.model.', '')
                    new_key = new_key.replace('.base_layer', '')
                    new_state_dict[new_key] = value
                
                self.model.load_state_dict(new_state_dict, strict=False)
                self.model = self.model.to(self.device)
                self.model.eval()
                logger.info("✅ 세그멘테이션 모델 로드 완료")
            
            # DICOM 변환
            slice_2d = None
            original_dicom = None
            
            if "sequences_3d" in data and len(data["sequences_3d"]) == 4:
                # 4-channel 3D DCE-MRI 모드
                logger.info("📊 4-channel 3D DCE-MRI 입력 감지 (96 slices per sequence)")
                sequences_3d = []
                
                for seq_idx, seq_slices_b64 in enumerate(data["sequences_3d"]):
                    slices_2d = []
                    for slice_idx, slice_b64 in enumerate(seq_slices_b64):
                        slice_bytes = base64.b64decode(slice_b64)
                        slice_2d_temp, dicom = dicom_to_numpy(slice_bytes)
                        slices_2d.append(slice_2d_temp)
                        if seq_idx == 0 and slice_idx == 48:  # 중앙 슬라이스
                            original_dicom = dicom
                            slice_2d = slice_2d_temp
                    
                    # [96, H, W] 형태로 스택
                    seq_volume = np.stack(slices_2d, axis=0)
                    sequences_3d.append(seq_volume)
                
                logger.info(f"✅ 3D 볼륨 로드 완료: 4 sequences × {len(seq_slices_b64)} slices")
                
                # 4D 입력 생성: [4, 96, 96, 96]
                volume_4d = create_4d_input_from_sequences(sequences_3d)
                logger.info(f"✅ 4채널 3D 입력 생성 완료: {volume_4d.shape}")
            elif "dicom_data" in data:
                # 단일 이미지 모드 (JSON with base64)
                logger.info("📊 단일 이미지 입력 감지 (JSON)")
                dicom_bytes = base64.b64decode(data["dicom_data"])
                slice_2d, original_dicom = dicom_to_numpy(dicom_bytes)
                volume_4d = create_mock_4d_input(slice_2d)
            elif "dicom_bytes" in data:
                # 단일 이미지 모드 (raw bytes - fallback)
                logger.info("📊 단일 이미지 입력 감지 (raw bytes)")
                dicom_bytes = data["dicom_bytes"]
                slice_2d, original_dicom = dicom_to_numpy(dicom_bytes)
                volume_4d = create_mock_4d_input(slice_2d)
            else:
                raise ValueError(f"지원하지 않는 데이터 형식입니다. Keys: {list(data.keys())}")
            
            input_tensor = torch.from_numpy(volume_4d).unsqueeze(0).float().to(self.device)
            logger.info(f"📊 Input shape: {input_tensor.shape}")
            
            # 추론 with sliding_window_inference
            with torch.no_grad():
                output = sliding_window_inference(
                    inputs=input_tensor,              # [1, 4, 96, 96, 96]
                    roi_size=(96, 96, 96),
                    sw_batch_size=1,
                    predictor=self.model,
                    overlap=0.5
                )
                # output: [1, 1, 96, 96, 96] (out_channels=1이므로)
                pred_prob = torch.sigmoid(output).squeeze(0).squeeze(0).cpu().numpy()  # [96, 96, 96]
                pred_mask = (pred_prob > 0.5).astype(np.uint8)
                logger.info(f"📊 Output shape: {pred_mask.shape}")
            
            # 중앙 슬라이스 추출
            center_idx = pred_mask.shape[0] // 2
            center_slice = pred_mask[center_idx, :, :]
            logger.info(f"📍 중앙 슬라이스 추출: index {center_idx}/{pred_mask.shape[0]}")
            
            # 후처리
            mask_cleaned = postprocess_mask(center_slice)
            
            # 원본 크기로 리사이즈
            from scipy.ndimage import zoom
            h, w = slice_2d.shape if slice_2d is not None else (256, 256)
            zoom_factors = (h / 96, w / 96)
            mask_resized = zoom(mask_cleaned, zoom_factors, order=0)
            
            # Base64 인코딩
            mask_pil = Image.fromarray((mask_resized * 255).astype(np.uint8), mode='L')
            mask_bytes = io.BytesIO()
            mask_pil.save(mask_bytes, format='PNG')
            mask_base64 = base64.b64encode(mask_bytes.getvalue()).decode('utf-8')
            
            # 통계
            tumor_pixels = int(np.sum(mask_resized))
            total_pixels = int(mask_resized.size)
            tumor_ratio = float(tumor_pixels / total_pixels)
            
            # Orthanc에 저장
            seg_instance_id = None
            try:
                seg_series_uid = data.get('seg_series_uid')
                instance_number = data.get('instance_number', 1)
                original_series_id = data.get('original_series_id', 'unknown')
                
                if original_dicom and seg_series_uid:
                    dicom_seg = create_dicom_seg(original_dicom, mask_resized, seg_series_uid, instance_number, original_series_id)
                    seg_instance_id = upload_to_orthanc(dicom_seg)
                    logger.info(f"✅ 세그멘테이션 결과 Orthanc 저장 완료: {seg_instance_id}")
            except Exception as e:
                logger.error(f"⚠️ Orthanc 저장 실패 (계속 진행): {e}")
            
            return {
                "success": True,
                "segmentation_mask_base64": mask_base64,
                "tumor_pixel_count": tumor_pixels,
                "total_pixel_count": total_pixels,
                "tumor_ratio_percent": tumor_ratio * 100,
                "image_size": [int(w), int(h)],
                "seg_instance_id": seg_instance_id,
                "saved_to_orthanc": seg_instance_id is not None
            }
            
        except Exception as e:
            logger.error(f"세그멘테이션 오류: {e}")
            import traceback
            traceback.print_exc()
            raise ValidationError(f"Segmentation failed: {e}")
    
    def serialize(self, data: dict) -> bytes:
        """응답 데이터 직렬화"""
        return json.dumps(data).encode('utf-8')


if __name__ == "__main__":
    server = Server()
    server.append_worker(
        SegmentationWorker,
        num=1,
        max_batch_size=1,
        timeout=600,
    )
    # CLI arguments are automatically parsed by Mosec
    # --max-body-size 옵션은 CLI에서 설정
    server.run()

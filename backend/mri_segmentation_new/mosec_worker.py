"""
Mosec Worker for MRI Segmentation
새로운 추론 파이프라인을 사용하는 Mosec 워커
"""
import mosec
import torch
import numpy as np
from pathlib import Path
import tempfile
import logging
from typing import Dict, Any
import json
import base64

from inference_pipeline import SegmentationInferencePipeline
from dicom_nifti_converter import dicom_series_to_nifti, nifti_to_dicom_seg
import config

logger = logging.getLogger(__name__)

# 모델 경로
MODEL_PATH = Path(__file__).parent / "checkpoints" / "best_model.pth"
if not MODEL_PATH.exists():
    MODEL_PATH = Path(__file__).parent / "best_model.pth"


class MRISegmentationWorker(mosec.Worker):
    """
    MRI 세그멘테이션 Mosec 워커
    4채널 DCE-MRI DICOM 파일을 받아서 세그멘테이션 수행
    """
    
    def __init__(self):
        super().__init__()
        
        # 디바이스 설정
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # 추론 파이프라인 초기화
        logger.info(f"Loading model from: {MODEL_PATH}")
        self.pipeline = SegmentationInferencePipeline(
            model_path=str(MODEL_PATH),
            device=self.device,
            threshold=0.5
        )
        logger.info("Model loaded successfully!")
    
    def deserialize(self, data: bytes) -> Dict[str, Any]:
        """
        요청 데이터 역직렬화
        JSON 형식: {
            "dicom_sequences": [base64_encoded_dicom1, base64_encoded_dicom2, ...],
            "orthanc_url": "http://localhost:8042",
            "orthanc_auth": ["admin", "admin123"],
            "orthanc_instance_ids": [[id1, id2, ...], [id3, id4, ...], ...],
            "seg_series_uid": "...",
            "original_series_id": "..."
        }
        """
        try:
            request_data = json.loads(data.decode('utf-8'))
            
            # DICOM 파일 디코딩
            if "dicom_sequences" in request_data:
                dicom_sequences = []
                for seq_base64 in request_data["dicom_sequences"]:
                    dicom_bytes = base64.b64decode(seq_base64)
                    dicom_sequences.append(dicom_bytes)
                request_data["dicom_sequences"] = dicom_sequences
            
            return request_data
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            raise
    
    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        세그멘테이션 추론 수행
        """
        try:
            # 1. DICOM → NIfTI 변환
            logger.info("Converting DICOM to NIfTI...")
            dicom_sequences = data.get("dicom_sequences", [])
            
            if len(dicom_sequences) != 4:
                raise ValueError(f"4개 시퀀스가 필요합니다. 현재 {len(dicom_sequences)}개 제공됨")
            
            # 임시 NIfTI 파일 생성
            with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_nifti:
                nifti_path = tmp_nifti.name
            
            nifti_path, metadata = dicom_series_to_nifti(
                dicom_sequences,
                output_path=nifti_path
            )
            
            logger.info(f"NIfTI 변환 완료: {nifti_path}, Shape: {metadata['shape']}")
            
            # 2. 세그멘테이션 추론
            logger.info("Running segmentation inference...")
            with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_seg:
                seg_nifti_path = tmp_seg.name
            
            result = self.pipeline.predict(
                image_path=nifti_path,
                output_path=seg_nifti_path
            )
            
            logger.info(f"Segmentation 완료: Tumor detected={result['tumor_detected']}, Volume={result['tumor_volume_voxels']} voxels")
            
            # 3. NIfTI → DICOM SEG 변환 (필요시)
            # 현재는 NIfTI 결과를 반환하고, Django에서 DICOM SEG 변환 수행
            
            # 임시 파일 정리
            try:
                os.unlink(nifti_path)
            except:
                pass
            
            return {
                "success": True,
                "segmentation_nifti_path": seg_nifti_path,
                "tumor_detected": result["tumor_detected"],
                "tumor_volume_voxels": result["tumor_volume_voxels"],
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Segmentation error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def serialize(self, data: Dict[str, Any]) -> bytes:
        """
        응답 데이터 직렬화
        """
        return json.dumps(data).encode('utf-8')


if __name__ == "__main__":
    # Mosec 서버 시작
    worker = MRISegmentationWorker()
    server = mosec.Server()
    server.append_worker(worker, num=1, max_batch_size=1)
    server.run()

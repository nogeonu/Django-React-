#!/usr/bin/env python3
"""
FastAPI 세그멘테이션 서버 (Mosec 우회)
- Django의 큰 요청을 받아서 직접 세그멘테이션 수행
- body size limit: 500MB
- 포트: 5007
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import logging
import gzip
import json
import sys
import os

# segmentation_mosec.py의 경로 추가
sys.path.insert(0, os.path.expanduser('~'))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="FastAPI Segmentation Server")

# SegmentationWorker import 및 초기화
try:
    from segmentation_mosec import SegmentationWorker
    worker = SegmentationWorker()
    logger.info("✅ SegmentationWorker 초기화 완료")
except Exception as e:
    logger.error(f"❌ SegmentationWorker 초기화 실패: {e}", exc_info=True)
    worker = None

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "healthy" if worker else "unhealthy",
        "service": "FastAPI Segmentation Server",
        "max_body_size": "500MB"
    }

@app.post("/inference")
async def inference(request: Request):
    """
    Django → FastAPI 직접 세그멘테이션
    """
    if worker is None:
        raise HTTPException(status_code=503, detail="SegmentationWorker not initialized")
    
    try:
        # 요청 받기
        content_type = request.headers.get("content-type", "")
        content_encoding = request.headers.get("content-encoding", "")
        
        logger.info(f"📥 요청 받음: Content-Type={content_type}, Encoding={content_encoding}")
        
        # Body 읽기 (최대 500MB)
        body = await request.body()
        body_size_mb = len(body) / (1024**2)
        
        logger.info(f"📦 요청 크기: {body_size_mb:.2f} MB")
        
        if body_size_mb > 500:
            raise HTTPException(
                status_code=413,
                detail=f"Payload too large: {body_size_mb:.2f} MB (max: 500 MB)"
            )
        
        # gzip 압축 해제 (필요한 경우)
        if content_encoding == "gzip" or body[:2] == b'\x1f\x8b':
            logger.info("🔓 gzip 압축 해제 중...")
            body = gzip.decompress(body)
            decompressed_size_mb = len(body) / (1024**2)
            logger.info(f"✅ 압축 해제 완료: {body_size_mb:.2f} MB → {decompressed_size_mb:.2f} MB")
        
        # JSON 파싱
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON 파싱 오류: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
        
        logger.info(f"🔄 세그멘테이션 시작...")
        
        # SegmentationWorker의 forward 메서드 호출
        result = worker.forward(data)
        
        logger.info(f"✅ 세그멘테이션 완료: success={result.get('success')}")
        
        return JSONResponse(content=result)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 세그멘테이션 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "proxy": "healthy",
        "worker": "healthy" if worker else "unhealthy"
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 FastAPI 세그멘테이션 서버 시작 중...")
    logger.info(f"   포트: 5007")
    logger.info(f"   Max body size: 500MB")
    
    # uvicorn.run()으로 시작
    # limit_request_body는 uvicorn.run()에서 지원되지 않으므로 제거
    # 대신 FastAPI의 Request.body()가 자동으로 큰 요청을 처리
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5007,
        timeout_keep_alive=600,
    )

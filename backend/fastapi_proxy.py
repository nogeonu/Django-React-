#!/usr/bin/env python3
"""
FastAPI 프록시 서버 for Mosec
- Django의 큰 요청을 받아서 Mosec으로 전달
- body size limit: 500MB
- 포트: 5007
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import httpx
import logging
import gzip

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mosec Proxy Server")

# Mosec 서버 주소
MOSEC_URL = "http://localhost:5006"

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "healthy",
        "service": "FastAPI Proxy for Mosec",
        "mosec_url": MOSEC_URL,
        "max_body_size": "500MB"
    }

@app.post("/inference")
async def proxy_inference(request: Request):
    """
    Django → FastAPI → Mosec 프록시
    """
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
        
        # Mosec으로 전달 (로컬 HTTP)
        logger.info(f"🔄 Mosec으로 요청 전달: {MOSEC_URL}/inference")
        
        async with httpx.AsyncClient(timeout=600.0) as client:
            mosec_response = await client.post(
                f"{MOSEC_URL}/inference",
                content=body,
                headers={
                    "Content-Type": "application/json",
                }
            )
            
            mosec_response.raise_for_status()
            result = mosec_response.json()
            
            logger.info(f"✅ Mosec 응답 받음: success={result.get('success')}")
            
            return JSONResponse(content=result)
            
    except httpx.TimeoutException:
        logger.error("⏱️ Mosec 타임아웃")
        raise HTTPException(status_code=504, detail="Mosec timeout")
    except httpx.HTTPStatusError as e:
        logger.error(f"❌ Mosec HTTP 오류: {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Mosec error: {e.response.text}"
        )
    except Exception as e:
        logger.error(f"❌ 프록시 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Mosec 상태 확인
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{MOSEC_URL}/")
            mosec_healthy = response.status_code == 200
    except:
        mosec_healthy = False
    
    return {
        "proxy": "healthy",
        "mosec": "healthy" if mosec_healthy else "unhealthy"
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 FastAPI 프록시 서버 시작 중...")
    logger.info(f"   포트: 5007")
    logger.info(f"   Max body size: 500MB")
    logger.info(f"   Mosec URL: {MOSEC_URL}")
    
    # uvicorn.run()으로 시작 (CLI 옵션은 제외)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5007,
        timeout_keep_alive=600,
    )

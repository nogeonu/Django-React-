@echo off
REM 병리 이미지 추론 워커 시작 스크립트 (Windows)
REM 교육원 컴퓨터에서 실행

echo ========================================
echo 병리 이미지 추론 워커 시작
echo ========================================
echo.

REM 가상환경 활성화
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo 가상환경 활성화 완료
) else (
    echo 경고: 가상환경이 없습니다. venv를 먼저 생성하세요.
    pause
    exit /b 1
)

REM 환경 변수 확인
if not exist .env (
    echo 경고: .env 파일이 없습니다. env.example을 복사하여 .env를 생성하세요.
    pause
    exit /b 1
)

echo.
echo 환경 설정:
echo GCP 서버: http://34.42.223.43
echo.

REM 워커 실행
echo 워커 시작 중...
python local_inference_worker_http.py

pause

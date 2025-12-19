#!/bin/bash
# Django 에러 로그 확인 스크립트

echo "=== Gunicorn 최근 에러 로그 ==="
sudo journalctl -u gunicorn --since "10 minutes ago" --no-pager | grep -B 10 -A 50 "Traceback\|Error\|Exception\|File.*line"

echo ""
echo "=== Python 패키지 확인 ==="
cd /srv/django-react/app/backend
source .venv/bin/activate
python3 -c "import pydicom; print('pydicom:', pydicom.__version__)"
python3 -c "import nibabel; print('nibabel:', nibabel.__version__)"

echo ""
echo "=== 환경변수 확인 ==="
echo "ORTHANC_URL: ${ORTHANC_URL:-not set}"
echo "ORTHANC_USER: ${ORTHANC_USER:-not set}"
echo "ORTHANC_PASSWORD: ${ORTHANC_PASSWORD:-not set}"

echo ""
echo "=== 최근 업로드 관련 로그 ==="
sudo journalctl -u gunicorn --since "10 minutes ago" --no-pager | grep -i "upload\|orthanc\|nifti"


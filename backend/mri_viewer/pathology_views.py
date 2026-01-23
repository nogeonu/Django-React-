"""
병리 이미지 분류 API 뷰
"""
import os
import logging
import json
import base64
import requests
from pathlib import Path
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

logger = logging.getLogger(__name__)

# Mosec 서비스 URL
PATHOLOGY_MOSEC_URL = os.getenv('PATHOLOGY_MOSEC_URL', 'http://127.0.0.1:5008/inference')

# 교육원 컴퓨터 추론 요청 디렉토리
PATHOLOGY_REQUEST_DIR = Path(os.getenv('PATHOLOGY_INFERENCE_REQUEST_DIR', '/tmp/pathology_inference_requests'))



@api_view(['POST'])
def pathology_ai_analysis(request):
    """
    병리 이미지 AI 분석 (CLAM)
    
    Request Body:
        {
            "instance_id": "Orthanc instance ID (참고용)",
            "filename": "로컬 wsi 폴더 기준 파일명 (예: tumor_076.svs 또는 2024/01/case1.svs)"
        }
    
    Response:
        {
            "success": true,
            "class_id": 1,
            "class_name": "Tumor",
            "confidence": 0.95,
            "probabilities": {
                "Normal": 0.05,
                "Tumor": 0.95
            },
            "num_patches": 856,
            "top_attention_patches": [123, 456, 789, ...]
        }
    """
    try:
        # 요청 데이터 파싱
        instance_id = request.data.get('instance_id')  # 참고용
        filename = request.data.get('filename')  # 필수: 교육원 워커가 사용할 파일명
        
        if not filename:
            return Response(
                {'error': 'filename이 필요합니다. 교육원 워커가 로컬 wsi 폴더에서 찾을 파일명을 입력해주세요.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        logger.info(f"📥 병리 이미지 분석 요청: instance_id={instance_id}, filename={filename}")
        
        # USE_LOCAL_INFERENCE 확인
        use_local_inference = os.getenv('USE_LOCAL_INFERENCE', 'false').lower() == 'true'
        
        if use_local_inference:
            # 교육원 컴퓨터에서 추론 실행
            logger.info("🏠 교육원 컴퓨터 추론 모드 활성화")
            logger.info(f"📁 filename: {filename} (교육원 워커가 wsi/ 폴더에서 찾을 파일)")
            return _create_local_inference_request(request, instance_id, filename)
        
        # Mosec 서비스 호출 (파일 경로만 전달)
        payload = {
            "svs_file_path": original_svs_path
        }
        
        logger.info(f"🚀 Mosec 서비스 호출: {PATHOLOGY_MOSEC_URL}")
        
        response = requests.post(
            PATHOLOGY_MOSEC_URL,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=300  # 5분 타임아웃 (WSI 처리 시간 고려)
        )
        
        if response.status_code != 200:
            logger.error(f"❌ Mosec 서비스 오류: {response.status_code} - {response.text}")
            return Response(
                {'error': f'Mosec 서비스 오류: {response.status_code}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        # 응답 파싱
        mosec_result = response.json()
        logger.info(f"📥 Mosec 응답 내용: {mosec_result}")
        
        # 결과 추출
        if 'results' in mosec_result:
            result = mosec_result['results']
        else:
            result = mosec_result
        
        logger.info(f"✅ 병리 이미지 분석 완료: {result.get('class_name', 'Unknown')}")
        
        return Response(result, status=status.HTTP_200_OK)
        
    except requests.exceptions.Timeout:
        logger.error(f"❌ Mosec 서비스 타임아웃")
        return Response(
            {'error': 'AI 분석 타임아웃 (5분 초과)'},
            status=status.HTTP_504_GATEWAY_TIMEOUT
        )
    except requests.exceptions.ConnectionError:
        logger.error(f"❌ Mosec 서비스 연결 실패")
        return Response(
            {'error': 'Mosec 서비스에 연결할 수 없습니다'},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )
    except Exception as e:
        logger.error(f"❌ 병리 이미지 분석 오류: {str(e)}", exc_info=True)
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


def _create_local_inference_request(request, instance_id, filename):
    """
    교육원 컴퓨터에서 추론 실행 요청 생성 (내부 함수)
    
    Args:
        request: Django request 객체
        instance_id: Orthanc instance ID (참고용)
        filename: 로컬 wsi 폴더 기준 파일명 (예: "tumor_076.svs" 또는 "2024/01/case1.svs")
    """
    try:
        PATHOLOGY_REQUEST_DIR.mkdir(exist_ok=True, parents=True)
        
        request_data = {
            'instance_id': instance_id,  # 참고용
            'filename': filename,  # 교육원 워커가 사용할 파일명
            'requested_at': timezone.now().isoformat(),
            'status': 'pending',
            'requested_by': getattr(request.user, 'username', 'anonymous') if hasattr(request, 'user') and hasattr(request.user, 'is_authenticated') and request.user.is_authenticated else 'anonymous'
        }
        
        timestamp = int(timezone.now().timestamp() * 1000)
        request_id = f"{instance_id}_{timestamp}"
        request_file = PATHOLOGY_REQUEST_DIR / f"{request_id}.json"
        
        with open(request_file, 'w', encoding='utf-8') as f:
            json.dump(request_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 추론 요청 생성: {request_file.name}")
        
        import time
        max_wait_time = 600  # 10분 (추론 시간 고려)
        check_interval = 2
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            time.sleep(check_interval)
            elapsed_time += check_interval
            
            try:
                with open(request_file, 'r', encoding='utf-8') as f:
                    current_data = json.load(f)
                
                current_status = current_data.get('status')
                if current_status == 'completed':
                    result = current_data.get('result', {})
                    return Response({
                        'success': True,
                        'instance_id': instance_id,
                        'request_id': request_id,
                        'class_id': result.get('class_id'),
                        'class_name': result.get('class_name'),
                        'confidence': result.get('confidence'),
                        'probabilities': result.get('probabilities'),
                        'num_patches': result.get('num_patches'),
                        'top_attention_patches': result.get('top_attention_patches', []),
                        'elapsed_time_seconds': result.get('elapsed_time_seconds'),
                        'processed_by': 'local_worker'
                    })
                elif current_status == 'failed':
                    result = current_data.get('result', {})
                    return Response({
                        'success': False,
                        'error': result.get('error', '알 수 없는 오류'),
                        'request_id': request_id
                    }, status=500)
            except:
                pass
        
        return Response({
            'success': False,
            'error': f'요청 처리 시간 초과 (최대 {max_wait_time}초)',
            'request_id': request_id
        }, status=504)
        
    except Exception as e:
        logger.error(f"❌ 추론 요청 생성 실패: {str(e)}", exc_info=True)
        return Response({'success': False, 'error': str(e)}, status=500)


@api_view(['GET'])
def pathology_ai_health(request):
    """병리 AI 서비스 헬스 체크"""
    try:
        response = requests.get(
            "http://localhost:5008/",
            timeout=5
        )
        return Response({
            'status': 'healthy' if response.status_code == 200 else 'unhealthy',
            'mosec_status_code': response.status_code
        })
    except Exception as e:
        return Response({
            'status': 'unhealthy',
            'error': str(e)
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)


# ============================================================
# 교육원 컴퓨터 추론 요청 API
# ============================================================

@api_view(['GET'])
@csrf_exempt
def get_pending_requests(request):
    """
    워커용: 대기 중인 요청 조회
    교육원 조원 요청사항에 맞춘 형식: {"count": 1, "requests": [{"id": 101, "filename": "..."}]}
    """
    try:
        PATHOLOGY_REQUEST_DIR.mkdir(exist_ok=True, parents=True)
        request_files = sorted(PATHOLOGY_REQUEST_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime)
        pending = []
        for rf in request_files:
            with open(rf, 'r', encoding='utf-8') as f:
                d = json.load(f)
                if d.get('status') == 'pending':
                    # 상태를 processing으로 변경하여 중복 할당 방지
                    d['status'] = 'processing'
                    d['started_at'] = timezone.now().isoformat()
                    with open(rf, 'w', encoding='utf-8') as f2:
                        json.dump(d, f2, indent=2, ensure_ascii=False)
                    
                    # 교육원 조원 요청 형식에 맞춤
                    filename = d.get('filename')
                    if not filename:
                        # filename이 없으면 건너뛰기 (필수 필드)
                        logger.warning(f"⚠️ filename이 없는 요청 건너뜀: {rf.stem}")
                        continue
                    
                    pending.append({
                        'id': rf.stem,  # request_id를 id로 사용 (task_id)
                        'filename': filename  # 교육원 워커가 wsi/ 폴더에서 찾을 파일명
                    })
                    break  # 가장 오래된 1개만 반환
        
        return Response({'count': len(pending), 'requests': pending})
    except Exception as e:
        logger.error(f"❌ 대기 중인 요청 조회 실패: {str(e)}", exc_info=True)
        return Response({'count': 0, 'requests': []}, status=500)


@api_view(['POST'])
@csrf_exempt
def update_request_status(request, request_id):
    """
    요청 상태 업데이트
    """
    try:
        request_file = PATHOLOGY_REQUEST_DIR / f"{request_id}.json"
        if not request_file.exists():
            return Response({'success': False, 'error': '요청을 찾을 수 없습니다'}, status=404)
        
        with open(request_file, 'r', encoding='utf-8') as f:
            d = json.load(f)
        
        d['status'] = request.data.get('status', d['status'])
        if request.data.get('started_at'):
            d['started_at'] = request.data.get('started_at')
        
        with open(request_file, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        
        return Response({'success': True})
    except Exception as e:
        logger.error(f"❌ 상태 업데이트 실패: {str(e)}", exc_info=True)
        return Response({'success': False, 'error': str(e)}, status=500)


@api_view(['POST'])
@csrf_exempt
def complete_request(request, request_id):
    """
    추론 완료 결과 업로드 (기존 방식 유지)
    """
    try:
        request_file = PATHOLOGY_REQUEST_DIR / f"{request_id}.json"
        if not request_file.exists():
            return Response({'success': False, 'error': '요청을 찾을 수 없습니다'}, status=404)
        
        with open(request_file, 'r', encoding='utf-8') as f:
            d = json.load(f)
        
        d['status'] = 'completed' if request.data.get('success') else 'failed'
        d['result'] = request.data
        d['completed_at'] = timezone.now().isoformat()
        
        with open(request_file, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 추론 완료 결과 저장: {request_id}")
        return Response({'success': True})
    except Exception as e:
        logger.error(f"❌ 결과 업로드 실패: {str(e)}", exc_info=True)
        return Response({'success': False, 'error': str(e)}, status=500)


@api_view(['POST'])
@csrf_exempt
def complete_task(request):
    """
    교육원 조원 요청 형식: POST /api/pathology/complete/
    Body: {"task_id": 101, "result": "Tumor", "confidence": 0.9923}
    """
    try:
        task_id = request.data.get('task_id')
        if not task_id:
            return Response({'error': 'task_id가 필요합니다'}, status=400)
        
        request_file = PATHOLOGY_REQUEST_DIR / f"{task_id}.json"
        if not request_file.exists():
            return Response({'error': '요청을 찾을 수 없습니다'}, status=404)
        
        with open(request_file, 'r', encoding='utf-8') as f:
            d = json.load(f)
        
        result_label = request.data.get('result')  # "Tumor" or "Normal"
        confidence = request.data.get('confidence', 0.0)
        
        # 결과 형식 변환
        class_id = 1 if result_label == "Tumor" else 0
        class_name = result_label
        probabilities = {
            "Normal": 1.0 - confidence if result_label == "Tumor" else confidence,
            "Tumor": confidence if result_label == "Tumor" else 1.0 - confidence
        }
        
        d['status'] = 'completed'
        d['result'] = {
            'success': True,
            'class_id': class_id,
            'class_name': class_name,
            'confidence': confidence,
            'probabilities': probabilities,
            'num_patches': request.data.get('num_patches', 0),
            'top_attention_patches': request.data.get('top_attention_patches', [])
        }
        d['completed_at'] = timezone.now().isoformat()
        
        with open(request_file, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 추론 완료 결과 저장: {task_id} - {class_name} ({confidence:.4f})")
        return Response({'success': True})
    except Exception as e:
        logger.error(f"❌ 결과 업로드 실패: {str(e)}", exc_info=True)
        return Response({'error': str(e)}, status=500)


@api_view(['POST'])
@csrf_exempt
def fail_task(request):
    """
    교육원 조원 요청 형식: POST /api/pathology/fail/
    Body: {"task_id": 101, "error": "File not found: ..."}
    """
    try:
        task_id = request.data.get('task_id')
        if not task_id:
            return Response({'error': 'task_id가 필요합니다'}, status=400)
        
        request_file = PATHOLOGY_REQUEST_DIR / f"{task_id}.json"
        if not request_file.exists():
            return Response({'error': '요청을 찾을 수 없습니다'}, status=404)
        
        with open(request_file, 'r', encoding='utf-8') as f:
            d = json.load(f)
        
        error_msg = request.data.get('error', '알 수 없는 오류')
        
        d['status'] = 'failed'
        d['result'] = {
            'success': False,
            'error': error_msg
        }
        d['completed_at'] = timezone.now().isoformat()
        
        with open(request_file, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        
        logger.error(f"❌ 추론 실패 처리: {task_id} - {error_msg}")
        return Response({'success': True})
    except Exception as e:
        logger.error(f"❌ 실패 처리 실패: {str(e)}", exc_info=True)
        return Response({'error': str(e)}, status=500)


# ============================================================
# 교육원 컴퓨터 추론 요청 API
# ============================================================

@api_view(['GET'])
@csrf_exempt
def get_pending_requests(request):
    """
    워커용: 대기 중인 요청 조회
    교육원 조원 요청사항에 맞춘 형식: {"count": 1, "requests": [{"id": 101, "filename": "..."}]}
    """
    try:
        PATHOLOGY_REQUEST_DIR.mkdir(exist_ok=True, parents=True)
        request_files = sorted(PATHOLOGY_REQUEST_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime)
        pending = []
        for rf in request_files:
            with open(rf, 'r', encoding='utf-8') as f:
                d = json.load(f)
                if d.get('status') == 'pending':
                    # 상태를 processing으로 변경하여 중복 할당 방지
                    d['status'] = 'processing'
                    d['started_at'] = timezone.now().isoformat()
                    with open(rf, 'w', encoding='utf-8') as f2:
                        json.dump(d, f2, indent=2, ensure_ascii=False)
                    
                    # 교육원 조원 요청 형식에 맞춤
                    filename = d.get('filename')
                    if not filename:
                        # filename이 없으면 건너뛰기 (필수 필드)
                        logger.warning(f"⚠️ filename이 없는 요청 건너뜀: {rf.stem}")
                        continue
                    
                    pending.append({
                        'id': rf.stem,  # request_id를 id로 사용 (task_id)
                        'filename': filename  # 교육원 워커가 wsi/ 폴더에서 찾을 파일명
                    })
                    break  # 가장 오래된 1개만 반환
        
        return Response({'count': len(pending), 'requests': pending})
    except Exception as e:
        logger.error(f"❌ 대기 중인 요청 조회 실패: {str(e)}", exc_info=True)
        return Response({'count': 0, 'requests': []}, status=500)


@api_view(['POST'])
@csrf_exempt
def update_request_status(request, request_id):
    """
    요청 상태 업데이트
    """
    try:
        request_file = PATHOLOGY_REQUEST_DIR / f"{request_id}.json"
        if not request_file.exists():
            return Response({'success': False, 'error': '요청을 찾을 수 없습니다'}, status=404)
        
        with open(request_file, 'r', encoding='utf-8') as f:
            d = json.load(f)
        
        d['status'] = request.data.get('status', d['status'])
        if request.data.get('started_at'):
            d['started_at'] = request.data.get('started_at')
        
        with open(request_file, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        
        return Response({'success': True})
    except Exception as e:
        logger.error(f"❌ 상태 업데이트 실패: {str(e)}", exc_info=True)
        return Response({'success': False, 'error': str(e)}, status=500)


@api_view(['POST'])
@csrf_exempt
def complete_request(request, request_id):
    """
    추론 완료 결과 업로드 (기존 방식 유지)
    """
    try:
        request_file = PATHOLOGY_REQUEST_DIR / f"{request_id}.json"
        if not request_file.exists():
            return Response({'success': False, 'error': '요청을 찾을 수 없습니다'}, status=404)
        
        with open(request_file, 'r', encoding='utf-8') as f:
            d = json.load(f)
        
        d['status'] = 'completed' if request.data.get('success') else 'failed'
        d['result'] = request.data
        d['completed_at'] = timezone.now().isoformat()
        
        with open(request_file, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 추론 완료 결과 저장: {request_id}")
        return Response({'success': True})
    except Exception as e:
        logger.error(f"❌ 결과 업로드 실패: {str(e)}", exc_info=True)
        return Response({'success': False, 'error': str(e)}, status=500)


@api_view(['POST'])
@csrf_exempt
def complete_task(request):
    """
    교육원 조원 요청 형식: POST /api/pathology/complete/
    Body: {"task_id": 101, "result": "Tumor", "confidence": 0.9923}
    """
    try:
        task_id = request.data.get('task_id')
        if not task_id:
            return Response({'error': 'task_id가 필요합니다'}, status=400)
        
        request_file = PATHOLOGY_REQUEST_DIR / f"{task_id}.json"
        if not request_file.exists():
            return Response({'error': '요청을 찾을 수 없습니다'}, status=404)
        
        with open(request_file, 'r', encoding='utf-8') as f:
            d = json.load(f)
        
        result_label = request.data.get('result')  # "Tumor" or "Normal"
        confidence = request.data.get('confidence', 0.0)
        
        # 결과 형식 변환
        class_id = 1 if result_label == "Tumor" else 0
        class_name = result_label
        probabilities = {
            "Normal": 1.0 - confidence if result_label == "Tumor" else confidence,
            "Tumor": confidence if result_label == "Tumor" else 1.0 - confidence
        }
        
        d['status'] = 'completed'
        d['result'] = {
            'success': True,
            'class_id': class_id,
            'class_name': class_name,
            'confidence': confidence,
            'probabilities': probabilities,
            'num_patches': request.data.get('num_patches', 0),
            'top_attention_patches': request.data.get('top_attention_patches', [])
        }
        d['completed_at'] = timezone.now().isoformat()
        
        with open(request_file, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 추론 완료 결과 저장: {task_id} - {class_name} ({confidence:.4f})")
        return Response({'success': True})
    except Exception as e:
        logger.error(f"❌ 결과 업로드 실패: {str(e)}", exc_info=True)
        return Response({'error': str(e)}, status=500)


@api_view(['POST'])
@csrf_exempt
def fail_task(request):
    """
    교육원 조원 요청 형식: POST /api/pathology/fail/
    Body: {"task_id": 101, "error": "File not found: ..."}
    """
    try:
        task_id = request.data.get('task_id')
        if not task_id:
            return Response({'error': 'task_id가 필요합니다'}, status=400)
        
        request_file = PATHOLOGY_REQUEST_DIR / f"{task_id}.json"
        if not request_file.exists():
            return Response({'error': '요청을 찾을 수 없습니다'}, status=404)
        
        with open(request_file, 'r', encoding='utf-8') as f:
            d = json.load(f)
        
        error_msg = request.data.get('error', '알 수 없는 오류')
        
        d['status'] = 'failed'
        d['result'] = {
            'success': False,
            'error': error_msg
        }
        d['completed_at'] = timezone.now().isoformat()
        
        with open(request_file, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        
        logger.error(f"❌ 추론 실패 처리: {task_id} - {error_msg}")
        return Response({'success': True})
    except Exception as e:
        logger.error(f"❌ 실패 처리 실패: {str(e)}", exc_info=True)
        return Response({'error': str(e)}, status=500)


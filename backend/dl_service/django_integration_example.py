"""
Django에서 딥러닝 서비스 호출 예시
이 코드를 참고하여 실제 views.py에 통합하세요.
"""
import os
import requests
from django.conf import settings

# 딥러닝 서비스 URL (환경 변수 또는 settings.py에서 설정)
DL_SERVICE_URL = os.environ.get('DL_SERVICE_URL', 'http://localhost:5003')

def call_dl_service_for_prediction(image_data, patient_id=None, metadata=None):
    """
    딥러닝 서비스를 호출하여 예측 수행
    
    Args:
        image_data: 이미지 데이터 (numpy array, list, 또는 base64 인코딩된 문자열)
        patient_id: 환자 ID (선택)
        metadata: 추가 메타데이터 (선택)
    
    Returns:
        dict: 예측 결과
    """
    try:
        # 이미지 데이터를 리스트로 변환 (필요시)
        if hasattr(image_data, 'tolist'):
            image_list = image_data.tolist()
        elif isinstance(image_data, list):
            image_list = image_data
        else:
            # base64 인코딩된 문자열인 경우
            image_list = image_data
        
        # 요청 데이터 구성
        request_data = {
            "image_data": image_list,
            "patient_id": patient_id,
            "metadata": metadata or {}
        }
        
        # 딥러닝 서비스 호출 (mosec은 /inference 엔드포인트 사용)
        response = requests.post(
            f'{DL_SERVICE_URL}/inference',
            json=request_data,
            timeout=30  # 딥러닝 추론은 시간이 걸릴 수 있음
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                'error': f'딥러닝 서비스 오류: {response.status_code}',
                'detail': response.text
            }
            
    except requests.exceptions.RequestException as e:
        return {
            'error': f'딥러닝 서비스 연결 실패: {str(e)}'
        }
    except Exception as e:
        return {
            'error': f'예측 중 오류 발생: {str(e)}'
        }


def call_dl_service_batch_predict(requests_list):
    """
    배치 예측 수행 (여러 이미지를 한 번에 처리)
    
    Args:
        requests_list: 예측 요청 리스트
    
    Returns:
        dict: 배치 예측 결과
    """
    try:
        # mosec은 자동 배칭을 지원하므로 단일 요청으로 여러 데이터 전송 가능
        # 또는 여러 요청을 순차적으로 전송
        response = requests.post(
            f'{DL_SERVICE_URL}/inference',
            json=requests_list[0] if requests_list else {},  # 첫 번째 요청만 처리 (배치는 mosec이 자동 처리)
            timeout=60  # 배치 처리이므로 더 긴 타임아웃
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                'error': f'배치 예측 오류: {response.status_code}',
                'detail': response.text
            }
            
    except Exception as e:
        return {
            'error': f'배치 예측 중 오류: {str(e)}'
        }


# Django views.py에서 사용 예시:
"""
from .dl_service.django_integration_example import call_dl_service_for_prediction
from rest_framework.decorators import action
from rest_framework.response import Response
import numpy as np
from PIL import Image
import io

class YourViewSet(viewsets.ModelViewSet):
    @action(detail=False, methods=['post'])
    def predict_with_dl(self, request):
        '''딥러닝 모델을 사용한 예측'''
        # 이미지 파일 받기
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({'error': '이미지 파일이 필요합니다.'}, status=400)
        
        # 이미지 전처리
        image = Image.open(image_file)
        image = image.resize((224, 224))  # 모델 입력 크기에 맞게 조정
        image_array = np.array(image) / 255.0  # 정규화
        image_list = image_array.flatten().tolist()
        
        # 딥러닝 서비스 호출
        result = call_dl_service_for_prediction(
            image_data=image_list,
            patient_id=request.data.get('patient_id'),
            metadata={'original_size': image.size}
        )
        
        if 'error' in result:
            return Response(result, status=500)
        
        return Response(result, status=200)
"""


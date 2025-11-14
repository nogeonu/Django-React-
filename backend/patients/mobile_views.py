"""
모바일 앱 관련 API
- 최신 APK 정보 제공
- 다운로드 통계
"""
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
import requests
from django.conf import settings


@api_view(['GET'])
@permission_classes([AllowAny])
def get_latest_apk_info(request):
    """
    GitHub Releases에서 최신 APK 정보를 가져옵니다.
    
    GET /api/mobile/latest-apk/
    
    Response:
    {
        "version": "1.0.0",
        "build_number": 123,
        "download_url": "https://github.com/.../app.apk",
        "release_notes": "변경사항...",
        "published_at": "2025-11-14T12:00:00Z",
        "file_size": 12345678
    }
    """
    try:
        # GitHub API로 최신 릴리스 정보 가져오기
        github_repo = getattr(settings, 'FLUTTER_GITHUB_REPO', 'nogeonu/flutter-mobile')
        api_url = f'https://api.github.com/repos/{github_repo}/releases/latest'
        
        response = requests.get(api_url, timeout=10)
        
        if response.status_code != 200:
            return Response(
                {
                    'error': 'GitHub API 호출 실패',
                    'detail': '최신 릴리스 정보를 가져올 수 없습니다.'
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        
        release_data = response.json()
        
        # APK 파일 찾기
        apk_asset = None
        for asset in release_data.get('assets', []):
            if asset['name'].endswith('.apk'):
                apk_asset = asset
                break
        
        if not apk_asset:
            return Response(
                {
                    'error': 'APK 파일을 찾을 수 없습니다.',
                    'detail': '릴리스에 APK 파일이 없습니다.'
                },
                status=status.HTTP_404_NOT_FOUND
            )
        
        # 버전 정보 파싱 (태그에서 추출: v1.0.0-123 -> version: 1.0.0, build: 123)
        tag_name = release_data.get('tag_name', '')
        version_parts = tag_name.lstrip('v').split('-')
        version = version_parts[0] if len(version_parts) > 0 else '0.0.0'
        build_number = version_parts[1] if len(version_parts) > 1 else '0'
        
        return Response({
            'version': version,
            'build_number': build_number,
            'download_url': apk_asset['browser_download_url'],
            'release_notes': release_data.get('body', ''),
            'published_at': release_data.get('published_at'),
            'file_size': apk_asset.get('size', 0),
            'file_name': apk_asset.get('name', 'app.apk'),
            'download_count': apk_asset.get('download_count', 0),
        })
    
    except requests.RequestException as e:
        return Response(
            {
                'error': '네트워크 오류',
                'detail': str(e)
            },
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )
    except Exception as e:
        return Response(
            {
                'error': '서버 오류',
                'detail': str(e)
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([AllowAny])
def record_apk_download(request):
    """
    APK 다운로드 통계 기록
    
    POST /api/mobile/download-stats/
    Body: {
        "version": "1.0.0",
        "user_agent": "..."
    }
    """
    # TODO: 다운로드 통계를 DB에 저장
    # 나중에 통계 분석에 활용 가능
    return Response({
        'message': '다운로드가 기록되었습니다.'
    })


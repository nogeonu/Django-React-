"""
GitHub API Proxy Views
"""
import requests
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.cache import cache_page


@require_http_methods(["GET"])
@cache_page(60 * 5)  # 5분 캐시
def get_latest_release(request):
    """
    GitHub의 최신 릴리즈 정보를 프록시로 가져옵니다.
    """
    owner = "nogeonu"
    repo = "flutter-mobile"
    
    try:
        url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "Konyang-Hospital-App",
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            # 필요한 정보만 필터링
            filtered_data = {
                "tag_name": data.get("tag_name", ""),
                "name": data.get("name", ""),
                "published_at": data.get("published_at", ""),
                "body": data.get("body", ""),
                "assets": [
                    {
                        "name": asset.get("name", ""),
                        "browser_download_url": asset.get("browser_download_url", ""),
                        "size": asset.get("size", 0),
                    }
                    for asset in data.get("assets", [])
                ],
            }
            return JsonResponse(filtered_data, status=200)
        elif response.status_code == 404:
            return JsonResponse(
                {
                    "error": "릴리즈를 찾을 수 없습니다.",
                    "message": "GitHub 저장소에 릴리즈가 아직 생성되지 않았습니다.",
                    "tag_name": "",
                    "name": "",
                    "published_at": "",
                    "body": "",
                    "assets": [],
                },
                status=200  # 404가 아닌 200으로 반환하여 프론트엔드에서 처리
            )
        else:
            return JsonResponse(
                {"error": f"GitHub API 오류: {response.status_code}"},
                status=response.status_code
            )
    
    except requests.exceptions.Timeout:
        return JsonResponse(
            {"error": "요청 시간이 초과되었습니다."},
            status=504
        )
    except requests.exceptions.RequestException as e:
        return JsonResponse(
            {"error": f"네트워크 오류가 발생했습니다: {str(e)}"},
            status=500
        )
    except Exception as e:
        return JsonResponse(
            {"error": f"서버 오류가 발생했습니다: {str(e)}"},
            status=500
        )

